#!/usr/bin/env python3
"""First lightweight history intervention utility pilot, not new CSS training.

For each outer fold, use ONLY its own training sessions to fit selectors.
Never pool other folds' OOF rows for selector training (their CSS backbones
have seen the outer held-out session). Reuse its existing held-out masking
rows for evaluation. Training-side backbone predictions are in-sample,
which may make learning utility difficult; this is an exploratory pilot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataloader import IEMOCAPDataset
from research.session_confusion_stability import SESSIONS, CLASSES, PAIRS
from research.history_masking_audit import matched_positions
from research.history_intervention_utility import (
    ARMS, feature_matrix, fit_selectors, predict_selectors, metrics,
)
from scripts.run_history_masking_audit import read_inputs, infer_masks
from scripts.run_session_confusion_stability import sha256_file
from scripts.run_v12_author_protocol import create_model

ROOT = Path("results/css_session_confusion_stability/seed61078")
PAIR_IDS = {frozenset((CLASSES.index(a), CLASSES.index(b))) for a,b in PAIRS}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_checkpoint(root, report, session, device):
    path = root / f"{session}_epoch200.pt"
    if sha256_file(path) != report["checkpoint_sha256"]:
        raise ValueError("checkpoint fingerprint mismatch")
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (cp.get("epoch") != 200 or cp.get("seed") != 61078 or
            cp.get("held_out_session") != session or
            cp.get("training_sessions") != [s for s in SESSIONS if s != session] or
            cp.get("selection") != "fixed_epoch_200_no_heldout_selection"):
        raise ValueError("checkpoint protocol mismatch")
    model = create_model("original").to(device)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    model.eval()
    return model


def collect_training_rows(args, ds, groups, model, heldout, device):
    rows = []
    for session in SESSIONS:
        if session == heldout:
            continue
        for idx in groups[session]:
            vid = str(ds.keys[idx])
            if not vid.startswith(session):
                raise ValueError("training-session identity mismatch")
            positions = matched_positions(ds.videoSpeakers[vid])
            if not positions:
                continue
            item = ds[idx]
            y = np.asarray(ds.videoLabels[vid], dtype=int)
            result = infer_masks(model, item, [None] + sorted({
                j for i,j,k,h in positions}), device, args.batch_size)
            base = result[None]
            for i,j,k,h in positions:
                b, m = base[i], result[j][i]
                rows.append({
                    "dialogue": vid, "index": i, "previous_own_index": j,
                    "true": int(y[i]), "previous_own_true": int(y[j]),
                    "prior_pred": int(base[j].argmax()),
                    "baseline_prob": b.tolist(), "baseline_pred": int(b.argmax()),
                    "previous_own_prob": m.tolist(), "previous_own_pred": int(m.argmax()),
                })
            if args.smoke:
                feature_matrix(rows, history=True)
                print("SMOKE PASS: training-side frozen inference and label-free "
                      f"features, {len(rows)} targets; no selector fit or held-out scoring.")
                return rows
        print(f"TRAIN session {session} for outer {heldout}: "
              f"cumulative matched={len(rows)}", flush=True)
    if not rows:
        raise ValueError("no training rows")
    return rows


def read_heldout(args, ds, groups, report, session):
    path = args.input_dir / "history_masking" / f"{session}.json"
    obj = read_json(path)
    p = obj["protocol"]
    if (p.get("held_out_session") != session or
            p.get("checkpoint_sha256") != report["checkpoint_sha256"] or
            p.get("seed") != 61078 or p.get("epoch") != 200 or
            p.get("full_fold") is not True or
            p.get("training") is not False or
            p.get("official_test_scored") is not False or
            p.get("baseline_argmax_verified") is not True or
            p.get("mask_selection_uses_gold") is not False):
        raise ValueError("invalid frozen masking report")
    raw = {(r["dialogue"], r["index"]): r for r in obj["rows"]}
    if len(raw) != len(obj["rows"]):
        raise ValueError("duplicate held-out masking rows")
    rows, indices, offset = [], [], 0
    for idx in groups[session]:
        vid = str(ds.keys[idx])
        y = ds.videoLabels[vid]
        b = report["predictions"][offset:offset+len(y)]
        for i,j,k,h in matched_positions(ds.videoSpeakers[vid]):
            r = dict(raw.pop((vid,i)))
            if (r["session"] != session or r["previous_own_index"] != j or
                    r["near_other_index"] != k or r["earlier_own_index"] != h or
                    r["true"] != int(y[i]) or r["previous_own_true"] != int(y[j]) or
                    r["baseline_pred"] != int(b[i])):
                raise ValueError("held-out masking row mismatch")
            r["prior_pred"] = int(b[j])
            rows.append(r)
            indices.append(offset+i)
        offset += len(y)
    if raw or offset != len(report["labels"]):
        raise ValueError("held-out masking coverage mismatch")
    feature_matrix(rows, history=True)
    return rows, np.asarray(indices, dtype=int)


def evaluate(y, b, outputs, indices, previous):
    changed = y[indices] != previous
    pair = np.array([frozenset((int(a),int(c))) in PAIR_IDS
                     for a,c in zip(y[indices],previous)], dtype=bool) & changed
    groups = {
        "all": np.arange(len(y)), "matched": indices,
        "changed": indices[changed], "stable": indices[~changed],
        "three_pair_transitions": indices[pair],
    }
    return {group: {arm: metrics(y[ix], b[ix], outputs[arm][ix])
                    for arm in ARMS} for group,ix in groups.items()}


def print_scores(summary):
    for group in ("all", "changed", "stable", "three_pair_transitions"):
        print("\n" + group)
        for arm in ARMS:
            m = summary[group][arm]
            if not m["n"]:
                continue
            print(f"  {arm:20s} n={m['n']} W-F1={m['weighted_f1_pct']:.3f}% "
                  f"dF1={m['weighted_f1_delta_pp']:+.3f}pp "
                  f"dACC={m['accuracy_delta_pp']:+.3f}pp "
                  f"fixed/broken={m['fixed']}/{m['broken']}")


def run_fold(args, ds, groups, reports, session):
    path = args.output_dir / f"{session}.json"
    if not args.smoke and path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    report = reports[session]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = load_checkpoint(args.input_dir, report, session, device)
    train = collect_training_rows(args, ds, groups, model, session, device)
    if args.smoke:
        return
    # Fit before reading any held-out mask rows or using their labels.
    selectors, diagnostics = fit_selectors(train)
    del train, model
    print("Selector training:", diagnostics, flush=True)

    rows, ix = read_heldout(args, ds, groups, report, session)
    # Strip gold from selector inference explicitly.
    features = [{k:v for k,v in r.items() if k not in
                 ("true", "previous_own_true")} for r in rows]
    chosen = predict_selectors(selectors, features)
    y = np.asarray(report["labels"], dtype=int)
    b = np.asarray(report["predictions"], dtype=int)
    outputs = {arm: b.copy() for arm in ARMS}
    for arm in ARMS:
        outputs[arm][ix] = chosen[arm]
    prev = np.asarray([r["previous_own_true"] for r in rows], dtype=int)
    summary = evaluate(y,b,outputs,ix,prev)
    result = {
        "protocol": {
            "outer_heldout": session,
            "selector_train_sessions": [s for s in SESSIONS if s != session],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "backbone_frozen": True, "training_predictions_in_sample": True,
            "official_test_scored": False, "selector_gold_inputs": False,
            "fixed_thresholds": True,
        },
        "training_diagnostics": diagnostics,
        "summary": summary,
        "predictions": {a:p.astype(int).tolist() for a,p in outputs.items()},
    }
    write_new(path, result)
    print_scores(summary)
    print("Saved:", path)


def aggregate(args, ds, groups, reports):
    path = args.output_dir / "summary.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    yy, bb, ix_all, prev_all, outputs = [], [], [], [], {a:[] for a in ARMS}
    per_session, offset = {}, 0
    for session in SESSIONS:
        report = reports[session]
        obj = read_json(args.output_dir / f"{session}.json")
        protocol = obj["protocol"]
        if (protocol.get("outer_heldout") != session or
                protocol.get("selector_train_sessions") != [s for s in SESSIONS if s != session] or
                protocol.get("checkpoint_sha256") != report["checkpoint_sha256"] or
                protocol.get("backbone_frozen") is not True or
                protocol.get("training_predictions_in_sample") is not True or
                protocol.get("official_test_scored") is not False or
                protocol.get("selector_gold_inputs") is not False or
                protocol.get("fixed_thresholds") is not True):
            raise ValueError(f"selector protocol mismatch: {session}")
        rows, ix = read_heldout(args, ds, groups, report, session)
        y = np.asarray(report["labels"], dtype=int)
        b = np.asarray(report["predictions"], dtype=int)
        masked = b.copy()
        masked[ix] = [r["previous_own_pred"] for r in rows]
        if set(obj["predictions"]) != set(ARMS):
            raise ValueError("missing selector arms")
        pp = {a:np.asarray(obj["predictions"][a], dtype=int) for a in ARMS}
        for a,p in pp.items():
            if p.shape != b.shape or np.any((p != b) & (p != masked)):
                raise ValueError("selector output is not an allowed candidate")
            outputs[a].append(p)
        if not np.array_equal(pp["baseline"], b) or not np.array_equal(pp["always_mask"], masked):
            raise ValueError("baseline / all-mask control mismatch")
        prev = np.asarray([r["previous_own_true"] for r in rows], dtype=int)
        per_session[session] = evaluate(y,b,pp,ix,prev)
        yy.append(y)
        bb.append(b)
        ix_all.append(ix+offset)
        prev_all.append(prev)
        offset += len(y)
    y,b = np.concatenate(yy),np.concatenate(bb)
    indices,previous = np.concatenate(ix_all),np.concatenate(prev_all)
    if len(y) != 5810 or len(indices) != 5502 or int((y[indices]!=previous).sum()) != 1597:
        raise ValueError("unexpected pooled coverage")
    summary = evaluate(y,b,{a:np.concatenate(v) for a,v in outputs.items()},indices,previous)
    u = summary["all"]["history_utility"]
    comparison = {a: u["weighted_f1_pct"]-summary["all"][a]["weighted_f1_pct"]
                  for a in ARMS if a!="history_utility"}
    positive_sessions = sum(per_session[s]["all"]["history_utility"]["weighted_f1_delta_pp"] > 0
                            for s in SESSIONS)
    # Frozen exploratory screen, not a statistical or novelty claim.
    checks = {
        "overall_wf1_gain_at_least_0p5": comparison["baseline"] >= .5,
        "beats_probability_and_shift_by_0p2": min(comparison["probability_utility"],
                                                  comparison["shift_only"]) >= .2,
        "beats_max_confidence": comparison["max_confidence"] > 0,
        "positive_sessions_at_least_3": positive_sessions >= 3,
        "changed_accuracy_positive": summary["changed"]["history_utility"]["accuracy_delta_pp"] > 0,
        "stable_accuracy_drop_at_most_0p25": summary["stable"]["history_utility"]["accuracy_delta_pp"] >= -.25,
    }
    result = {"pooled": summary, "per_session": per_session,
              "history_utility_wf1_deltas_pp": comparison,
              "positive_sessions": positive_sessions, "screen_checks": checks,
              "screen_pass": all(checks.values()), "exploratory": True,
              "official_test_scored": False,
              "limits": ["Selector training features from in-sample CSS predictions.",
                         "Previously examined outer sessions; not independent confirmation.",
                         "Two forward variants per target; not an efficient final architecture."]}
    write_new(path,result)
    print_scores(summary)
    print("\nHistory utility W-F1 differences:", comparison)
    print("Positive sessions:", positive_sessions, "/ 4")
    print("Screen checks:", checks)
    print("SCREEN:", "PASS" if result["screen_pass"] else "NO-GO")
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fold", choices=SESSIONS)
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"history_utility")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    ds = IEMOCAPDataset(train=True)
    _, groups, reports = read_inputs(args.input_dir, ds)
    if args.aggregate:
        aggregate(args,ds,groups,reports)
    else:
        run_fold(args,ds,groups,reports,args.fold or "Ses01")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen CSS history-key masking audit; official TRAIN-only four folds.

No training, parameter updates, or official Session5 TEST evaluation.
For each dialogue, cache the output of masking each requested history key
once and reuse it across targets. Current utterance and positional/speaker
embeddings are unchanged; no gold labels enter the forward pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataloader import IEMOCAPDataset
from scripts.run_v12_author_protocol import create_model
from scripts.run_session_confusion_stability import sha256_file
from research.session_confusion_stability import (
    SESSIONS, split_by_session, validate_train_only, aggregate_folds,
)
from research.history_masking_audit import matched_positions, summarize

DEFAULT_INPUT = Path("results/css_session_confusion_stability/seed61078")
DEFAULT_OUTPUT = DEFAULT_INPUT / "history_masking"


def read_inputs(root, ds):
    meta = validate_train_only(ds.keys, ds.videoLabels)
    groups = split_by_session(ds.keys)
    reports = {}
    for s in SESSIONS:
        f = json.loads((root / f"{s}.json").read_text(encoding="utf-8"))
        protocol = f["protocol"]
        if (protocol.get("official_train_only") is not True or
                protocol.get("official_test_scored") is not False or
                protocol.get("original_css_pgm_spf") is not True or
                protocol.get("epochs") != 200 or protocol.get("seed") != 61078 or
                protocol.get("training_sessions") != [x for x in SESSIONS if x != s]):
            raise ValueError(f"incompatible fold {s}")
        ids = [str(ds.keys[i]) for i in groups[s]]
        lengths = [len(ds.videoLabels[k]) for k in ids]
        if ids != f["held_out_dialogue_ids"] or lengths != f["held_out_dialogue_lengths"]:
            raise ValueError(f"dialogue identity/order/length mismatch: {s}")
        y = np.concatenate([np.asarray(ds.videoLabels[k]) for k in ids])
        if not np.array_equal(y, np.asarray(f["labels"])):
            raise ValueError(f"gold label alignment mismatch: {s}")
        reports[s] = f
    pooled = aggregate_folds(reports)
    original = json.loads((root / "cross_session_summary.json").read_text(encoding="utf-8"))
    if pooled["pooled_out_of_fold"] != original["pooled_out_of_fold"]:
        raise ValueError("frozen OOF summary mismatch")
    if original.get("checkpoint_sha256_by_fold") != {
        s: reports[s]["checkpoint_sha256"] for s in SESSIONS
    }:
        raise ValueError("checkpoint provenance mismatch")
    return meta, groups, reports


@torch.no_grad()
def infer_masks(model, item, positions, device, batch_size=8):
    """None = unmasked; integer = globally exclude that attention key.

    Preserve original dialogue length, current query, features and speaker
    embeddings. Use separate rows for interventions; no weight changes.
    Original CSS attention receives u_mask only as key validity.
    """
    text, visual, audio, speakers, valid, _, _ = item
    length = len(valid)
    if not bool(torch.all(valid == 1)):
        raise ValueError("expected one complete unpadded dialogue")
    if any(p is not None and not 0 <= p < length for p in positions):
        raise ValueError("mask position out of range")
    text, visual, audio, speakers = [
        x.to(device) for x in (text, visual, audio, speakers)
    ]
    results = {}
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start+batch_size]
        b = len(chunk)
        u_mask = torch.ones(b, length, device=device)
        for j, pos in enumerate(chunk):
            if pos is not None:
                u_mask[j, pos] = 0.0
        def expand(x):
            return x.unsqueeze(1).expand(-1, b, -1).contiguous()
        qmask = speakers.unsqueeze(0).expand(b, -1, -1).contiguous()
        out = model(expand(text), expand(visual), expand(audio),
                    u_mask, qmask, [length]*b)[4]
        if out.shape != (b, length, 6) or not torch.isfinite(out).all():
            raise RuntimeError(f"bad model probabilities: {out.shape}")
        if not torch.allclose(out.sum(-1), torch.ones_like(out[...,0]), atol=2e-4, rtol=0):
            raise RuntimeError("probability rows are not normalized")
        for j, pos in enumerate(chunk):
            results[pos] = out[j].detach().cpu().numpy()
    return results


def run_fold(args, ds, groups, reports, s):
    output = args.output_dir / f"{s}.json"
    if output.exists() and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    f = reports[s]
    path = args.input_dir / f"{s}_epoch200.pt"
    if sha256_file(path) != f["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {s}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["epoch"] != 200 or checkpoint["seed"] != 61078 or
            checkpoint["held_out_session"] != s or
            checkpoint["training_sessions"] != [x for x in SESSIONS if x != s] or
            checkpoint["selection"] != "fixed_epoch_200_no_heldout_selection"):
        raise ValueError(f"checkpoint protocol mismatch: {s}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = create_model("original").to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval()

    rows, offset, total_expected = [], 0, 0
    for number, ds_index in enumerate(groups[s], 1):
        item = ds[ds_index]
        labels = ds.videoLabels[ds.keys[ds_index]]
        vid = str(ds.keys[ds_index])
        length = len(labels)
        expected = np.asarray(f["predictions"][offset:offset+length])
        offset += length
        matched = matched_positions(ds.videoSpeakers[vid])
        total_expected += len(matched)
        # Verify the ORIGINAL per-dialogue argmax against saved OOF baseline.
        original = infer_masks(model, item, [None], device, args.batch_size)[None]
        if not np.array_equal(original.argmax(-1), expected):
            raise RuntimeError(f"baseline argmax does not reproduce saved OOF: {vid}")
        if args.smoke:
            matched = matched[:4]
        if not matched:
            continue
        positions = sorted({p for record in matched for p in record[1:] if p is not None})
        masked = infer_masks(model, item, positions, device, args.batch_size)
        for i,j,k,h in matched:
            row = {
                "dialogue": vid, "session": s, "index": i,
                "previous_own_index": j, "near_other_index": k,
                "earlier_own_index": h,
                "true": int(labels[i]), "previous_own_true": int(labels[j]),
            }
            variants = {"baseline": original[i], "previous_own": masked[j][i],
                        "near_other": masked[k][i]}
            if h is not None:
                variants["earlier_own"] = masked[h][i]
            for arm, p in variants.items():
                row[arm+"_prob"] = p.astype(float).tolist()
                row[arm+"_pred"] = int(p.argmax())
            rows.append(row)
        print(f"{s}: dialogue {number}/{len(groups[s])} "
              f"matched targets so far={len(rows)}", flush=True)
        if args.smoke:
            result = summarize(rows)
            print("SMOKE: verified original OOF argmax and past-only masks;",
                  "matched targets =", result["matched_n"])
            return
    if args.smoke:
        raise RuntimeError("no eligible dialogue found for smoke test")
    if offset != len(f["labels"]) or len(rows) != total_expected:
        raise RuntimeError("incomplete fold coverage")
    result = {
        "protocol": {
            "held_out_session": s, "checkpoint_sha256": f["checkpoint_sha256"],
            "seed": 61078, "epoch": 200, "official_test_scored": False,
            "training": False, "baseline_argmax_verified": True,
            "mask_selection_uses_gold": False,
            "full_fold": True, "total_held_out_utterances": offset,
        },
        "summary": summarize(rows), "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print("Saved:", output, flush=True)


def print_group(name, group):
    print(f"\n{name}")
    for arm, v in group.items():
        if not v["n"]:
            print(f"  {arm}: no comparable samples")
            continue
        print(f"  {arm}: n={v['n']} ACC_delta={v['accuracy_delta_pp']:+.3f}pp "
              f"fixed/broken={v['fixed']}/{v['broken']} "
              f"net_history_errors_removed={v['net_reduction_wrong_previous_emotion']} "
              f"margin_delta={v['mean_true_vs_previous_probability_margin_change']}")


def aggregate(args, ds, groups, reports):
    out = args.output_dir / "summary.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    rows, per_session = [], {}
    for s in SESSIONS:
        f = json.loads((args.output_dir / f"{s}.json").read_text(encoding="utf-8"))
        p = f["protocol"]
        if (p["held_out_session"] != s or p["checkpoint_sha256"] != reports[s]["checkpoint_sha256"]
                or p["seed"] != 61078 or p["epoch"] != 200
                or p["official_test_scored"] is not False or p["training"] is not False
                or p["mask_selection_uses_gold"] is not False
                or p["baseline_argmax_verified"] is not True or p["full_fold"] is not True):
            raise ValueError(f"incompatible audit fold: {s}")
        # Reconstruct all expected target/control indices and OOF labels.
        expected, offset = {}, 0
        for ds_index in groups[s]:
            vid = str(ds.keys[ds_index])
            labels = ds.videoLabels[vid]
            for i,j,k,h in matched_positions(ds.videoSpeakers[vid]):
                expected[(vid,i)] = (j,k,h,int(labels[i]),int(labels[j]),
                                     int(reports[s]["predictions"][offset+i]))
            offset += len(labels)
        observed = {}
        for r in f["rows"]:
            key = (r["dialogue"],r["index"])
            if key in observed or r["session"] != s:
                raise ValueError("duplicate or foreign audit target")
            observed[key] = (r["previous_own_index"],r["near_other_index"],r["earlier_own_index"],
                             r["true"],r["previous_own_true"],r["baseline_pred"])
        if observed != expected:
            raise ValueError(f"target/control alignment or coverage mismatch: {s}")
        per_session[s] = summarize(f["rows"])
        rows.extend(f["rows"])
    result = {
        "protocol": {
            "official_test_scored": False, "training": False,
            "candidate_pairs_previously_selected": True,
            "interpretation": "model attention-mask sensitivity, not human causality",
        },
        "pooled": summarize(rows), "per_session": per_session,
    }
    with out.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    pooled = result["pooled"]
    print(f"TRAIN-only matched targets: {pooled['matched_n']}")
    print("Control mean lags (previous own, near other):",
          pooled["mean_previous_own_lag"], pooled["mean_near_other_lag"])
    for name, group in pooled["primary_groups"].items():
        print_group(name, group)
    print_group("earlier-own controlled changed subset",
                pooled["earlier_own_control"]["changed"])
    for s in SESSIONS:
        print_group(s + " changed", per_session[s]["primary_groups"]["changed"])
    print("Exploratory mask intervention, not a deployable correction or causal claim.")
    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fold", choices=SESSIONS)
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    ds = IEMOCAPDataset(train=True)
    _, groups, reports = read_inputs(args.input_dir, ds)
    if args.aggregate:
        aggregate(args, ds, groups, reports)
    else:
        run_fold(args, ds, groups, reports, args.fold or "Ses01")


if __name__ == "__main__":
    main()

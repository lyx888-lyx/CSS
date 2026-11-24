#!/usr/bin/env python3
"""Matched CSS EII-SCL transfer: TRAIN-only, fixed epoch 200.

Reuse the original 4-fold raw CSS reference (seed61078). Train three loss
arms from identical CSS initialization and shuffle seeds. No held-out epoch
selection or official Session5 evaluation. The dynamic arm uses fixed
identity projections, NOT a claim of exact official EII-SCL reproduction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

from dataloader import IEMOCAPDataset
import train_IEMOCAP as trainer
from research.session_confusion_stability import (
    SESSIONS, CLASSES, fold_indices, summarize_fold,
)
from research.context_confusion_association import normalize_speakers
from research.eii_scl_transfer import (
    ARMS, TAU, ALPHA, FIXED_WINDOW, SupervisedContrastHook,
)
from scripts.run_history_masking_audit import read_inputs
from scripts.run_session_confusion_stability import (
    setup_train, sha256_file, FINAL_EPOCH, BATCH_SIZE,
)

ROOT = Path("results/css_session_confusion_stability/seed61078")
OUTPUT = Path("results/css_eii_scl_transfer/seed61078")
SEED = 61078
PILOT = "Ses01"
BASELINE = "baseline"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def verify_baseline(root, reports, session):
    r = reports[session]
    p = r["protocol"]
    if (p.get("batch_size") != BATCH_SIZE or
            p.get("optimizer") != "Adam lr=1e-4 wd=1e-5" or
            p.get("checkpoint_selection") != "fixed_epoch_200_no_heldout_selection"):
        raise ValueError("baseline protocol differs from transfer arms")
    path = root / f"{session}_epoch200.pt"
    if not path.is_file() or sha256_file(path) != r["checkpoint_sha256"]:
        raise ValueError(f"baseline checkpoint missing or changed: {session}")


def shift_groups(ds, report):
    changed, stable, first = [], [], []
    offset = 0
    for vid, n in zip(report["held_out_dialogue_ids"],
                      report["held_out_dialogue_lengths"]):
        spk = normalize_speakers(ds.videoSpeakers[vid])
        y = ds.videoLabels[vid]
        if len(spk) != n or len(y) != n:
            raise ValueError("history subgroup alignment mismatch")
        last = {}
        for i, s in enumerate(spk):
            if s not in last:
                first.append(offset+i)
            elif y[i] != y[last[s]]:
                changed.append(offset+i)
            else:
                stable.append(offset+i)
            last[s] = i
        offset += n
    if offset != len(report["labels"]):
        raise ValueError("history groups do not cover baseline")
    return {name: np.asarray(ix, dtype=int) for name, ix in (
        ("changed", changed), ("stable", stable), ("first", first))}


def metric(y, p, b):
    if not len(y):
        return {"n": 0}
    fixed = int(((b != y) & (p == y)).sum())
    broken = int(((b == y) & (p != y)).sum())
    return {
        "n": len(y),
        "accuracy_pct": 100 * float(accuracy_score(y,p)),
        "weighted_f1_pct": 100 * float(f1_score(y,p,average="weighted",zero_division=0)),
        "macro_f1_pct": 100 * float(f1_score(y,p,labels=list(range(6)),average="macro",zero_division=0)),
        "fixed": fixed, "broken": broken,
        "accuracy_delta_pp": 100 * (fixed-broken)/len(y),
        "weighted_f1_delta_pp": 100 * float(
            f1_score(y,p,average="weighted",zero_division=0) -
            f1_score(y,b,average="weighted",zero_division=0)),
    }


def summarize(y, pred, baseline, groups):
    indices = np.arange(len(y))
    out = {
        "all": metric(y, pred, baseline),
        **{name: metric(y[ix],pred[ix],baseline[ix]) for name, ix in groups.items()},
        "confusion": summarize_fold(y,pred),
        "per_class_f1_pct": {
            name: float(v) for name,v in zip(CLASSES,100*f1_score(
                y,pred,labels=list(range(6)),average=None,zero_division=0))
        },
    }
    return out


def print_scores(name, obj):
    print(name, flush=True)
    for group in ("all","changed","stable","first"):
        v = obj[group]
        if v["n"]:
            print(f"  {group:8s} N={v['n']} ACC={v['accuracy_pct']:.3f}% "
                  f"W-F1={v['weighted_f1_pct']:.3f}% "
                  f"dF1={v['weighted_f1_delta_pp']:+.3f}pp "
                  f"fixed/broken={v['fixed']}/{v['broken']}", flush=True)


def run_arm(args, ds, groups, reports, session, arm):
    baseline = reports[session]
    verify_baseline(args.baseline_dir, reports, session)
    outdir = args.output_dir / session / arm
    if not args.smoke and outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {outdir}")
    train_idx, valid_idx = fold_indices(groups, session)
    if set(train_idx) & set(valid_idx):
        raise RuntimeError("fold overlap")
    device, model, optim, nll, kl = setup_train(SEED,args.no_cuda)
    # Setup, optimizer, data order and dropout stream match saved raw CSS.
    loader = DataLoader(
        Subset(ds, train_idx[:2] if args.smoke else train_idx),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=ds.collate_fn,
        num_workers=0, generator=torch.Generator().manual_seed(SEED+1000))
    hook = SupervisedContrastHook(model,arm)
    history = []
    try:
        for epoch in range(1 if args.smoke else FINAL_EPOCH):
            hook.reset()
            loss,acc,_,_,_,wf = trainer.train_or_eval_model(
                model,nll,kl,loader,epoch,optimizer=optim,train=True,
                supervised_auxiliary_loss_fn=hook)
            stats = dict(hook.stats)
            if not np.isfinite(loss) or not stats["batches"]:
                raise RuntimeError("nonfinite training loss or no auxiliary batches")
            if not args.smoke and not stats["eligible_anchors"]:
                raise RuntimeError("full training epoch has no positive anchors")
            stats["aux_loss_mean"] = stats["raw_aux_loss_sum"]/stats["batches"]
            stats["mean_window"] = (stats["window_sum"]/stats["window_count"]
                                    if stats["window_count"] else None)
            history.append({"epoch":epoch+1,"train_loss":float(loss),
                            "train_accuracy":float(acc),"train_wf1":float(wf),
                            **stats})
            print(f"{session} {arm} epoch={epoch+1} train_W-F1={wf:.2f}% "
                  f"aux={stats['aux_loss_mean']:.5f} "
                  f"positive_anchors={stats['eligible_anchors']}/{stats['valid_anchors']} "
                  f"hard_pairs={stats['hard_negative_pairs']} "
                  f"mean_window={stats['mean_window']}", flush=True)
    finally:
        hook.close()
    if args.smoke:
        print(f"SMOKE PASS {arm}: actual CSS train forward/backward; "
              "no held-out inference or saved checkpoint.", flush=True)
        return

    # Single final held-out evaluation, same order as frozen baseline.
    valid = DataLoader(Subset(ds,valid_idx),batch_size=BATCH_SIZE,shuffle=False,
                       collate_fn=ds.collate_fn,num_workers=0)
    _,labels,preds,masks,_ = trainer.evaluate_loaded_model(model,valid)
    mask = np.asarray(masks,dtype=bool)
    y,p = np.asarray(labels,dtype=int)[mask],np.asarray(preds,dtype=int)[mask]
    b = np.asarray(baseline["predictions"],dtype=int)
    if ([str(ds.keys[i]) for i in valid_idx] != baseline["held_out_dialogue_ids"] or
            not np.array_equal(y,baseline["labels"]) or y.shape != b.shape):
        raise RuntimeError("final held-out data do not match frozen raw CSS reference")
    summary = summarize(y,p,b,shift_groups(ds,baseline))
    outdir.mkdir(parents=True,exist_ok=False)
    cp_path = outdir/"epoch200.pt"
    protocol = {
        "held_out_session":session, "training_sessions":[s for s in SESSIONS if s!=session],
        "seed":SEED, "epochs":FINAL_EPOCH, "batch_size":BATCH_SIZE,
        "arm":arm,"tau":TAU,"alpha":ALPHA,"fixed_window":FIXED_WINDOW,
        "dynamic_qk":"fixed_identity_no_gradient",
        "hard_weight_gradient":"literal_differentiable",
        "anchor_reduction":"mean_anchors_with_positives",
        "pairs":"within_dialogue_only","checkpoint_selection":"fixed_epoch_200",
        "baseline_checkpoint_sha256":baseline["checkpoint_sha256"],
        "official_test_scored":False,"original_pgm_weights":True,
        "weight_averaging":False,
    }
    torch.save({"model_state_dict":model.state_dict(),"protocol":protocol},cp_path)
    write_new(outdir/"report.json",{
        "protocol":protocol, "checkpoint_sha256":sha256_file(cp_path),
        "held_out_dialogue_ids":baseline["held_out_dialogue_ids"],
        "held_out_dialogue_lengths":baseline["held_out_dialogue_lengths"],
        "labels":y.tolist(),"predictions":p.tolist(),"summary":summary,
        "history":history})
    print_scores(f"FINAL {session} {arm}",summary)
    print("Saved:",outdir/"report.json")


def collect(args, ds, reports, sessions, tag):
    path = args.output_dir / f"{tag}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    yy, bb, offset = [], [], 0
    pp = {a:[] for a in (BASELINE,)+ARMS}
    subgroup_lists = {g:[] for g in ("changed","stable","first")}
    per_session = {}
    for s in sessions:
        verify_baseline(args.baseline_dir,reports,s)
        base = reports[s]
        y,b = np.asarray(base["labels"],dtype=int),np.asarray(base["predictions"],dtype=int)
        groups = shift_groups(ds,base)
        yy.append(y); bb.append(b); pp[BASELINE].append(b)
        for g,ix in groups.items():
            subgroup_lists[g].append(ix+offset)
        per_session[s] = {BASELINE:summarize(y,b,b,groups)}
        for arm in ARMS:
            root = args.output_dir/s/arm
            obj = read_json(root/"report.json")
            p = obj["protocol"]
            expected = {
                "held_out_session":s,"training_sessions":[v for v in SESSIONS if v!=s],
                "seed":SEED,"epochs":FINAL_EPOCH,"batch_size":BATCH_SIZE,
                "arm":arm,"tau":TAU,"alpha":ALPHA,"fixed_window":FIXED_WINDOW,
                "dynamic_qk":"fixed_identity_no_gradient",
                "hard_weight_gradient":"literal_differentiable",
                "anchor_reduction":"mean_anchors_with_positives",
                "pairs":"within_dialogue_only","checkpoint_selection":"fixed_epoch_200",
                "baseline_checkpoint_sha256":base["checkpoint_sha256"],
                "official_test_scored":False,"original_pgm_weights":True,
                "weight_averaging":False,
            }
            if p != expected or sha256_file(root/"epoch200.pt") != obj["checkpoint_sha256"]:
                raise ValueError(f"bad checkpoint protocol or hash: {s} {arm}")
            if (obj["held_out_dialogue_ids"] != base["held_out_dialogue_ids"] or
                    obj["held_out_dialogue_lengths"] != base["held_out_dialogue_lengths"] or
                    obj["labels"] != base["labels"]):
                raise ValueError("held-out alignment mismatch")
            pred = np.asarray(obj["predictions"],dtype=int)
            if pred.shape != y.shape or np.any((pred<0)|(pred>=6)):
                raise ValueError("bad held-out predictions")
            pp[arm].append(pred)
            per_session[s][arm] = summarize(y,pred,b,groups)
        offset += len(y)
    y,b = np.concatenate(yy),np.concatenate(bb)
    if sessions == list(SESSIONS) and len(y) != 5810:
        raise ValueError("incorrect OOF utterance count")
    groups = {g:np.concatenate(ix) for g,ix in subgroup_lists.items()}
    if sessions == list(SESSIONS) and (len(groups["changed"]) != 1610 or
            len(groups["stable"]) != 3960 or len(groups["first"]) != 240):
        raise ValueError("incorrect OOF subgroup coverage")
    result = {
        "protocol":{"sessions":sessions,"seed":SEED,"epochs":FINAL_EPOCH,
                    "exploratory":True,"official_test_scored":False,
                    "dynamic_window_is_approximation":True},
        "pooled":{arm:summarize(y,np.concatenate(v),b,groups) for arm,v in pp.items()},
        "per_session":per_session,
    }
    for arm,item in result["pooled"].items():
        print_scores(arm,item)
    print("\nNo official TEST. Dynamic EII arm is an explicit approximation.")
    write_new(path,result)
    print("Saved:",path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--fold",choices=SESSIONS)
    mode.add_argument("--pilot-summary",action="store_true")
    mode.add_argument("--aggregate",action="store_true")
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--baseline-dir",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--no-cuda",action="store_true")
    args = parser.parse_args()
    if bool(args.arm) != bool(args.fold):
        parser.error("--fold and --arm must be used together")
    ds = IEMOCAPDataset(train=True)
    metadata,groups,reports = read_inputs(args.baseline_dir,ds)
    if args.preflight:
        for s in SESSIONS:
            verify_baseline(args.baseline_dir,reports,s)
        print(json.dumps(metadata,indent=2))
        print("PREFLIGHT PASS: existing raw CSS baseline, TRAIN-only, 4 folds.")
    elif args.smoke:
        for arm in ARMS:
            run_arm(args,ds,groups,reports,PILOT,arm)
    elif args.pilot_summary:
        collect(args,ds,reports,[PILOT],"pilot_summary")
    elif args.aggregate:
        collect(args,ds,reports,list(SESSIONS),"cross_session_summary")
    else:
        run_arm(args,ds,groups,reports,args.fold,args.arm)


if __name__ == "__main__":
    main()

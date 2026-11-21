#!/usr/bin/env python3
"""Posthoc V15 error audit from SAVED TRAIN-only predictions.

No model instantiation, feature loading, training, test loader, or
official TEST inference/scoring. Reads only the user-generated V15
TRAIN-only report.json and quantifies fixed/broken predictions on
the same aligned A-B-A examples. Descriptive; NEVER choose future
hyperparameters or threshold values from these analyses.
"""
import argparse
import json
from pathlib import Path

import numpy as np

INPUT = Path("results/v15_conditional_cross_speaker/report.json")
OUTPUT = Path("results/v15_conditional_cross_speaker/audit_posthoc.json")
REQUIRED = ("a_only","self","real_b","shuffled_b","real_b_replaced")


def paired_counts(y,reference,challenger,mask=None):
    y=np.asarray(y,dtype=np.int64)
    ref=np.asarray(reference,dtype=np.int64)
    cand=np.asarray(challenger,dtype=np.int64)
    if y.ndim!=1 or y.shape!=ref.shape or y.shape!=cand.shape:
        raise ValueError("unaligned labels / predictions")
    if mask is not None:
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=y.shape:
            raise ValueError("unaligned subgroup mask")
        y,ref,cand=y[mask],ref[mask],cand[mask]
    before=(ref==y)
    after=(cand==y)
    fixed=int((~before & after).sum())
    broken=int((before & ~after).sum())
    flips=int((ref!=cand).sum())
    return {
        "n":int(len(y)),
        "fixed":fixed,
        "broken":broken,
        "net_more_correct":fixed-broken,
        "raw_prediction_flips":flips,
        "wrong_to_different_wrong":int((~before & ~after & (ref!=cand)).sum()),
        "changed_prediction_rate_pct":float(100*flips/len(y)) if len(y) else None,
    }


def audit(report):
    p=report.get("protocol",{})
    if (p.get("partition")!="ORIGINAL TRAIN ONLY (Ses01-Ses04)"
            or p.get("official_test_scored") is not False
            or p.get("official_test_loader_created") is not False
            or p.get("smoke") is not False):
        raise ValueError("not a complete four-fold TRAIN-only V15 report")
    rows=report.get("seed_results",[])
    if len(rows)!=3:
        raise ValueError("expected 3 predeclared seed runs")
    output=[]
    for row in rows:
        seed=int(row["train_seed"])
        y=np.asarray(row["pooled_labels"],dtype=np.int64)
        ch=np.asarray(row["pooled_changed_for_posthoc_only"],dtype=bool)
        if y.shape!=(3751,) or ch.shape!=y.shape or int(ch.sum())!=1166:
            raise ValueError(f"seed {seed}: nonstandard 3751/1166 TRAIN cohort")
        predictions=row["pooled_predictions"]
        if set(predictions)!=set(REQUIRED):
            raise ValueError(f"seed {seed}: missing comparison predictions")
        for name in REQUIRED:
            pred=np.asarray(predictions[name],dtype=np.int64)
            if pred.shape!=y.shape or np.any((pred<0)|(pred>5)):
                raise ValueError(f"seed {seed}: unaligned/invalid {name}")
        analyses={}
        for comparison in ("self","real_b","shuffled_b","real_b_replaced"):
            preds=predictions[comparison]
            analyses[f"{comparison}_vs_a_only"]={
                "all":paired_counts(y,predictions["a_only"],preds),
                "stable":paired_counts(y,predictions["a_only"],preds,~ch),
                "changed":paired_counts(y,predictions["a_only"],preds,ch),
            }
        for base in ("self","shuffled_b","real_b_replaced"):
            analyses[f"real_b_vs_{base}"]={
                "all":paired_counts(y,predictions[base],predictions["real_b"]),
                "stable":paired_counts(y,predictions[base],predictions["real_b"],~ch),
                "changed":paired_counts(y,predictions[base],predictions["real_b"],ch),
            }
        # Original fold logs already contain each correction head's
        # prediction flip fraction, without needing to infer again.
        output.append({
            "seed":seed,
            "pooled_deltas_pp":row["pooled_deltas_pp"],
            "pairwise_correctness":analyses,
            "fold_prediction_flip_fractions":[
                {
                    "session":fold["held_train_session"],
                    "diagnostics":fold["prediction_flip_fraction"],
                } for fold in row["folds"]
            ],
        })
    return {
        "analysis":"TRAIN-only posthoc paired correctness; no model training/evaluation",
        "test_data_accessed":False,
        "input_partition":"V15 pooled out-of-fold Ses01-Ses04 A-B-A",
        "n_per_seed":3751,
        "n_changed_per_seed":1166,
        "repeated_seed_predictions_not_independent_samples":True,
        "results":output,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=INPUT)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing saved V15 report: {args.input}")
    if args.output.exists():
        parser.error(f"Refuse to overwrite existing posthoc audit: {args.output}")
    report=json.loads(args.input.read_text(encoding="utf-8"))
    result=audit(report)
    print("V15 TRAIN-ONLY PAIRED CORRECTNESS AUDIT")
    for row in result["results"]:
        print(f"\nSeed {row['seed']} (each subgroup on same saved predictions)")
        for name in (
            "real_b_vs_a_only","real_b_vs_self",
            "real_b_vs_shuffled_b","real_b_vs_real_b_replaced"
        ):
            comparisons=row["pairwise_correctness"][name]
            x=comparisons["all"]
            changed=comparisons["changed"]
            print(
                f"  {name:<28} N={x['n']} "
                f"fixed={x['fixed']} broken={x['broken']} "
                f"net={x['net_more_correct']:+d} "
                f"flips={x['raw_prediction_flips']} "
                f"| changed(N={changed['n']}) fixed={changed['fixed']} "
                f"broken={changed['broken']} "
                f"net={changed['net_more_correct']:+d}"
            )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    print("\nNo retraining, TEST access or checkpoint/model selection occurred.")
    print("Saved:",args.output)


if __name__=="__main__":
    main()

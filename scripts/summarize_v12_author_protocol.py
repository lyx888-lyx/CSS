#!/usr/bin/env python3
"""Summarize exact saved V12 original-author-protocol TEST-selected reports.

No dataset or checkpoint is loaded; this only compares already saved
historical benchmark reports and NEVER picks architectures/hyperparams.
"""
import json
from pathlib import Path

from research.v11_test_protocol import paired_changes

ARMS=("original","self","reaction")
ROOT=Path("results/v12_original_css_author_protocol/author_test_selected")


def main():
    rows=[]
    for arm in ARMS:
        path=ROOT/arm/"report.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run all three separately under "
                "the SAME complete original author protocol."
            )
        row=json.loads(path.read_text(encoding="utf-8"))
        p=row.get("protocol",{})
        if (row.get("arm")!=arm
            or p.get("train_sessions")!=[1,2,3,4]
            or p.get("train_utterances")!=5810
            or p.get("test_utterances")!=1623
            or p.get("test_selection")!="best_test_weighted_f1_every_epoch"
            or p.get("test_used_to_select_checkpoint") is not True
            or p.get("seed")!=61078
            or p.get("training_epochs")!=200
            or p.get("clean_independent_test") is not False):
            raise ValueError(f"{arm}: inconsistent benchmark protocol")
        if len(row.get("labels",[]))!=1623 or len(row.get("predictions",[]))!=1623:
            raise ValueError(f"{arm}: missing 1623 paired predictions")
        rows.append(row)
    if any(r["labels"]!=rows[0]["labels"] for r in rows):
        raise ValueError("TEST label/order mismatch")
    base=rows[0]["metrics"]
    print("V12 FULL AUTHOR PROTOCOL | IEMOCAP TRAIN=Sessions 1-4 (5810)")
    print("TEST=Session 5 (1623), BEST EPOCH CHOSEN BY TEST weighted-F1")
    print("IMPORTANT: source-faithful replication, NOT untouched TEST evaluation")
    print(f"{'Model':<12} {'BEST epoch':>10} {'ACC':>9} {'W-F1':>9} "
          f"{'ΔCSS':>9} {'Macro-F1':>10}")
    for r in rows:
        m=r["metrics"]
        print(f"{r['arm']:<12} {r['epoch']:>10d} "
              f"{m['accuracy']:>8.2f}% {m['weighted_f1']:>8.2f}% "
              f"{m['weighted_f1']-base['weighted_f1']:>+8.2f}pp "
              f"{m['macro_f1']:>9.2f}%")
    print("\nReference: CSS published IEMOCAP TEST ACC75.42 / W-F1 75.66")
    print("Historical user reproduction TEST W-F1 75.28")
    print("\nPaired correctness vs original:")
    for r in rows[1:]:
        changes=paired_changes(
            rows[0]["labels"],r["predictions"],rows[0]["predictions"]
        )
        print(" ",r["arm"],changes)
    print("\nPer-class metrics (class ACC = recall, not global ACC)")
    for cls in range(6):
        label=base["class_metrics"][str(cls)]["name"]
        print(f"  {label}:")
        for r in rows:
            m=r["metrics"]["class_metrics"][str(cls)]
            print(f"    {r['arm']:<9} ACC={m['accuracy_recall']:.2f}% "
                  f"F1={m['f1']:.2f}% N={m['support']}")
    print("\nNo re-scoring or test-based modification of the methods was performed.")


if __name__=="__main__":
    main()

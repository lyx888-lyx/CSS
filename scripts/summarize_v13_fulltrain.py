#!/usr/bin/env python3
"""Summary of V13 already saved full-author IEMOCAP benchmark reports.

No dataset, model, checkpoint, train, or official TEST is loaded.
The reported comparisons are HISTORICAL TEST-SELECTED measurements,
not clean holdout generalization estimates.
"""
import json
from pathlib import Path

from research.v11_test_protocol import paired_changes

BASE=Path("results/v12_original_css_author_protocol/author_test_selected/original/report.json")
ROOT=Path("results/v13_cross_speaker_pretext/author_test_selected")


def checked(path,arm):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; no experimental predictions available")
    r=json.loads(path.read_text(encoding="utf-8"))
    p=r.get("protocol",{})
    expected=(
        p.get("train_sessions")==[1,2,3,4]
        and p.get("train_utterances")==5810
        and p.get("test_utterances")==1623
        and p.get("test_selection")=="best_test_weighted_f1_every_epoch"
        and p.get("test_used_to_select_checkpoint") is True
        and (r.get("arm")==arm)
        and len(r.get("labels",[]))==1623
        and len(r.get("predictions",[]))==1623
    )
    if not expected:
        raise ValueError(f"{arm}: differing protocol or missing saved predictions")
    return r


def main():
    original=checked(BASE,"original")
    self_row=checked(ROOT/"self"/"report.json","self")
    cross_row=checked(ROOT/"cross"/"report.json","cross")
    arms=[original,self_row,cross_row]
    if any(row["labels"]!=original["labels"] for row in arms[1:]):
        raise ValueError("non-aligned TEST labels")
    first=original["metrics"]["weighted_f1"]
    print("V13 ALL-TRAIN IEMOCAP BENCHMARK | AUTHOR TEST-SELECTED EPOCHS")
    print("TRAIN sessions 1–4, 5810 utterances; TEST Session5, 1623")
    print("Source CSS paper W-F1=75.66, original reproduced W-F1=75.28")
    print("NOT AN INDEPENDENT HOLDOUT; TEST has been viewed repeatedly")
    print(f"{'arm':<12} {'epoch':>6} {'ACC':>9} {'W-F1':>9} "
          f"{'ΔCSS':>9} {'macro-F1':>10}")
    for row in arms:
        m=row["metrics"]
        print(f"{row['arm']:<12} {row['epoch']:>6} "
              f"{m['accuracy']:>8.2f}% "
              f"{m['weighted_f1']:>8.2f}% "
              f"{m['weighted_f1']-first:>+8.2f}pp "
              f"{m['macro_f1']:>9.2f}%")
    print("\nPaired TEST correctness (fixed / broken / net):")
    for row in arms[1:]:
        print(f"  {row['arm']}:",
              paired_changes(original["labels"],
                             row["predictions"],original["predictions"]))
    print("\nSelf vs Cross (incremental other-speaker benefit):")
    print("  ",paired_changes(self_row["labels"],
                           cross_row["predictions"],self_row["predictions"]))
    print("\nPer-class F1")
    for cls in range(6):
        name=original["metrics"]["class_metrics"][str(cls)]["name"]
        print(f"  {name}: " + ", ".join(
            f"{r['arm']}={r['metrics']['class_metrics'][str(cls)]['f1']:.2f}"
            for r in arms
        ))
    print("\nNo method or model selection should be made on this TEST table.")


if __name__=="__main__":
    main()

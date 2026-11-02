#!/usr/bin/env python3
"""Summarize the original MELD TEST-selected protocol from two run logs.

Important: Best test F1 is the best TEST-selected epoch, as the original
train_MELD.py does. It is NOT an unbiased held-out test estimate.
"""
import argparse
import json
import re
from pathlib import Path


BEST = re.compile(
    r"^AUTHOR_PROTOCOL_BEST epoch=(\d+) test_acc=([0-9.]+) "
    r"test_weighted_f1=([0-9.]+) checkpoint=(.+)$",
    re.MULTILINE
)


def parse_log(logfile):
    if not logfile.is_file():
        raise FileNotFoundError(f"Missing training log: {logfile}")
    content = logfile.read_text(encoding="utf-8", errors="replace")
    results = list(BEST.finditer(content))
    if len(results) != 1:
        raise RuntimeError(
            f"{logfile}: expected exactly one final AUTHOR_PROTOCOL_BEST line; "
            f"found {len(results)}. Training may not have completed."
        )
    last = results[0]
    checkpoint = Path(last.group(4).strip())
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing saved best checkpoint: {checkpoint}")
    if "AUTHOR_PROTOCOL_NOTE test_set_used_for_epoch_selection=true" not in content:
        raise RuntimeError(f"{logfile}: missing test-selection protocol marker")
    return {
        "selected_epoch": int(last.group(1)),
        "test_accuracy": float(last.group(2)),
        "test_weighted_f1": float(last.group(3)),
        "checkpoint": str(checkpoint),
        "log": str(logfile),
    }


def summarize(run_dir):
    a = parse_log(run_dir / "css.log")
    b = parse_log(run_dir / "v0.log")
    delta = round(b["test_weighted_f1"] - a["test_weighted_f1"], 4)
    result = {
        "dataset": "MELD",
        "selection_rule": "BEST TEST weighted F1 across epochs (author's original protocol)",
        "test_selected": True,
        "warning": (
            "This reports TEST-selected results, not an independently held-out "
            "generalization estimate. Do not use repeated test comparisons to "
            "select or redesign the new model."
        ),
        "css_baseline": a,
        "css_plus_v0": b,
        "delta_weighted_f1_percentage_points": delta,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    lines = [
        "MELD | ORIGINAL AUTHOR PROTOCOL | TEST-SELECTED RESULTS",
        "======================================================",
        "Caution: BEST epoch is selected using the TEST set, not independent validation.",
        "Original CSS  | best TEST weighted F1: {:.2f}% | test ACC: {:.2f}% | epoch: {}".format(
            a["test_weighted_f1"], a["test_accuracy"], a["selected_epoch"]),
        "CSS + V0      | best TEST weighted F1: {:.2f}% | test ACC: {:.2f}% | epoch: {}".format(
            b["test_weighted_f1"], b["test_accuracy"], b["selected_epoch"]),
        "V0 - CSS      | TEST weighted F1 difference: {:+.2f} percentage points".format(delta),
        "",
        "Original CSS checkpoint: " + a["checkpoint"],
        "V0 checkpoint:           " + b["checkpoint"],
        "Original CSS full log:   " + a["log"],
        "V0 full log:             " + b["log"],
        "Structured summary:      " + str(run_dir / "summary.json"),
        "Full classification reports and confusion matrices are in each training log.",
        "Research caution: Test-selected F1 alone cannot prove an innovative module helps.",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    print(summarize(args.run_dir))


if __name__ == "__main__":
    main()

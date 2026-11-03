#!/usr/bin/env python3
"""Compare five independent V2 state probes on the SAME validation split."""
import argparse
import json
from pathlib import Path

from research.v2_state_model import MODES


GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")


def validate_and_summarize(reports):
    if set(reports) != set(MODES):
        raise ValueError("expected exactly five modes: " + ", ".join(MODES))
    first = reports["current"]
    conf = first["config"]
    baseline_idx = first["valid_dialogue_indices"]
    if not baseline_idx:
        raise ValueError("missing validation dialogue indices")
    for mode in MODES:
        r = reports[mode]
        c = r["config"]
        if c["mode"] != mode:
            raise ValueError(f"mismatch between mode key and stored config: {mode}")
        for key in ("dataset", "seed", "split_seed", "valid_ratio",
                    "batch_size", "epochs", "hidden_dim", "dropout",
                    "lr", "weight_decay", "max_train_batches",
                    "validation_dialogues", "train_dialogues"):
            if c[key] != conf[key]:
                raise ValueError(f"{mode} differs from current on {key}")
        if r["valid_dialogue_indices"] != baseline_idx:
            raise ValueError(f"{mode} used different validation dialogue indices")
        if r["validation_utterances"] != first["validation_utterances"]:
            raise ValueError(f"{mode} used different validation utterances")
        if c["test_evaluated"]:
            raise ValueError(f"{mode}: official TEST must be untouched")
    result = {
        "protocol": {
            "dataset": conf["dataset"],
            "train_seed": conf["seed"],
            "split_seed": conf["split_seed"],
            "train_dialogues": conf["train_dialogues"],
            "validation_dialogues": conf["validation_dialogues"],
            "validation_utterances": first["validation_utterances"],
            "epochs": conf["epochs"],
            "max_train_batches": conf["max_train_batches"],
            "test_evaluated": False,
            "warning": (
                "Independent lightweight current-utterance model. "
                "NOT an architecture-matched CSS reproduction. "
                "One seed/split pilot; not proof of novelty or causal influence."
            ),
        },
        "rows": [],
    }
    current = first["validation"]
    for mode in MODES:
        report = reports[mode]
        score = report["validation"]
        row = {
            "mode": mode, "selected_epoch": report["selected_epoch"],
            "checkpoint": report["checkpoint"],
            "scores": {k: score[k] for k in GROUPS},
            "delta_vs_current_pp": {
                k: None if score[k]["weighted_f1"] is None
                or current[k]["weighted_f1"] is None
                else round(score[k]["weighted_f1"] - current[k]["weighted_f1"], 2)
                for k in GROUPS
            },
        }
        result["rows"].append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    reports = {}
    for mode in MODES:
        path = args.suite / mode / "summary.json"
        if not path.is_file():
            p.error("missing result: " + str(path))
        reports[mode] = json.loads(path.read_text(encoding="utf-8"))
    result = validate_and_summarize(reports)
    output = args.output or args.suite / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nV2 FIRST-ORDER SPEAKER STATE | MELD DIALOGUE-VALIDATION ONLY")
    print("Mode              Overall W-F1  Stable W-F1  Changed W-F1  After-other-change  vs current")
    for r in result["rows"]:
        s = r["scores"]
        fmt = lambda k: (
            "N/A" if s[k]["weighted_f1"] is None
            else f"{s[k]['weighted_f1']:.2f}%"
        )
        d = r["delta_vs_current_pp"]["all"]
        print(
            f"{r['mode']:<18}"
            f"{fmt('all'):>12} {fmt('stable'):>12} "
            f"{fmt('changed'):>13} {fmt('changed_after_other'):>19} "
            f"{d:>+9.2f}pp"
        )
    print("Selected best epochs:")
    for r in result["rows"]:
        print(f"  {r['mode']}: {r['selected_epoch']}")
    if result["protocol"]["max_train_batches"]:
        print("SMOKE ONLY: sub-sampled training batches; not a fair full-data result.")
    print("These models do NOT reproduce CSS and their F1 must not be "
          "presented as an improved CSS score.")
    print(f"Full results: {output}")


if __name__ == "__main__":
    main()

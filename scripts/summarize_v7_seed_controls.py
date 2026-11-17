#!/usr/bin/env python3
"""V7: summarize exactly three preregistered matched full-vs-residual seeds.

The fixed split already selected V6.1 checkpoints. These
measure training-seed sensitivity ON THE SAME 87 VALID dialogues,
not independent validation replicates or official TEST performance.
No model checkpoint or dataset is read by this script.
"""
import json
import statistics
from pathlib import Path

GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other"
)
SEEDS = (10068, 21431, 31907)
FIXED_SPLIT_SEED = 36427
ROOT = Path("results")


def read_pair(path, seed):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("split") != "validation_only"
            or len(report.get("models", [])) != 2):
        raise ValueError(f"not a same-cohort paired VALID report: {path}")
    full, residual = report["models"]
    parsed = {}
    for group in GROUPS:
        a = full["scores"][group]["weighted_f1"]
        b = residual["scores"][group]["weighted_f1"]
        n = full["scores"][group]["n"]
        if (n != residual["scores"][group]["n"]
                or a is None or b is None):
            raise ValueError(f"invalid paired support for {group}: {path}")
        parsed[group] = {
            "n": n,
            "full": float(a),
            "residual": float(b),
            "delta_residual_minus_full": float(b - a),
        }
    if parsed["all"]["n"] != 875:
        raise ValueError(f"not V6.1 fixed 875-turn cohort: {path}")
    return {
        "seed": seed,
        "source": str(path),
        "full_selected_epoch": full["selected_epoch"],
        "residual_selected_epoch": residual["selected_epoch"],
        "groups": parsed,
    }


def summarize(rows):
    if [d["seed"] for d in rows] != list(SEEDS):
        raise ValueError("expected exactly preregistered seed order")
    summary = {}
    for group in GROUPS:
        deltas = [
            r["groups"][group]["delta_residual_minus_full"]
            for r in rows
        ]
        summary[group] = {
            "delta_pp_per_seed": deltas,
            "mean_delta_pp": float(statistics.mean(deltas)),
            "seed_sd_delta_pp": float(statistics.stdev(deltas)),
            "positive_seed_count": sum(x > 0 for x in deltas),
            "negative_seed_count": sum(x < 0 for x in deltas),
            "zero_seed_count": sum(x == 0 for x in deltas),
            "validation_dialogues_reused_across_seeds": 87,
        }
    return summary


def main():
    sources = [
        ROOT / ("v61_spf_paired_split36427_trainseed10068"
                "/paired_validation.json"),
        ROOT / ("v7_spf_seed21431_split36427"
                "/paired_validation.json"),
        ROOT / ("v7_spf_seed31907_split36427"
                "/paired_validation.json"),
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing paired results (do not fabricate scores):\n"
            + "\n".join(missing)
        )
    rows = [
        read_pair(p, seed) for p, seed in zip(sources, SEEDS)
    ]
    overall = summarize(rows)
    print("V7 FIXED THREE-SEED SPF ABLATION | SAME ALREADY INSPECTED VALID")
    print("MELD: 846 TRAIN/87 VALID; 100 epochs; original PGM; no TEST.")
    print("Residual-only minus original full-SPF W-F1 in percentage points")
    print("group                    " +
          "  ".join(str(s) for s in SEEDS) +
          "      mean       seed SD    positive/3")
    for group in GROUPS:
        row = overall[group]
        vals = row["delta_pp_per_seed"]
        print(
            f"{group:<23} " +
            " ".join(f"{v:+7.2f}" for v in vals) +
            f"  {row['mean_delta_pp']:+7.2f} "
            f"  {row['seed_sd_delta_pp']:>7.2f} "
            f"  {row['positive_seed_count']:>2d}/3"
        )
    report = {
        "protocol": {
            "split_seed": FIXED_SPLIT_SEED,
            "seeds": list(SEEDS),
            "epochs": 100,
            "dataset": "MELD",
            "training_dialogues": 846,
            "validation_dialogues": 87,
            "validation_utterances": 875,
            "already_inspected_validation": True,
            "test_accessed": False,
            "purpose": (
                "Estimate variation across three matched training "
                "random seeds on fixed and previously inspected "
                "development VALID; NOT three independent validation "
                "cohorts or publication-level generalization proof."
            ),
        },
        "per_seed": rows,
        "summary": overall,
    }
    path = ROOT / "v7_three_seed_spf_summary.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print("\nSaved:", path)
    print("No algorithm improvement has been attempted or claimed.")


if __name__ == "__main__":
    main()

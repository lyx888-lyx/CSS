#!/usr/bin/env python3
"""V5.1: exact PGM projection-vs-backprop direction audit from SAVED V5 JSON.

NO PyTorch, checkpoint, GPU, dataset, gradients, or model execution.
This consumes ONLY gradients' norms, cosines and alphas saved during V5.

CSS PGM solves min_{alpha in simplex} ||sum alpha_i u_i||^2
for u_i = g_i / ||g_i|| (shared parameter subspace).
But the actual source backpropagates sum alpha_i L_i, giving
d_raw = sum alpha_i g_i = sum alpha_i ||g_i|| u_i.
We measure the angle between d_unit = sum alpha_i u_i and d_raw.

A mismatch is mathematical and observational; it DOES NOT establish
that the normalized update would yield a better classification model.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

TASKS = ("L1", "L2", "L3")
PAIRS = (("L1", "L2"), ("L1", "L3"), ("L2", "L3"))
EPS = 1e-12


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def matvec(a, x):
    return [dot(row, x) for row in a]


def quadratic(x, q, y):
    return dot(x, matvec(q, y))


def gram_matrix(cosines):
    q = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        q[i][i] = 1.0
    for i, (x, y) in enumerate(PAIRS):
        value = cosines.get(f"{x}_{y}")
        if value is None:
            raise ValueError(
                f"Undefined cosine for {x}_{y}: zero task gradient; "
                "V5.1 direction comparison requires three nonzero gradients"
            )
        if not math.isfinite(float(value)) or not -1.0001 <= value <= 1.0001:
            raise ValueError(f"Invalid cosine {x}_{y}={value}")
        c = max(-1.0, min(1.0, float(value)))
        first = TASKS.index(x)
        second = TASKS.index(y)
        q[first][second] = q[second][first] = c
    return q


def per_batch(batch, index=0):
    """Derive exact direction comparison in original PGM shared space."""
    geom = batch["pgm_shared"]
    n = [float(geom["norms"][name]) for name in TASKS]
    alpha = [float(batch["pgm_weights"][name]) for name in TASKS]
    if any(not math.isfinite(x) or x <= EPS for x in n):
        raise ValueError(f"batch {index}: nonpositive/invalid gradient norm")
    if any(not math.isfinite(a) or a < -1e-5 or a > 1+1e-5
           for a in alpha) or abs(sum(alpha)-1) > 1e-3:
        raise ValueError(f"batch {index}: invalid alpha simplex")
    q = gram_matrix(geom["cosines"])

    # The optimization considered unit-normalized task directions.
    unit_coef = alpha
    # The training loss actually creates unnormalized task gradients.
    raw_coef = [a * b for a, b in zip(alpha, n)]
    unit_sq = quadratic(unit_coef, q, unit_coef)
    raw_sq = quadratic(raw_coef, q, raw_coef)
    if unit_sq <= EPS or raw_sq <= EPS:
        raise ValueError(
            f"batch {index}: degenerate or numerically invalid direction"
        )

    n_unit = math.sqrt(unit_sq)
    n_raw = math.sqrt(raw_sq)
    cosine = quadratic(unit_coef, q, raw_coef) / (n_unit * n_raw)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    q_unit = matvec(q, unit_coef)
    q_raw = matvec(q, raw_coef)
    desired_align = {
        name: float(q_unit[i] / n_unit) for i, name in enumerate(TASKS)
    }
    actual_align = {
        name: float(q_raw[i] / n_raw) for i, name in enumerate(TASKS)
    }
    raw_total = sum(raw_coef)
    return {
        "batch": index,
        "utterances": batch.get("utterances"),
        "angle_deg_between_PGM_target_and_actual_update": angle,
        "cosine_between_directions": cosine,
        "raw_vs_unit_gradient_norm_ratio": n_raw / n_unit,
        "alpha": dict(zip(TASKS, alpha)),
        "gradient_norms": dict(zip(TASKS, n)),
        "PGM_target_task_cosines": desired_align,
        "actual_update_task_cosines": actual_align,
        "task_cosine_gap_actual_minus_target": {
            name: actual_align[name] - desired_align[name]
            for name in TASKS
        },
        "pre_cancellation_raw_magnitude_share": {
            name: raw_coef[i] / raw_total
            for i, name in enumerate(TASKS)
        },
    }


def summarize(rows):
    if not rows:
        raise ValueError("no batch observations")
    angles = [r["angle_deg_between_PGM_target_and_actual_update"]
              for r in rows]
    def avg(values):
        return float(statistics.mean(values))
    return {
        "batches": len(rows),
        "mean_angle_deg": avg(angles),
        "median_angle_deg": float(statistics.median(angles)),
        "min_angle_deg": min(angles),
        "max_angle_deg": max(angles),
        "batches_angle_gt_20_deg": sum(a > 20 for a in angles),
        "batches_angle_gt_30_deg": sum(a > 30 for a in angles),
        "mean_PGM_target_task_cosines": {
            name: avg([r["PGM_target_task_cosines"][name] for r in rows])
            for name in TASKS
        },
        "mean_actual_update_task_cosines": {
            name: avg([r["actual_update_task_cosines"][name] for r in rows])
            for name in TASKS
        },
        "mean_actual_minus_target_task_cosines": {
            name: avg([r["task_cosine_gap_actual_minus_target"][name]
                       for r in rows])
            for name in TASKS
        },
        "mean_raw_magnitude_shares": {
            name: avg([r["pre_cancellation_raw_magnitude_share"][name]
                       for r in rows])
            for name in TASKS
        },
    }


def audit_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("protocol") or "batches" not in data:
        raise ValueError("input does not look like V5 source PGM audit JSON")
    protocol = data["protocol"]
    if (protocol.get("dataset") != "MELD"
            or protocol.get("optimizer_steps") != 0
            or protocol.get("official_test_evaluated") is not False
            or protocol.get("validation_predictions_evaluated") is not False):
        raise ValueError("requires strict train-only, no-update V5 audit JSON")
    batches = data["batches"]
    if not isinstance(batches, list) or not batches:
        raise ValueError("input has no batches")
    rows = [per_batch(b, i) for i, b in enumerate(batches, start=1)]
    return {
        "source_v5_json": str(path),
        "protocol": {
            "source": "original source-aligned PGM shared-parameter gradients",
            "dataset": "MELD",
            "reused_v5_saved_geometry_only": True,
            "new_model_or_dataset_reads": False,
            "new_autograd_calls": 0,
            "new_optimizer_steps": 0,
            "official_test_evaluated": False,
            "warning": (
                "This checks geometric consistency of the normalized-gradient "
                "PGM simplex solution versus source implementation's "
                "raw-loss weighted gradient. It is not a comparison of model "
                "accuracy, optimizer effectiveness, or held-out generalization."
            ),
        },
        "summary": summarize(rows),
        "batches": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/v5_css_pgm_gradient_audit.json")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v51_pgm_direction_alignment.json")
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"missing V5 result JSON: {args.input}")
    result = audit_json(args.input)
    summary = result["summary"]

    print("V5.1 PGM DIRECTION CONSISTENCY | NO TRAINING OR INFERENCE")
    print("Model source: normalized gradients -> SLSQP alpha -> raw weighted loss")
    print("Batch  angle_deg  align_L1(actual/target)  "
          "align_L2(actual/target)  align_L3(actual/target)")
    for row in result["batches"]:
        a = row["actual_update_task_cosines"]
        p = row["PGM_target_task_cosines"]
        print(
            f"{row['batch']:>5d}  "
            f"{row['angle_deg_between_PGM_target_and_actual_update']:>9.2f}  "
            f"{a['L1']:+.3f}/{p['L1']:+.3f}                 "
            f"{a['L2']:+.3f}/{p['L2']:+.3f}                 "
            f"{a['L3']:+.3f}/{p['L3']:+.3f}"
        )
    print("\nPGM DIRECTION MISMATCH SUMMARY (original shared-parameter space)")
    print(
        f"angle: mean={summary['mean_angle_deg']:.2f}deg "
        f"median={summary['median_angle_deg']:.2f}deg "
        f"min={summary['min_angle_deg']:.2f}deg "
        f"max={summary['max_angle_deg']:.2f}deg"
    )
    print(
        f"large offsets: >20deg {summary['batches_angle_gt_20_deg']}/"
        f"{summary['batches']} | >30deg "
        f"{summary['batches_angle_gt_30_deg']}/{summary['batches']}"
    )
    for name in TASKS:
        actual = summary["mean_actual_update_task_cosines"][name]
        target = summary["mean_PGM_target_task_cosines"][name]
        print(
            f"{name} alignment with its UNIT task gradient: "
            f"actual={actual:+.4f}, PGM-target={target:+.4f}, "
            f"delta={actual-target:+.4f}"
        )
    print("Mean pre-cancellation alpha*||gradient|| shares:",
          " ".join(
              f"{name}={summary['mean_raw_magnitude_shares'][name]:.4f}"
              for name in TASKS
          ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("Saved:", args.output)
    print("No evidence of performance gain: this is a geometric diagnostic ONLY.")


if __name__ == "__main__":
    main()

"""Paired dialogue-cluster uncertainty for an ALREADY run V6.1 ablation.

These are descriptive, paired bootstrap intervals. They are not new
model selection, not independent evidence, not a significance guarantee.
Resample entire dialogue records, preserving utterance correlations
and the exact per-utterance full/residual prediction pairing.
"""
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other"
)
NAMES = ("Neutral", "Surprise", "Fear", "Sadness",
         "Joy", "Disgust", "Anger")


def _flatten_group(records, name):
    y, full, resid = [], [], []
    for d in records:
        lengths = {
            len(d["labels"]), len(d["full"]), len(d["residual"]), len(d["tags"])
        }
        if len(lengths) != 1:
            raise ValueError("misaligned paired dialogue predictions")
        for label, f, r, tag in zip(
            d["labels"], d["full"], d["residual"], d["tags"]
        ):
            if name != "all" and name not in tag:
                continue
            y.append(int(label))
            full.append(int(f))
            resid.append(int(r))
    return y, full, resid


def score_group(records, group, n_classes=7):
    y, full, residual = _flatten_group(records, group)
    if not y:
        return {
            "n": 0, "full_f1": None, "residual_f1": None,
            "delta_residual_minus_full": None,
            "full_accuracy": None, "residual_accuracy": None,
            "residual_fixed": 0, "residual_broken": 0,
            "both_correct": 0, "both_wrong": 0,
        }
    if any(z < 0 or z >= n_classes for z in y + full + residual):
        raise ValueError("invalid true/predicted class id")
    sf = float(f1_score(
        y, full, labels=list(range(n_classes)),
        average="weighted", zero_division=0
    ) * 100)
    sr = float(f1_score(
        y, residual, labels=list(range(n_classes)),
        average="weighted", zero_division=0
    ) * 100)
    good_f = [p == gold for p, gold in zip(full, y)]
    good_r = [p == gold for p, gold in zip(residual, y)]
    return {
        "n": len(y),
        "full_f1": sf,
        "residual_f1": sr,
        "delta_residual_minus_full": sr - sf,
        "full_accuracy": float(accuracy_score(y, full)*100),
        "residual_accuracy": float(accuracy_score(y, residual)*100),
        "residual_fixed": sum(not f and r for f, r in zip(good_f, good_r)),
        "residual_broken": sum(f and not r for f, r in zip(good_f, good_r)),
        "both_correct": sum(f and r for f, r in zip(good_f, good_r)),
        "both_wrong": sum(not f and not r for f, r in zip(good_f, good_r)),
    }


def per_class_paired(records, n_classes=7):
    y, full, resid = _flatten_group(records, "all")
    out = {}
    for k in range(n_classes):
        idx = [i for i, label in enumerate(y) if label == k]
        out[NAMES[k]] = {
            "support": len(idx),
            "full_correct": sum(full[i] == k for i in idx),
            "residual_correct": sum(resid[i] == k for i in idx),
            "residual_fixed": sum(
                full[i] != k and resid[i] == k for i in idx
            ),
            "residual_broken": sum(
                full[i] == k and resid[i] != k for i in idx
            ),
            "full_predicted_count": sum(z == k for z in full),
            "residual_predicted_count": sum(z == k for z in resid),
        }
    return out


def bootstrap_dialogues(records, n_classes=7, reps=1500, seed=64217):
    """Percentile 95% intervals of paired residual-minus-full W-F1.

    Confidence intervals can be wide with 87 dialogue units and
    highly variable dialogue lengths; replication still needed.
    """
    if len(records) < 2 or reps < 100:
        raise ValueError("requires >=2 dialogues and >=100 replications")
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in GROUPS}
    for _ in range(reps):
        ix = rng.integers(0, len(records), len(records))
        sample = [records[int(i)] for i in ix]
        for name in GROUPS:
            y, f, r = _flatten_group(sample, name)
            if not y:
                continue
            labels = list(range(n_classes))
            left = f1_score(
                y, f, labels=labels, average="weighted", zero_division=0
            )
            right = f1_score(
                y, r, labels=labels, average="weighted", zero_division=0
            )
            deltas[name].append(100*(right-left))
    out = {}
    for name in GROUPS:
        v = np.asarray(deltas[name], dtype=np.float64)
        out[name] = {
            "bootstrap_valid_replicates": len(v),
            "ci95_pp": (np.percentile(v, [2.5, 97.5]).tolist()
                        if len(v) else None),
            "mean_delta_pp": float(v.mean()) if len(v) else None,
            "residual_higher_fraction": float((v > 0).mean())
            if len(v) else None,
            "interval_includes_zero": bool(v.min() <= 0 <= v.max())
            if len(v) else None,
        }
        if len(v):
            lo, hi = out[name]["ci95_pp"]
            out[name]["interval_includes_zero"] = bool(lo <= 0 <= hi)
    return out


def ensure_paired_dialogues(rows):
    if not rows:
        raise ValueError("no paired dialogues")
    ids = [d["dialogue_id"] for d in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dialogue ID")
    if any(not d["labels"] for d in rows):
        raise ValueError("empty dialogue")
    for d in rows:
        _flatten_group([d], "all")

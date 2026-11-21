"""Pure, CPU-only paired evaluation helpers for V11 IEMOCAP.

Computed from stored label and top-1 predictions ONLY. Bootstrap samples
whole dialogues to avoid treating dependent dialogue turns as independent.
Selected checkpoints came from the SAME inspected validation session; this
cannot provide independent confirmation of a gain or calibrated significance.
"""
import numpy as np

GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")
PAIR_NAMES = (("reaction", "original"), ("reaction", "self"),
              ("self", "original"))


def weighted_f1(y, pred, classes=6):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if y.shape != pred.shape or y.ndim != 1:
        raise ValueError("expected paired 1D y and pred arrays")
    if not len(y):
        return None
    if ((y < 0).any() or (y >= classes).any() or
            (pred < 0).any() or (pred >= classes).any()):
        raise ValueError("invalid class IDs")
    cm = np.bincount(
        classes * y + pred, minlength=classes * classes
    ).reshape(classes, classes)
    support = cm.sum(1)
    pred_count = cm.sum(0)
    denom = support + pred_count
    f1 = np.divide(2 * np.diag(cm), denom,
                   out=np.zeros(classes, dtype=np.float64),
                   where=(denom > 0))
    return float(100.0 * np.dot(support, f1) / support.sum())


def fixed_broken(y, left, right):
    y, left, right = [np.asarray(z, dtype=np.int64)
                       for z in (y, left, right)]
    if y.shape != left.shape or y.shape != right.shape:
        raise ValueError("paired predictions must have the same shape")
    left_ok = left == y
    right_ok = right == y
    return {
        "fixed": int((left_ok & ~right_ok).sum()),
        "broken": int((~left_ok & right_ok).sum()),
        "top1_flips": int((left != right).sum()),
    }


def summarize_groups(dialogues, names, groups=GROUPS):
    if not dialogues:
        raise ValueError("no dialogues")
    out = {}
    for group in groups:
        y, pred = [], {name: [] for name in names}
        for d in dialogues:
            selected = d["masks"][group]
            y.extend(d["labels"][selected].tolist())
            for name in names:
                pred[name].extend(d["predictions"][name][selected].tolist())
        y = np.asarray(y, dtype=np.int64)
        out[group] = {
            "n": int(len(y)),
            "f1": {name: weighted_f1(y, pred[name]) for name in names},
            "paired": {
                f"{left}_minus_{right}": {
                    "f1_delta_pp": (
                        weighted_f1(y, pred[left]) -
                        weighted_f1(y, pred[right])
                    ) if len(y) else None,
                    **fixed_broken(y, pred[left], pred[right])
                }
                for left,right in PAIR_NAMES
                if left in names and right in names
            },
        }
    return out


def bootstrap_dialogues(dialogues, groups=("all", "changed",
                                           "changed_after_other"),
                        repetitions=1500, seed=20261121):
    if len(dialogues) < 2:
        raise ValueError("must have at least two independent dialogues")
    if repetitions < 100:
        raise ValueError("need at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    values = {
        group: {f"{left}_minus_{right}": []
                for left,right in PAIR_NAMES}
        for group in groups
    }
    n = len(dialogues)
    for _ in range(repetitions):
        subset = [dialogues[int(i)]
                  for i in rng.integers(0, n, size=n)]
        summary = summarize_groups(
            subset, ("original", "self", "reaction"), groups=groups)
        for group in groups:
            for key, data in summary[group]["paired"].items():
                if data["f1_delta_pp"] is not None:
                    values[group][key].append(data["f1_delta_pp"])
    output = {}
    for group, pairs in values.items():
        output[group] = {}
        for pair, arr in pairs.items():
            arr = np.asarray(arr)
            output[group][pair] = {
                "replicates_with_support": int(len(arr)),
                "percentile_95_low_pp": (
                    float(np.quantile(arr,.025)) if len(arr) else None
                ),
                "percentile_95_high_pp": (
                    float(np.quantile(arr,.975)) if len(arr) else None
                ),
                "fraction_delta_positive": (
                    float((arr>0).mean()) if len(arr) else None
                ),
            }
    return output

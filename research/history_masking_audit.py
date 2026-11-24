"""Matched past-turn attention-mask diagnostics for frozen original CSS.

Mask choices use only speaker identities and turn indices. Gold emotions are
used AFTER inference for subgroup and paired outcome analysis, not as model
inputs or to choose a turn to mask. This is an intervention on model inputs,
NOT a causal experiment on human emotions.
"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from research.session_confusion_stability import CLASSES, PAIRS, SESSIONS
from research.context_confusion_association import normalize_speakers

ARMS = ("baseline", "previous_own", "near_other")
PAIR_IDS = {frozenset((CLASSES.index(a), CLASSES.index(b))) for a, b in PAIRS}


def matched_positions(speakers):
    """For each target, select latest own history and nearest other to it.

    Other-history control minimizes |k-j| (ties: later k), not using gold.
    Both j and k must be strictly before target. Earlier-own control is the
    second latest own history when available. No sample selection by correctness.
    """
    speakers = normalize_speakers(speakers)
    result = []
    for i, s in enumerate(speakers):
        own = [j for j in range(i) if speakers[j] == s]
        other = [j for j in range(i) if speakers[j] != s]
        if not own or not other:
            continue
        j = own[-1]
        k = min(other, key=lambda x: (abs(x-j), -x))
        result.append((i, j, k, own[-2] if len(own) >= 2 else None))
    return result


def paired_metrics(rows, arm):
    if not rows:
        return {"n": 0}
    y = np.array([r["true"] for r in rows])
    base = np.array([r["baseline_pred"] for r in rows])
    pred = np.array([r[arm + "_pred"] for r in rows])
    prior = np.array([r["previous_own_true"] for r in rows])
    changed = y != prior
    bc, pc = base == y, pred == y
    fixed, broken = int((~bc & pc).sum()), int((bc & ~pc).sum())
    # Count predictions matching the prior label only when that label is wrong.
    history_error_base = int(((base == prior) & changed).sum())
    history_error_new = int(((pred == prior) & changed).sum())
    margin_delta = [
        (r[arm + "_prob"][r["true"]] - r[arm + "_prob"][r["previous_own_true"]])
        - (r["baseline_prob"][r["true"]] - r["baseline_prob"][r["previous_own_true"]])
        for r in rows if r["true"] != r["previous_own_true"]
    ]
    return {
        "n": len(rows), "changed_n": int(changed.sum()),
        "baseline_accuracy_pct": float(100 * bc.mean()),
        "accuracy_pct": float(100 * pc.mean()),
        "accuracy_delta_pp": 100 * (fixed-broken) / len(rows),
        "baseline_weighted_f1_pct": float(100 * f1_score(y, base, average="weighted", zero_division=0)),
        "weighted_f1_pct": float(100 * f1_score(y, pred, average="weighted", zero_division=0)),
        "fixed": fixed, "broken": broken, "net_more_correct": fixed-broken,
        "prediction_flips": int((base != pred).sum()),
        "wrong_previous_emotion_baseline": history_error_base,
        "wrong_previous_emotion_after_mask": history_error_new,
        "net_reduction_wrong_previous_emotion": history_error_base-history_error_new,
        "mean_true_vs_previous_probability_margin_change": (
            float(np.mean(margin_delta)) if margin_delta else None),
    }


def summarize(rows):
    if not rows:
        raise ValueError("empty matched cohort")
    if len({(r["dialogue"], r["index"]) for r in rows}) != len(rows):
        raise ValueError("duplicate target")
    for r in rows:
        i,j,k = r["index"],r["previous_own_index"],r["near_other_index"]
        if not (0 <= j < i and 0 <= k < i and j != k):
            raise ValueError("intervention targets must be distinct past turns")
        if r.get("earlier_own_index") is not None and not 0 <= r["earlier_own_index"] < j:
            raise ValueError("invalid earlier-own control")
        for arm in ARMS + (("earlier_own",) if r.get("earlier_own_index") is not None else ()):
            p = np.asarray(r[arm+"_prob"], dtype=float)
            if p.shape != (6,) or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum()-1)>2e-4:
                raise ValueError("invalid probability vector")
            if r[arm+"_pred"] != int(p.argmax()):
                raise ValueError("probability / prediction mismatch")
        if not (0 <= r["true"] < 6 and 0 <= r["previous_own_true"] < 6):
            raise ValueError("invalid true class")

    def subgroups(rr):
        changed = [r for r in rr if r["true"] != r["previous_own_true"]]
        stable = [r for r in rr if r["true"] == r["previous_own_true"]]
        pair = [r for r in changed if frozenset((r["true"],r["previous_own_true"])) in PAIR_IDS]
        groups = {"all_matched": rr, "changed": changed, "stable": stable,
                  "three_pair_transitions": pair}
        for a,b in PAIRS:
            for cur,prev in ((a,b),(b,a)):
                groups[f"{prev}_to_{cur}_true_transition"] = [
                    r for r in changed if r["true"] == CLASSES.index(cur)
                    and r["previous_own_true"] == CLASSES.index(prev)]
        return {name: {arm: paired_metrics(group,arm)
                       for arm in ("previous_own","near_other")}
                for name,group in groups.items()}

    earlier = [r for r in rows if r.get("earlier_own_index") is not None]
    earlier_changed = [r for r in earlier if r["true"] != r["previous_own_true"]]
    earlier_stable = [r for r in earlier if r["true"] == r["previous_own_true"]]
    return {
        "matched_n": len(rows),
        "mean_previous_own_lag": float(np.mean([r["index"]-r["previous_own_index"] for r in rows])),
        "mean_near_other_lag": float(np.mean([r["index"]-r["near_other_index"] for r in rows])),
        "mean_control_position_distance": float(np.mean([abs(r["near_other_index"]-r["previous_own_index"]) for r in rows])),
        "primary_groups": subgroups(rows),
        "earlier_own_control": {
            name: {arm: paired_metrics(group,arm) for arm in ("previous_own","near_other","earlier_own")}
            for name,group in (("all",earlier),("changed",earlier_changed),("stable",earlier_stable))
        },
        "limitations": [
            "One fixed mask per target, chosen without gold labels.",
            "The near-other and earlier-own masks cannot have exactly the same position as the previous-own mask; role and lag may confound comparison.",
            "Masking a key globally changes contextual attention; it is not an intervention on human emotion and may introduce an out-of-distribution mask.",
            "Gold labels define analysis subgroups only. Original CSS is bidirectional.",
            "Candidate pairs and Session4 have prior development exposure; these findings are exploratory.",
        ],
    }

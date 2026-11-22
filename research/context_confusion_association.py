"""Read-only, label-conditioned context/confusion association diagnostics.

Uses existing TRAIN-only out-of-fold predictions and real TRAIN dialogue
metadata. True previous-speaker emotion labels are used for posthoc analyses
ONLY and must never be treated as features available to an inference model.
"""
from collections import Counter
import numpy as np

from research.session_confusion_stability import CLASSES, SESSIONS, PAIRS

FEATURES = ("own_emotion_changed", "other_intervened",
            "previous_own_label_matches_wrong_target")


def normalize_speakers(values):
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 2:
        if np.any(arr.sum(axis=1) < .999) or np.any(arr.sum(axis=1) > 1.001):
            raise ValueError("unexpected one-hot speaker mask")
        return arr.argmax(axis=1).astype(int).tolist()
    if arr.ndim != 1:
        raise ValueError(f"invalid speaker sequence shape: {arr.shape}")
    vals = [str(x) for x in arr.tolist()]
    unique = list(dict.fromkeys(vals))
    if not 1 <= len(unique) <= 2:
        raise ValueError(f"expected one or two IEMOCAP speakers: {unique}")
    encoding = {s: i for i, s in enumerate(unique)}
    return [encoding[s] for s in vals]


def dialogue_observations(vid, speakers, labels, predictions):
    """Preserve utterance order; never use future/gold labels for prediction."""
    session = str(vid)[:5]
    if session not in SESSIONS:
        raise ValueError(f"non-TRAIN dialogue: {vid}")
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    who = normalize_speakers(speakers)
    if (y.ndim != 1 or len(y) == 0 or p.shape != y.shape or
            len(who) != len(y) or
            np.any((y < 0) | (y >= 6)) or
            np.any((p < 0) | (p >= 6))):
        raise ValueError("dialogue speakers/labels/predictions not aligned")
    last_own = {}
    rows = []
    for i, speaker in enumerate(who):
        own_prev = last_own.get(speaker)
        other_intervened = (
            None if own_prev is None
            else any(who[j] != speaker for j in range(own_prev + 1, i))
        )
        prev_label = None if own_prev is None else int(y[own_prev])
        rows.append({
            "dialogue": str(vid), "session": session, "index": i,
            "dialogue_length": len(y),
            "true": int(y[i]), "pred": int(p[i]),
            "prev_own_true_label": prev_label,
            "own_emotion_changed": None if own_prev is None
                else bool(y[own_prev] != y[i]),
            "other_intervened": other_intervened,
        })
        last_own[speaker] = i
    return rows


def _cell(rows, source, destination, feature):
    """Rates condition on the same CURRENT true emotion class."""
    if source == destination or feature not in FEATURES:
        raise ValueError("invalid direction or context feature")
    cells = {"true": {"n": 0, "mistakes": 0},
             "false": {"n": 0, "mistakes": 0}}
    for row in rows:
        if row["true"] != source or row["prev_own_true_label"] is None:
            continue
        if feature == "previous_own_label_matches_wrong_target":
            condition = row["prev_own_true_label"] == destination
        else:
            condition = row[feature]
            if condition is None:
                raise ValueError("missing context condition")
        key = "true" if condition else "false"
        cells[key]["n"] += 1
        cells[key]["mistakes"] += int(row["pred"] == destination)
    for item in cells.values():
        item["error_pct"] = (
            round(100 * item["mistakes"] / item["n"], 4) if item["n"] else None
        )
    a, b = cells["true"], cells["false"]
    gap = (round(a["error_pct"] - b["error_pct"], 4)
           if a["error_pct"] is not None and b["error_pct"] is not None else None)
    return {"present": a, "absent": b, "risk_difference_pp": gap}


def analyze(rows):
    if not rows:
        raise ValueError("no observations")
    ids = {(r["dialogue"], r["index"]) for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate dialogue/turn")
    if set(r["session"] for r in rows) != set(SESSIONS):
        raise ValueError("expected four TRAIN sessions")
    by_session = {s: [r for r in rows if r["session"] == s] for s in SESSIONS}
    out = {}
    for a, b in PAIRS:
        for first, second in ((a, b), (b, a)):
            src, dst = CLASSES.index(first), CLASSES.index(second)
            direction = f"{first}->{second}"
            selected = [r for r in rows if r["true"] == src]
            out[direction] = {
                "current_true_class_support": len(selected),
                "mistaken_as_target": sum(r["pred"] == dst for r in selected),
                "has_prior_own": sum(r["prev_own_true_label"] is not None
                                     for r in selected),
                "features": {},
            }
            for feature in FEATURES:
                per_session = {
                    s: _cell(by_session[s], src, dst, feature) for s in SESSIONS
                }
                nonzero_comparable = [
                    v["risk_difference_pp"] for v in per_session.values()
                    if v["risk_difference_pp"] is not None
                ]
                out[direction]["features"][feature] = {
                    "pooled": _cell(rows, src, dst, feature),
                    "per_session": per_session,
                    "session_comparable_count": len(nonzero_comparable),
                    "session_positive_gap_count": sum(
                        x > 0 for x in nonzero_comparable),
                    "session_negative_gap_count": sum(
                        x < 0 for x in nonzero_comparable),
                }
    coverage = {
        "observations": len(rows),
        "sessions": {s: len(by_session[s]) for s in SESSIONS},
        "dialogues": len(set(r["dialogue"] for r in rows)),
        "has_prior_own": sum(r["prev_own_true_label"] is not None for r in rows),
        "own_changed": sum(r["own_emotion_changed"] is True for r in rows),
        "own_stable": sum(r["own_emotion_changed"] is False for r in rows),
        "other_intervened": sum(r["other_intervened"] is True for r in rows),
        "no_other_intervened": sum(r["other_intervened"] is False for r in rows),
    }
    return {
        "coverage": coverage,
        "directions": out,
        "limitations": [
            "Posthoc true past emotion labels only for analysis, never inference.",
            "Each directed comparison fixes the current TRUE class; pooled gaps can still be confounded by session and other covariates.",
            "The original CSS encoder is bidirectional and uses dialogue context; associations cannot isolate causal historical influences.",
            "Prior Session4 analysis selected the candidate class pairs; all results remain exploratory.",
            "Utterances within a dialogue are dependent; no unclustered significance tests or p-values are reported.",
            "Small or empty conditional cells are reported instead of silently imputing error rates.",
        ],
    }

"""TRAIN-only, leave-one-session-out IEMOCAP confusion diagnostics.

Descriptive analysis, not a new CSS model or independent confirmation.
Candidate pairs came from the previously inspected Session-4 audit.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ("happy", "sad", "neutral", "angry", "excited", "frustrated")
SESSIONS = ("Ses01", "Ses02", "Ses03", "Ses04")
EXPECTED_DIALOGUES = {"Ses01": 28, "Ses02": 30, "Ses03": 32, "Ses04": 30}
PAIRS = (("angry", "frustrated"), ("happy", "excited"),
         ("neutral", "frustrated"))


def split_by_session(keys, expected_counts=EXPECTED_DIALOGUES):
    """Accept only the 120 official TRAIN dialogue IDs; never TEST IDs."""
    keys = [str(k) for k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dialogue ID")
    groups = {name: [] for name in SESSIONS}
    for idx, key in enumerate(keys):
        session = key[:5]
        if session not in groups or not key.startswith("Ses0"):
            raise ValueError(f"non-TRAIN or unrecognized session: {key}")
        groups[session].append(idx)
    if expected_counts is not None:
        observed = {s: len(indices) for s, indices in groups.items()}
        if observed != dict(expected_counts):
            raise ValueError(f"unexpected TRAIN session dialogue counts: {observed}")
    return groups


def validate_train_only(keys, labels_by_id):
    """Validate 120 dialogues and 5810 labeled turns without accessing TEST."""
    groups = split_by_session(keys)
    total = 0
    turns = {}
    for session, idxs in groups.items():
        n = 0
        for i in idxs:
            y = np.asarray(labels_by_id[keys[i]])
            if y.ndim != 1 or len(y) == 0 or not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"invalid TRAIN labels: {keys[i]}")
            if np.any((y < 0) | (y >= len(CLASSES))):
                raise ValueError(f"out-of-range TRAIN labels: {keys[i]}")
            n += len(y)
        turns[session] = n
        total += n
    if total != 5810:
        raise ValueError(f"expected 5810 TRAIN utterances, got {total}")
    return {"dialogues": len(keys), "utterances": total,
            "dialogues_by_session": {s: len(v) for s, v in groups.items()},
            "utterances_by_session": turns, "official_test_scored": False}


def fold_indices(groups, held_out):
    if held_out not in SESSIONS:
        raise ValueError(f"invalid held-out session: {held_out}")
    valid = list(groups[held_out])
    train = [idx for s in SESSIONS if s != held_out for idx in groups[s]]
    if not train or not valid or set(train) & set(valid):
        raise ValueError("empty or overlapping train/held-out indices")
    return train, valid


def _arrays(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or pred.shape != y.shape:
        raise ValueError("nonempty aligned 1D labels/predictions required")
    if np.any((y < 0) | (y >= 6)) or np.any((pred < 0) | (pred >= 6)):
        raise ValueError("class index out of IEMOCAP-6 range")
    return y, pred


def _matrix(y, pred):
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    return matrix


def pair_stats(matrix):
    """Directional error rates use true-class denominators, not raw counts."""
    mat = np.asarray(matrix, dtype=np.int64)
    if mat.shape != (6, 6) or np.any(mat < 0):
        raise ValueError("expected nonnegative 6x6 confusion matrix")
    support = mat.sum(axis=1)
    total_errors = int(mat.sum() - np.trace(mat))
    out = {}
    for name_a, name_b in PAIRS:
        a, b = CLASSES.index(name_a), CLASSES.index(name_b)
        ab, ba = int(mat[a, b]), int(mat[b, a])
        denom = int(support[a] + support[b])
        out[f"{name_a}<->{name_b}"] = {
            "a_to_b": ab, "b_to_a": ba,
            "a_support": int(support[a]), "b_support": int(support[b]),
            "a_to_b_pct": round(100 * ab / support[a], 4) if support[a] else None,
            "b_to_a_pct": round(100 * ba / support[b], 4) if support[b] else None,
            "pair_error_count": ab + ba,
            "pair_errors_per_pair_support_pct":
                round(100 * (ab + ba) / denom, 4) if denom else None,
            "share_of_all_errors_pct":
                round(100 * (ab + ba) / total_errors, 4) if total_errors else None,
        }
    return out


def summarize_fold(y, pred):
    y, pred = _arrays(y, pred)
    matrix = _matrix(y, pred)
    return {
        "utterances": int(len(y)),
        "accuracy": round(100 * float(accuracy_score(y, pred)), 4),
        "weighted_f1": round(100 * float(f1_score(
            y, pred, labels=list(range(6)), average="weighted", zero_division=0)), 4),
        "macro_f1": round(100 * float(f1_score(
            y, pred, labels=list(range(6)), average="macro", zero_division=0)), 4),
        "errors": int((y != pred).sum()),
        "class_support": {name: int((y == j).sum()) for j, name in enumerate(CLASSES)},
        "confusion_matrix": matrix.tolist(),
        "pairs": pair_stats(matrix),
    }


def aggregate_folds(folds):
    """Aggregate four disjoint, final-epoch out-of-fold predictions."""
    if set(folds) != set(SESSIONS):
        raise ValueError("require one full fold for each TRAIN session")
    ys, ps = [], []
    for session in SESSIONS:
        obj = folds[session]
        if obj.get("held_out_session") != session:
            raise ValueError("held-out session identity mismatch")
        if obj.get("checkpoint_selection") != "fixed_epoch_200_no_heldout_selection":
            raise ValueError("held-out results must not select checkpoints")
        ids = obj.get("held_out_dialogue_ids")
        if len(ids) != EXPECTED_DIALOGUES[session] or len(set(ids)) != len(ids):
            raise ValueError("held-out dialogue coverage mismatch")
        if any(str(k)[:5] != session for k in ids):
            raise ValueError("held-out fold contains another session")
        y, pred = _arrays(obj["labels"], obj["predictions"])
        if len(y) != sum(obj["held_out_dialogue_lengths"]):
            raise ValueError("held-out flattened label/length mismatch")
        if summarize_fold(y, pred)["confusion_matrix"] != obj["summary"]["confusion_matrix"]:
            raise ValueError("reported confusion matrix differs from predictions")
        ys.append(y)
        ps.append(pred)
    all_ids = [k for session in SESSIONS for k in folds[session]["held_out_dialogue_ids"]]
    if len(set(all_ids)) != 120:
        raise ValueError("duplicate dialogues across folds")
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    if len(y_all) != 5810:
        raise ValueError(f"expected 5810 out-of-fold TRAIN utterances, got {len(y_all)}")
    pooled = summarize_fold(y_all, p_all)
    stability = {}
    for pair in pooled["pairs"]:
        values = [folds[s]["summary"]["pairs"][pair] for s in SESSIONS]
        stability[pair] = {
            "nonzero_a_to_b_folds": sum(v["a_to_b"] > 0 for v in values),
            "nonzero_b_to_a_folds": sum(v["b_to_a"] > 0 for v in values),
            "bidirectional_nonzero_folds": sum(
                v["a_to_b"] > 0 and v["b_to_a"] > 0 for v in values),
            "per_session_pair_error_rates_pct": {
                session: values[i]["pair_errors_per_pair_support_pct"]
                for i, session in enumerate(SESSIONS)
            },
            "per_session_directed_rates_pct": {
                session: {"a_to_b": values[i]["a_to_b_pct"],
                          "b_to_a": values[i]["b_to_a_pct"]}
                for i, session in enumerate(SESSIONS)
            },
        }
    return {
        "protocol": {
            "data": "official IEMOCAP original TRAIN Ses01-Ses04 only",
            "held_out": "leave-one-session-out, four folds",
            "selection": "fixed epoch 200; never select on held-out session",
            "official_test_evaluated": False,
            "exploratory": True,
            "prior_pair_selection": "three pairs were chosen after seeing Session4 audit",
            "causal_interpretation": False,
        },
        "per_session": {s: folds[s]["summary"] for s in SESSIONS},
        "pooled_out_of_fold": pooled,
        "pair_stability": stability,
    }

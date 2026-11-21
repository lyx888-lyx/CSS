"""Protocol and metric helpers for a ONE-TIME exploratory V11 official TEST evaluation.

The models were trained on IEMOCAP sessions 1-3 and selected on session 4.
The official Session-5 TEST is NOT used to select arms or checkpoints.
A single test run here consumes independent TEST status for later V11 tuning.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

NAMES = ("happy", "sad", "neutral", "angry", "excited", "frustrated")
ARMS = ("original", "self", "reaction")
SEED = 61078


def validate_saved_checkpoint(obj, arm, train_indices, valid_indices):
    """Fail before TEST access unless ALL three checkpoints are comparable."""
    if arm not in ARMS:
        raise ValueError("invalid arm")
    if not isinstance(obj, dict):
        raise ValueError("checkpoint must be a dictionary")
    cfg = obj.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("missing checkpoint configuration")
    required = {
        "dataset": "IEMOCAP", "seed": SEED, "epochs": 200,
        "v10_feature_mode": "original", "v10_validation_session": 4,
        "speaker_mode": "off", "precontext_mode": "off",
        "v61_spf_mode": "full", "v52_pgm_mode": "original",
        "v8_spf_mode": "off", "classes": 6, "speakers": 2,
    }
    for key, expected in required.items():
        got = cfg.get(key, "off" if key in (
            "precontext_mode", "v52_pgm_mode", "v8_spf_mode") else None)
        if got != expected:
            raise ValueError(f"{arm}: {key} should be {expected!r}, got {got!r}")
    if cfg.get("v11_acoustic_mode", "off") != (
            "off" if arm == "original" else arm):
        raise ValueError(f"{arm}: wrong V11 architecture type")
    if (obj.get("selection") != "validation_weighted_f1" or
            list(obj.get("train_dialogue_indices", [])) != train_indices or
            list(obj.get("valid_dialogue_indices", [])) != valid_indices):
        raise ValueError(f"{arm}: mismatched TRAIN/VALID split or selection")
    if "model_state_dict" not in obj or not obj["model_state_dict"]:
        raise ValueError(f"{arm}: missing trained state_dict")
    if not isinstance(obj.get("epoch"), int) or not (1 <= obj["epoch"] <= 200):
        raise ValueError(f"{arm}: invalid validation-selected epoch")
    if not isinstance(obj.get("validation_f1"), (int, float)):
        raise ValueError(f"{arm}: missing saved validation F1")
    return {
        "arm": arm,
        "selected_epoch": int(obj["epoch"]),
        "validation_f1": float(obj["validation_f1"]),
        "training_sessions": [1, 2, 3],
        "validation_session": 4,
        "selection": "validation_weighted_f1",
    }


def test_metrics(y, pred, class_names=NAMES):
    """Match CSS's standard top-1 Accuracy and weighted F1 definitions."""
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    n = len(class_names)
    if y.ndim != 1 or y.shape != pred.shape or len(y) == 0:
        raise ValueError("nonempty aligned 1D predictions required")
    if ((y < 0).any() or (y >= n).any() or
            (pred < 0).any() or (pred >= n).any()):
        raise ValueError("invalid class index")
    p, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(n)), zero_division=0)
    out = {
        "utterances": int(len(y)),
        "accuracy": float(100 * accuracy_score(y, pred)),
        "weighted_f1": float(100 * f1_score(
            y, pred, average="weighted", labels=list(range(n)),
            zero_division=0)),
        "macro_f1": float(100 * f1_score(
            y, pred, average="macro", labels=list(range(n)),
            zero_division=0)),
        "class_metrics": {
            str(i): {
                "name": name, "support": int(support[i]),
                "accuracy_recall": float(100 * recall[i]),
                "precision": float(100 * p[i]),
                "f1": float(100 * f1[i]),
            }
            for i, name in enumerate(class_names)
        },
    }
    return out


def paired_changes(y, candidate, reference):
    """Count fixed/broken correctness separately from ALL raw class flips."""
    y = np.asarray(y, dtype=np.int64)
    left = np.asarray(candidate, dtype=np.int64)
    right = np.asarray(reference, dtype=np.int64)
    if y.shape != left.shape or y.shape != right.shape:
        raise ValueError("unaligned predictions")
    lo, ro = (left == y), (right == y)
    return {
        "fixed": int((lo & ~ro).sum()),
        "broken": int((~lo & ro).sum()),
        "net_more_correct": int(lo.sum() - ro.sum()),
        "top1_flips": int((left != right).sum()),
    }

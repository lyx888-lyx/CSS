#!/usr/bin/env python3
"""V4 read-only MELD class confusion audit for SAME CSS VALIDATION dialogues.

No training. No evaluation of official TEST. No hyperparameter search.
Uses only the original MELD TRAIN-partition 1037/115 split defined
by split seed 10068. Loads one or two trusted locally created CSS
research checkpoints and compares classwise error destinations.

Motivation: the V3 class-balanced SupCon pilot reduced overall weighted
F1 by 0.23 pp; low-support fear/disgust had poor validation F1. Before
proposing an emotion-geometry or balanced-classifier V4, first check if
these groups are under-predicted or simply confused with similar labels.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from scripts.compare_validation import read_checkpoint, predict_valid
from scripts.train_research import build_model, dialogue_split_indices


# MELD official labelling map (https://github.com/declare-lab/MELD):
# neutral 0, surprise 1, fear 2, sadness 3, joy 4, disgust 5, anger 6.
NAMES = ("Neutral", "Surprise", "Fear", "Sadness",
         "Joy", "Disgust", "Anger")
N_CLASSES = len(NAMES)


def summarize_confusion(y, pred):
    """Numerical classwise report with real TOP confusion destinations."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("labels/predictions must be aligned nonempty 1D arrays")
    if ((y < 0) | (y >= N_CLASSES)).any() or ((p < 0) | (p >= N_CLASSES)).any():
        raise ValueError("unexpected MELD class label (expected 0-6)")
    cm = confusion_matrix(y, p, labels=list(range(N_CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=list(range(N_CLASSES)), zero_division=0
    )
    per_class = {}
    for i, name in enumerate(NAMES):
        mistakes = [
            {"predicted_class": j, "predicted_name": NAMES[j],
             "count": int(cm[i, j])}
            for j in range(N_CLASSES) if j != i and cm[i, j] > 0
        ]
        mistakes.sort(key=lambda m: (-m["count"], m["predicted_class"]))
        per_class[str(i)] = {
            "name": name, "support": int(support[i]),
            "predicted_count": int(cm[:, i].sum()),
            "correct": int(cm[i, i]),
            "precision": round(100 * float(precision[i]), 2),
            "recall": round(100 * float(recall[i]), 2),
            "f1": round(100 * float(f1[i]), 2),
            "mistakes": int(cm[i, :].sum() - cm[i, i]),
            "top_confusions": mistakes[:3],
        }
    return {
        "n": len(y),
        "confusion_true_rows_pred_columns": cm.tolist(),
        "class_names": list(NAMES),
        "per_class": per_class,
    }


def same_train_validation_split(ckpt, dataset):
    cfg = ckpt["config"]
    if cfg.get("dataset") != "MELD":
        raise ValueError("expected MELD checkpoint")
    if ckpt.get("selection") != "validation_weighted_f1":
        raise ValueError("must use validation-selected checkpoint, not TEST")
    valid_indices = list(map(int, ckpt["valid_dialogue_indices"]))
    tr, val = dialogue_split_indices(
        len(dataset), 0.1, 10068
    )
    if valid_indices != val:
        raise ValueError("not the fixed MELD split-seed 10068 validation cohort")
    if "train_dialogue_indices" in ckpt:
        if list(map(int, ckpt["train_dialogue_indices"])) != tr:
            raise ValueError("saved train split differs from fixed split")
    if len(val) != 115 or len(tr) != 1037:
        raise ValueError("unexpected MELD training/validation dialogue count")
    if cfg.get("split_seed", cfg.get("seed")) != 10068:
        raise ValueError("unexpected split seed in checkpoint metadata")
    return tr, val


@torch.no_grad()
def predict_checkpoint(path, dataset, expected_val, device):
    ckpt = read_checkpoint(path)
    _, val = same_train_validation_split(ckpt, dataset)
    if val != expected_val:
        raise ValueError("candidate validation cohort differs from baseline")
    cfg = ckpt["config"]
    if cfg.get("speaker_mode", "off") != "off":
        raise ValueError("expected original CSS architecture")
    if cfg.get("precontext_mode", "off") != "off":
        raise ValueError("expected original CSS architecture (no precontext V1)")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    loader = DataLoader(
        Subset(dataset, val),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    y, p, tags = predict_valid(model, loader, device)
    if len(y) != 1044:
        raise ValueError("expected 1044 fixed validation utterances")
    result = {
        "checkpoint": str(path), "selected_epoch": int(ckpt["epoch"]),
        "supcon_weight": cfg.get("v3_supcon_weight", 0.0),
        "train_seed": cfg["seed"], "split_seed": cfg.get("split_seed", cfg["seed"]),
        "report": summarize_confusion(y, p),
        "emotion_shift_subgroups": {
            group: summarize_confusion(
                [y[i] for i, tag in enumerate(tags) if group in tag],
                [p[i] for i, tag in enumerate(tags) if group in tag],
            ) if any(group in tag for tag in tags) else None
            for group in ("stable", "changed", "changed_after_other")
        },
    }
    del model, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y, p, tags, result


def train_class_counts(dataset, train_indices):
    counts = Counter()
    for index in train_indices:
        vid = dataset.keys[index]
        counts.update(map(int, dataset.videoLabels[vid]))
    return {
        str(i): {"name": name, "n": int(counts.get(i, 0))}
        for i, name in enumerate(NAMES)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--data", type=Path,
                        default=Path("data/meld_multimodal_features.pkl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/v4_meld_class_confusions.json"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for path in (args.baseline, args.candidate, args.data):
        if path is not None and not path.is_file():
            parser.error(f"file not found: {path}")

    ds = MELDDataset(str(args.data), train=True)  # NEVER train=False
    tr_idx, val_idx = dialogue_split_indices(len(ds), 0.1, 10068)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    y0, p0, t0, base = predict_checkpoint(
        args.baseline, ds, val_idx, device
    )
    data = {
        "protocol": {
            "dataset": "MELD", "train_dialogues": len(tr_idx),
            "validation_dialogues": len(val_idx), "split_seed": 10068,
            "official_test_evaluated": False,
            "warning": (
                "Validation data have been repeatedly inspected during "
                "V0/V1/V2/V3. This is diagnostic, not independent model "
                "selection or a confirmatory performance claim."
            ),
        },
        "train_class_support": train_class_counts(ds, tr_idx),
        "baseline": base,
    }
    if args.candidate is not None:
        y1, p1, t1, candidate = predict_checkpoint(
            args.candidate, ds, val_idx, device
        )
        if y1 != y0 or t1 != t0:
            raise RuntimeError("candidate utterance cohort/ordering mismatch")
        candidate["net_correct_gain"] = int(sum(
            a != y and b == y for a, b, y in zip(p0, p1, y0)
        ) - sum(
            a == y and b != y for a, b, y in zip(p0, p1, y0)
        ))
        data["candidate"] = candidate

    print("V4 CLASS CONFUSION AUDIT — MELD VALIDATION ONLY; TEST UNTOUCHED")
    print("Train class counts:", [
        data["train_class_support"][str(i)]["n"] for i in range(N_CLASSES)
    ])
    print("Label  Name       N-train N-valid Predicted Correct Precision Recall F1")
    for i, name in enumerate(NAMES):
        x = base["report"]["per_class"][str(i)]
        n_train = data["train_class_support"][str(i)]["n"]
        print(f"{i:>5}  {name:<9}{n_train:>7} {x['support']:>7} "
              f"{x['predicted_count']:>9} {x['correct']:>7} "
              f"{x['precision']:>9.2f} {x['recall']:>6.2f} {x['f1']:>6.2f}")
    for i in (2, 5, 3, 6, 0):
        x = base["report"]["per_class"][str(i)]
        print(f"\nTRUE {NAMES[i]} ({x['support']} VALID), top wrong predictions:")
        for item in x["top_confusions"]:
            print(f"  -> {item['predicted_name']}: {item['count']}")
        if not x["top_confusions"]:
            print("  (no misclassifications)")
    if args.candidate is not None:
        print("\nCSS vs V3 trained model, per-class F1 delta:")
        for i, name in enumerate(NAMES):
            a = base["report"]["per_class"][str(i)]["f1"]
            b = data["candidate"]["report"]["per_class"][str(i)]["f1"]
            print(f"  {i} {name:<9} CSS {a:.2f} V3 {b:.2f} Δ{b-a:+.2f}pp")
        print("Net correct predictions:",
              data["candidate"]["net_correct_gain"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Saved:", args.output)
    print("No training/TEST or validation-based threshold tuning was done.")


if __name__ == "__main__":
    main()

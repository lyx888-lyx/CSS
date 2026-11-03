#!/usr/bin/env python3
"""Analyze paired validation errors: CSS vs pre-context V1.

This is a READ-ONLY diagnostic of two already trained and validation-selected
research checkpoints. It does not train, tune, or access official TEST data.
It compares each matched utterance's prediction to the gold validation label.

CAUTION: tags such as 'changed_after_other' use true labels only AFTER both
models have already predicted. Their differences are descriptive, not causal.
Only load trusted, locally generated PyTorch checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataloader import IEMOCAPDataset, MELDDataset
from scripts.compare_validation import (
    compute_report, predict_valid, read_checkpoint,
)
from scripts.train_research import build_model


GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other",
)


def correctness_changes(y, css_pred, v1_pred, tags, indices=None):
    """Prediction changes in a subset, with no model or dataset dependency.

    'fixed' = CSS wrong and V1 correct.
    'broken' = CSS correct and V1 wrong.
    'both_wrong_changed_answer' = neither correct but predictions differ.
    """
    n = len(y)
    if len(css_pred) != n or len(v1_pred) != n or len(tags) != n:
        raise ValueError("predictions/labels/tags are not aligned")
    if indices is None:
        indices = range(n)
    indices = list(indices)
    counts = {
        "n": len(indices), "css_correct": 0, "v1_correct": 0,
        "fixed": 0, "broken": 0, "both_correct": 0, "both_wrong": 0,
        "both_wrong_changed_answer": 0, "predicted_label_changed": 0,
    }
    for i in indices:
        correct_a = int(css_pred[i] == y[i])
        correct_b = int(v1_pred[i] == y[i])
        counts["css_correct"] += correct_a
        counts["v1_correct"] += correct_b
        counts["fixed"] += int(not correct_a and correct_b)
        counts["broken"] += int(correct_a and not correct_b)
        counts["both_correct"] += int(correct_a and correct_b)
        counts["both_wrong"] += int(not correct_a and not correct_b)
        changed = css_pred[i] != v1_pred[i]
        counts["predicted_label_changed"] += int(changed)
        counts["both_wrong_changed_answer"] += int(
            changed and not correct_a and not correct_b
        )
    counts["net_correct_change"] = counts["fixed"] - counts["broken"]
    counts["accuracy_delta_pp"] = (
        round(100 * counts["net_correct_change"] / len(indices), 2)
        if indices else None
    )
    return counts


def per_group_changes(y, css_pred, v1_pred, tags, n_classes):
    """Add subgroup/class counts and weighted F1 without retraining."""
    summary = {
        "model_scores": {
            "css": compute_report(y, css_pred, tags, n_classes),
            "v1": compute_report(y, v1_pred, tags, n_classes),
        },
        "groups": {},
        "by_true_emotion_class": {},
    }
    for group in GROUPS:
        ids = [
            i for i, tt in enumerate(tags)
            if group == "all" or group in tt
        ]
        summary["groups"][group] = correctness_changes(
            y, css_pred, v1_pred, tags, indices=ids
        )
    for class_idx in range(n_classes):
        ids = [i for i, lab in enumerate(y) if lab == class_idx]
        summary["by_true_emotion_class"][str(class_idx)] = correctness_changes(
            y, css_pred, v1_pred, tags, indices=ids
        )
    return summary


@torch.no_grad()
def predict_checkpoint(path, loader, device):
    ckpt = read_checkpoint(path)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    try:
        y, prediction, tags = predict_valid(model, loader, device)
    finally:
        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return y, prediction, tags


def analyze_pair(css_path, v1_path, device):
    a = read_checkpoint(css_path)
    b = read_checkpoint(v1_path)
    ca, cb = a["config"], b["config"]
    keys = ("dataset", "seed", "classes", "batch_size")
    if any(ca[k] != cb[k] for k in keys):
        raise ValueError("CSS/V1 checkpoints do not share dataset, seed, class count, batch size")
    if ca.get("split_seed", ca["seed"]) != cb.get("split_seed", cb["seed"]):
        raise ValueError("different validation split seed")
    if a["valid_dialogue_indices"] != b["valid_dialogue_indices"]:
        raise ValueError("different validation dialogue indices")
    if ca.get("precontext_mode", "off") != "off":
        raise ValueError("first checkpoint should be CSS with V1 disabled")
    if cb.get("precontext_mode", "off") != "both":
        raise ValueError("second checkpoint should be V1 precontext both")
    if ca["speaker_mode"] != "off" or cb["speaker_mode"] != "off":
        raise ValueError("V0 must be disabled in this experiment")
    valid_idx = a["valid_dialogue_indices"]
    if ca["dataset"] == "MELD":
        ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    elif ca["dataset"] == "IEMOCAP":
        ds = IEMOCAPDataset(train=True)
    else:
        raise ValueError("unknown dataset")
    if any(i < 0 or i >= len(ds) for i in valid_idx):
        raise ValueError("out-of-range validation dialogue index")
    loader = DataLoader(
        Subset(ds, valid_idx), batch_size=ca["batch_size"],
        shuffle=False, collate_fn=ds.collate_fn
    )
    del a, b
    y1, p1, t1 = predict_checkpoint(css_path, loader, device)
    y2, p2, t2 = predict_checkpoint(v1_path, loader, device)
    if y1 != y2 or t1 != t2:
        raise RuntimeError("validation utterances or groups did not align")
    result = per_group_changes(y1, p1, p2, t1, ca["classes"])
    for name, save_path in (("css", css_path), ("v1", v1_path)):
        result["model_scores"][name]["checkpoint"] = str(save_path)
    result["metadata"] = {
        "dataset": ca["dataset"], "train_seed": ca["seed"],
        "split_seed": ca.get("split_seed", ca["seed"]),
        "validation_dialogues": len(valid_idx),
        "validation_utterances": len(y1),
        "test_evaluated": False,
        "meaning": "One-seed, post-hoc validation error analysis; no causal or statistical significance claim",
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--css", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/v1_paired_errors.json"))
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()
    for path in (args.css, args.v1):
        if not path.is_file():
            p.error(f"checkpoint does not exist: {path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    result = analyze_pair(args.css, args.v1, device)
    print("\nCSS -> V1: error migration on the SAME validation utterances")
    print("Group                N    fixed   broken   net   pred_changed")
    for group in GROUPS:
        x = result["groups"][group]
        print(
            f"{group:<21}{x['n']:>4d} {x['fixed']:>8d} "
            f"{x['broken']:>8d} {x['net_correct_change']:>+5d} "
            f"{x['predicted_label_changed']:>13d}"
        )
    print("\nBy true emotion label (MELD: Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger)")
    print("Label            N    fixed   broken   net")
    for label, x in result["by_true_emotion_class"].items():
        print(
            f"{label:<13}{x['n']:>5d} {x['fixed']:>8d} "
            f"{x['broken']:>8d} {x['net_correct_change']:>+5d}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved validation-only paired-error report: {args.output}")


if __name__ == "__main__":
    main()

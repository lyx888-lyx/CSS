#!/usr/bin/env python3
"""V4 read-only, ONE fixed train-prior logit correction on CSS.

Goal: assess whether severe Fear/Disgust underprediction is partly
explained by a high-frequency-class logit prior. Standard post-hoc
logit adjustment (Menon et al., ICLR 2021), NOT an original method.

No retraining, no official TEST, no tuning against VALID labels.
The strength 0.15 was fixed before collecting this script's results.
For all classes equally:
  z_corrected[c] = log(p_css[c]) - 0.15 * log(train_prior[c]).
Adding any class-independent constant leaves argmax unchanged.
Score baseline and this ONE adjusted predictor on the SAME
1,044 held-out TRAIN-partition validation utterances.

CAUTION: long-tail rebalancing generally targets balanced error,
not necessarily MELD weighted-F1; a gain is not guaranteed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from scripts.audit_meld_class_confusions import (
    NAMES, summarize_confusion, same_train_validation_split
)
from scripts.compare_validation import (
    compute_report, group_labels, read_checkpoint
)
from scripts.diagnose_v1_errors import correctness_changes
from scripts.train_research import build_model, dialogue_split_indices


# ONE strength fixed before inspecting candidate VALID scores.
FIXED_TAU = 0.15
GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other",
)


def training_priors(dataset, train_indices):
    """Compute priors from ORIGINAL TRAIN dialogues only."""
    counts = np.zeros(len(NAMES), dtype=np.int64)
    for i in train_indices:
        key = dataset.keys[int(i)]
        labels = np.asarray(dataset.videoLabels[key], dtype=np.int64)
        if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= len(NAMES)):
            raise ValueError("invalid training emotion labels")
        counts += np.bincount(labels, minlength=len(NAMES))
    if counts.min() < 1:
        raise ValueError("all seven classes require positive TRAIN support")
    return counts / counts.sum(), counts


def fixed_logit_adjustment(probs, priors, tau=FIXED_TAU):
    """Raw probabilities -> class predictions using TRAIN priors only."""
    p = np.asarray(probs, dtype=np.float64)
    q = np.asarray(priors, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(NAMES):
        raise ValueError("expected probability rows [N,7]")
    if q.shape != (len(NAMES),):
        raise ValueError("expected seven class priors")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("all probabilities/priors must be finite")
    if (p < 0).any() or (np.abs(p.sum(-1) - 1) > 1e-4).any():
        raise ValueError("CSS probabilities must be nonnegative and normalized")
    if (q <= 0).any() or abs(q.sum() - 1) > 1e-6:
        raise ValueError("train priors must be strictly positive, normalized")
    if not (0 <= tau <= 1):
        raise ValueError("tau outside [0,1]")
    scores = np.log(np.clip(p, 1e-15, 1)) - tau * np.log(q)[None, :]
    return scores.argmax(-1).astype(int)


def paired_report(y, before, after, tags):
    scores_a = compute_report(y, before, tags, len(NAMES))
    scores_b = compute_report(y, after, tags, len(NAMES))
    rows = {}
    for group in GROUPS:
        indices = [
            i for i, tag in enumerate(tags)
            if group == "all" or group in tag
        ]
        move = correctness_changes(y, before, after, tags, indices=indices)
        rows[group] = {
            "original_weighted_f1": scores_a[group]["weighted_f1"],
            "adjusted_weighted_f1": scores_b[group]["weighted_f1"],
            "delta_weighted_f1_pp": round(
                scores_b[group]["weighted_f1"] - scores_a[group]["weighted_f1"], 2
            ) if scores_a[group]["weighted_f1"] is not None else None,
            **move,
        }
    return scores_a, scores_b, rows


@torch.no_grad()
def collect_baseline_predictions(checkpoint, dataset, valid_indices, device):
    ckpt = read_checkpoint(checkpoint)
    _, saved_valid = same_train_validation_split(ckpt, dataset)
    if saved_valid != valid_indices:
        raise ValueError("checkpoint and analysis VALID cohorts differ")
    config = ckpt["config"]
    if (config.get("speaker_mode", "off") != "off"
            or config.get("precontext_mode", "off") != "off"
            or config.get("v3_supcon_weight", 0.0) != 0):
        raise ValueError("original CSS checkpoint required, not V1 or V3")
    model = build_model(config).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    loader = DataLoader(
        Subset(dataset, valid_indices),
        batch_size=config["batch_size"],
        shuffle=False, collate_fn=dataset.collate_fn, num_workers=0,
    )
    probabilities, labels, tags = [], [], []
    for batch in loader:
        text, visual, audio, qmask, umask, gold, _ = batch
        lengths = umask.sum(-1).long().tolist()
        p = model(
            text.to(device), visual.to(device), audio.to(device),
            umask.to(device), qmask.permute(1, 0, 2).to(device),
            lengths
        )[4].detach().cpu().numpy()
        who = qmask.argmax(-1).transpose(0, 1).cpu().numpy()
        yy = gold.cpu().numpy()
        for b, n in enumerate(lengths):
            labels.extend(yy[b, :n].astype(int).tolist())
            probabilities.append(p[b, :n])
            tags.extend(group_labels(
                yy[b, :n].astype(int).tolist(), who[b, :n].tolist()
            ))
    probs = np.concatenate(probabilities, axis=0)
    if probs.shape != (1044, len(NAMES)) or len(labels) != len(tags):
        raise ValueError("expected precisely 1044 MELD validation utterances")
    return probs, labels, tags, {
        "checkpoint": str(checkpoint),
        "selected_epoch": int(ckpt["epoch"]),
        "train_seed": int(config["seed"]),
        "split_seed": int(config.get("split_seed", config["seed"])),
    }


def run(checkpoint, device):
    dataset = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    train_idx, valid_idx = dialogue_split_indices(len(dataset), 0.1, 10068)
    priors, counts = training_priors(dataset, train_idx)
    p, y, tags, ckpt = collect_baseline_predictions(
        checkpoint, dataset, valid_idx, device
    )
    original = p.argmax(-1)
    adjusted = fixed_logit_adjustment(p, priors)
    baseline_scores, adjusted_scores, paired = paired_report(
        y, original, adjusted, tags
    )
    confusion_a = summarize_confusion(y, original)
    confusion_b = summarize_confusion(y, adjusted)
    moved_to_class = {
        str(c): {
            "name": NAMES[c],
            "original_predicted": int((original == c).sum()),
            "corrected_predicted": int((adjusted == c).sum()),
            "original_true_positives": int(
                ((original == c) & (np.asarray(y) == c)).sum()
            ),
            "corrected_true_positives": int(
                ((adjusted == c) & (np.asarray(y) == c)).sum()
            ),
        } for c in range(len(NAMES))
    }
    result = {
        "protocol": {
            "dataset": "MELD", "train_dialogues": len(train_idx),
            "validation_dialogues": len(valid_idx),
            "validation_utterances": len(y),
            "split_seed": 10068, "official_test_evaluated": False,
            "baseline_checkpoint": ckpt,
            "method": "ICLR 2021 post-hoc prior logit adjustment",
            "fixed_tau": FIXED_TAU,
            "train_class_counts": counts.astype(int).tolist(),
            "train_class_priors": priors.tolist(),
            "warning": (
                "ONE fixed train-prior correction; not trained or tuned on "
                "VALID labels. This VALID cohort has been inspected across "
                "earlier V0-V3 experiments, so any gain is exploratory only. "
                "Balanced classification is not equivalent to weighted F1."
            ),
        },
        "original_css": baseline_scores,
        "prior_adjusted_css": adjusted_scores,
        "paired_migration": paired,
        "classwise_original": confusion_a["per_class"],
        "classwise_adjusted": confusion_b["per_class"],
        "class_prediction_counts": moved_to_class,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="original CSS strict VALIDATION-best checkpoint"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v4_fixed_prior_adjustment.json")
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint missing: " + str(args.checkpoint))
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    report = run(args.checkpoint, device)
    b = report["original_css"]
    a = report["prior_adjusted_css"]
    print("V4 ONE FIXED PRIOR LOGIT ADJUSTMENT | VALIDATION ONLY, TEST UNTOUCHED")
    print("Fixed tau = 0.15; prior from 1037 TRAIN dialogues only; no retraining")
    print("Group                  CSS W-F1    Prior W-F1    Delta   fixed broken")
    for group in GROUPS:
        x = report["paired_migration"][group]
        print(f"{group:<23}{x['original_weighted_f1']:>7.2f}% "
              f"{x['adjusted_weighted_f1']:>12.2f}% "
              f"{x['delta_weighted_f1_pp']:>+8.2f}pp "
              f"{x['fixed']:>5} {x['broken']:>6}")
    print("\nClass         N-true  Pred before/after  F1 before/after  Recall before/after")
    for i, name in enumerate(NAMES):
        key = str(i)
        x = report["classwise_original"][key]
        v = report["classwise_adjusted"][key]
        print(
            f"{name:<12}{x['support']:>6}  "
            f"{x['predicted_count']:>4}/{v['predicted_count']:<5} "
            f"{x['f1']:>6.2f}/{v['f1']:<6.2f} "
            f"{x['recall']:>6.2f}/{v['recall']:<6.2f}"
        )
    print(f"Net correct labels: {report['paired_migration']['all']['net_correct_change']:+d}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("Saved:", args.output)
    print("Stop rule: if overall or stable drops, no tau sweep on this VALID cohort.")


if __name__ == "__main__":
    main()

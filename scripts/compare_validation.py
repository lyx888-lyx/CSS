#!/usr/bin/env python3
"""Compare CSS research checkpoints on the SAME held-out validation dialogues.

This report never accesses official test dialogues. True validation labels are
used only AFTER predictions, to partition scores into stable/changed emotion
groups. Do not use any ground-truth emotion label as a model input.

Run from repo root:
  python -m scripts.compare_validation --checkpoints checkpoints/....pt ...
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from dataloader import IEMOCAPDataset, MELDDataset
from scripts.train_research import build_model


def group_labels(y, speaker_ids):
    """Return per-utterance analysis categories for one REAL dialogue.

    'first': target speaker has not yet spoken.
    'stable': target emotion equals their previous emotion.
    'changed': target emotion differs from previous emotion.
    '..._after_other': same, but at least one other speaker spoke
      between the target speaker's two most recent turns.
    """
    previous = {}
    categories = []
    for i, (label, speaker) in enumerate(zip(y, speaker_ids)):
        speaker = int(speaker)
        label = int(label)
        if speaker not in previous:
            categories.append(("first",))
        else:
            j = previous[speaker]
            stable = label == int(y[j])
            status = "stable" if stable else "changed"
            extras = (status + "_after_other",) if i - j > 1 else ()
            categories.append((status,) + extras)
        previous[speaker] = i
    return categories


def compute_report(y, pred, tags, n_classes):
    """Compute accuracy and fixed-label weighted/macro-F1 per subgroup."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if y.shape != pred.shape or len(tags) != len(y):
        raise ValueError("prediction/label/tag lengths do not match")
    categories = ("all", "first", "stable", "changed",
                  "stable_after_other", "changed_after_other")
    groups = {}
    for name in categories:
        ids = [i for i, tt in enumerate(tags) if name == "all" or name in tt]
        if not ids:
            groups[name] = {"n": 0, "accuracy": None,
                            "weighted_f1": None, "macro_f1": None}
            continue
        yt, yp = y[ids], pred[ids]
        groups[name] = {
            "n": len(ids),
            "accuracy": round(accuracy_score(yt, yp) * 100, 2),
            "weighted_f1": round(f1_score(
                yt, yp, average="weighted", labels=list(range(n_classes)),
                zero_division=0
            ) * 100, 2),
            "macro_f1": round(f1_score(
                yt, yp, average="macro", labels=list(range(n_classes)),
                zero_division=0
            ) * 100, 2),
        }
    by_class = f1_score(
        y, pred, labels=list(range(n_classes)),
        average=None, zero_division=0
    )
    support = Counter(map(int, y))
    groups["per_class"] = {
        str(k): {"n": support.get(k, 0), "f1": round(v * 100, 2)}
        for k, v in enumerate(by_class)
    }
    return groups


def speaker_identity(one_hot):
    """One-hot inputs from IEMOCAP/MELD, [length, n_speakers]."""
    a = np.asarray(one_hot)
    if a.ndim != 2:
        raise ValueError("expected one-hot speaker IDs [L, S]")
    return a.argmax(-1).astype(int).tolist()


def validate_and_load_checkpoints(paths):
    ckpts = []
    for path in paths:
        # Only load research checkpoints YOU generated: torch.load unpickles data.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "valid_dialogue_indices", "config"):
            if key not in obj:
                raise ValueError(f"{path}: missing {key}")
        ckpts.append(obj)

    reference = ckpts[0]
    ref_config = reference["config"]
    ref_indices = reference["valid_dialogue_indices"]
    for path, obj in zip(paths[1:], ckpts[1:]):
        if obj["config"]["dataset"] != ref_config["dataset"]:
            raise ValueError(f"{path}: inconsistent dataset")
        if obj["valid_dialogue_indices"] != ref_indices:
            raise ValueError(f"{path}: inconsistent validation split")
        if obj["config"]["classes"] != ref_config["classes"]:
            raise ValueError(f"{path}: inconsistent label count")
        if obj["config"]["seed"] != ref_config["seed"]:
            raise ValueError(f"{path}: inconsistent seed")
    return ckpts


@torch.no_grad()
def predict_valid(model, loader, device):
    model.eval()
    true_labels, predictions, categories = [], [], []
    for batch in loader:
        text, visual, audio, qmask, umask, label, _ = batch
        speaker_ids = qmask.argmax(-1).transpose(0, 1).cpu().numpy()
        lengths = umask.sum(dim=1).long().tolist()
        data = (text.to(device), visual.to(device), audio.to(device),
                umask.to(device), qmask.permute(1, 0, 2).to(device), lengths)
        probs = model(*data)[4]
        pred = probs.argmax(dim=-1).cpu().numpy()
        labels = label.cpu().numpy()

        for i, size in enumerate(lengths):
            if not size:
                raise ValueError("empty validation dialogue")
            y = labels[i, :size].astype(int).tolist()
            p = pred[i, :size].astype(int).tolist()
            sp = speaker_ids[i, :size].tolist()
            true_labels.extend(y)
            predictions.extend(p)
            categories.extend(group_labels(y, sp))
    return true_labels, predictions, categories


def analyze_checkpoints(paths, device):
    ckpts = validate_and_load_checkpoints(paths)
    config = ckpts[0]["config"]
    if config["dataset"] == "MELD":
        ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    elif config["dataset"] == "IEMOCAP":
        ds = IEMOCAPDataset(train=True)
    else:
        raise ValueError("unsupported dataset")
    loader = DataLoader(
        Subset(ds, ckpts[0]["valid_dialogue_indices"]),
        batch_size=config["batch_size"], shuffle=False, collate_fn=ds.collate_fn
    )
    report = {
        "dataset": config["dataset"],
        "split": "validation_only",
        "seed": config["seed"],
        "dialogues": len(ckpts[0]["valid_dialogue_indices"]),
        "notice": ("Per-group labels are used only for post-prediction analysis. "
                   "Differences do not establish causality or robustness."),
        "models": [],
    }
    reference_y, reference_tags = None, None
    for path, ckpt in zip(paths, ckpts):
        model = build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        y, pred, tags = predict_valid(model, loader, device)
        if reference_y is None:
            reference_y, reference_tags = y, tags
        elif reference_y != y or reference_tags != tags:
            raise RuntimeError("misaligned validation utterances")
        scores = compute_report(y, pred, tags, config["classes"])
        report["models"].append({
            "checkpoint": str(path),
            "speaker_mode": ckpt["config"]["speaker_mode"],
            "time_decay": ckpt["config"]["time_decay"],
            "selected_epoch": ckpt["epoch"],
            "saved_validation_f1": ckpt["validation_f1"],
            "scores": scores,
        })
        print(
            f'{Path(path).name}: valid F1={scores["all"]["weighted_f1"]:.2f} '
            f'changed n={scores["changed"]["n"]} '
            f'changed F1={scores["changed"]["weighted_f1"]:.2f} '
            f'stable F1={scores["stable"]["weighted_f1"]:.2f}',
            flush=True
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/meld_validation_ablation.json"))
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    missing = [p for p in args.checkpoints if not p.is_file()]
    if missing:
        parser.error(f"missing checkpoints: {missing}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    report = analyze_checkpoints(args.checkpoints, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        report, ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print(f"Saved VALIDATION-only report: {args.output}")


if __name__ == "__main__":
    main()

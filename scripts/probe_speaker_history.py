#!/usr/bin/env python3
"""Validation-only diagnostic: how much does trained speaker-history fusion do?

Given ONE validation-selected research checkpoint, compare its predictions
with the same model and same parameters, but speaker_history temporarily
disabled at inference. This is an *inference-time intervention*, NOT a
fairly retrained CSS ablation. No official test utterances are read.
Only use trusted self-created checkpoints (torch.load uses pickle).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataloader import IEMOCAPDataset, MELDDataset
from research.speaker_history import build_speaker_history_masks
from scripts.compare_validation import compute_report, predict_valid
from scripts.train_research import build_model


def get_validation_loader(ckpt):
    config = ckpt["config"]
    if config["dataset"] == "MELD":
        ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    elif config["dataset"] == "IEMOCAP":
        ds = IEMOCAPDataset(train=True)
    else:
        raise ValueError("unknown dataset")
    indices = ckpt["valid_dialogue_indices"]
    if not indices or any(i < 0 or i >= len(ds) for i in indices):
        raise ValueError("invalid validation dialogue indices")
    loader = DataLoader(
        Subset(ds, indices), batch_size=config["batch_size"], shuffle=False,
        collate_fn=ds.collate_fn, num_workers=0
    )
    return loader


def change_report(y, p1, p0, categories, n_classes):
    if len(y) != len(p1) or len(y) != len(p0):
        raise ValueError("misaligned predictions")
    r1 = compute_report(y, p1, categories, n_classes)
    r0 = compute_report(y, p0, categories, n_classes)
    changed = int(np.count_nonzero(np.asarray(p1) != np.asarray(p0)))
    return {
        "with_module": r1,
        "same_checkpoint_without_module_at_inference": r0,
        "predictions_changed_when_module_removed": changed,
        "total_validation_utterances": len(y),
        "changed_prediction_percent": round(100 * changed / len(y), 2),
    }


@torch.no_grad()
def run(ckpt, device):
    conf = ckpt["config"]
    if conf.get("speaker_mode") in (None, "off"):
        raise ValueError("checkpoint must have speaker-history module enabled")
    loader = get_validation_loader(ckpt)
    model = build_model(conf).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    stats = {"ratios": [], "history_ratios": []}

    def collect_ratios(module, inputs, output):
        h, speaker_ids, valid_mask = inputs
        valid = valid_mask.bool()
        numerator = (output - h).norm(dim=-1)
        denominator = h.norm(dim=-1).clamp_min(1e-8)
        ratios = (numerator / denominator).detach()
        stats["ratios"].append(ratios[valid].cpu())
        same, other = build_speaker_history_masks(
            speaker_ids, valid, max_lag=module.max_lag
        )
        has_history = (same | other).any(dim=-1) & valid
        stats["history_ratios"].append(ratios[has_history].cpu())

    original_module = model.speaker_history
    handle = original_module.register_forward_hook(collect_ratios)
    try:
        labels, pred_with, tags = predict_valid(model, loader, device)
    finally:
        handle.remove()

    # Remove the module AFTER full inference; no weights are retrained.
    model.speaker_history = None
    try:
        labels0, pred_without, tags0 = predict_valid(model, loader, device)
    finally:
        model.speaker_history = original_module
    if labels != labels0 or tags != tags0:
        raise RuntimeError("validation order and sample groups did not match")

    report = change_report(
        labels, pred_with, pred_without, tags, conf["classes"]
    )
    all_ratios = torch.cat(stats["ratios"])
    history_ratios = torch.cat(stats["history_ratios"])
    report["residual_delta_over_input_norm"] = {
        "all_real_utterances_mean": round(all_ratios.mean().item(), 6),
        "all_real_utterances_median": round(all_ratios.median().item(), 6),
        "with_eligible_history_count": int(history_ratios.numel()),
        "with_eligible_history_mean": (
            round(history_ratios.mean().item(), 6)
            if history_ratios.numel() else None
        ),
        "with_eligible_history_median": (
            round(history_ratios.median().item(), 6)
            if history_ratios.numel() else None
        ),
    }
    report["learned_scalars"] = {
        "history_strength": round(
            torch.sigmoid(original_module.raw_strength).item(), 6
        ),
        "self_decay": round(
            torch.nn.functional.softplus(original_module.raw_decay_self).item(),
            6
        ),
        "other_decay": round(
            torch.nn.functional.softplus(original_module.raw_decay_other).item(),
            6
        ),
    }
    report["warning"] = (
        "Turning off a module post-training changes the model's inference "
        "distribution, but is not a substitute for a retrained baseline. "
        "Original CSS upstream context encoders remain bidirectional. "
        "This analysis uses validation labels for post-hoc grouping only."
    )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/meld_history_probe.json"))
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        ap.error(f"missing checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not all(k in ckpt for k in
               ("config", "model_state_dict", "valid_dialogue_indices")):
        ap.error("expected a validation-selected research checkpoint")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    results = run(ckpt, device)
    for group in ("all", "changed", "changed_after_other", "stable", "first"):
        a = results["with_module"][group]
        b = results["same_checkpoint_without_module_at_inference"][group]
        print(
            f"{group:20s} n={a['n']:4d} "
            f"with F1={a['weighted_f1']:.2f} "
            f"disabled F1={b['weighted_f1']:.2f} "
            f"delta={a['weighted_f1']-b['weighted_f1']:+.2f} pp"
        )
    print(
        "Changed predicted labels:",
        results["predictions_changed_when_module_removed"],
        "/", results["total_validation_utterances"],
    )
    print(
        "Mean ||added history|| / ||SPF feature|| (eligible):",
        results["residual_delta_over_input_norm"]["with_eligible_history_mean"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"Saved validation-only diagnostic: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V6.3 READ-ONLY logit/margin diagnostic on PREVIOUSLY INSPECTED TRAIN.

No training; no VALID (old 115, V5.2 104, V6.1 87) or TEST
inference. Reuse exactly the fixed V6 sampled 64 TRAIN dialogues
(seed73117, size64) so this analysis does NOT consume another
previously uninspected holdout candidate.

Original CSS SPF = high-order polynomial + linear residual.
Original classifier is linear: full class logits can be decomposed
exactly into projected polynomial and projected residual + bias.
We measure logit and TRUE-class margin effects, even when top1
never changes. Not a novel method or a causal training ablation.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.spf_diagnostics import decompose_spfusion
from research.spf_logit_diagnostics import (
    decompose_linear_logits, per_turn_logit_statistics, summarize_turns
)
from scripts.compare_validation import group_labels, read_checkpoint
from scripts.train_research import (
    build_model, v52_disjoint_dialogue_split_indices,
    v61_disjoint_spf_ablation_split_indices
)


CLASS_NAMES = (
    "Neutral", "Surprise", "Fear", "Sadness",
    "Joy", "Disgust", "Anger"
)
GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other"
)


def verify_source_checkpoint(ckpt, ds):
    cfg = ckpt["config"]
    if (cfg.get("dataset") != "MELD"
        or cfg.get("speaker_mode", "off") != "off"
        or cfg.get("precontext_mode", "off") != "off"
        or cfg.get("v52_pgm_mode", "original") != "original"
        or cfg.get("v61_spf_mode", "full") != "full"
        or cfg.get("v61_clean_split") is not True
        or ckpt.get("selection") != "validation_weighted_f1"
        or int(cfg.get("seed", -1)) != 10068
        or int(cfg.get("split_seed", -1)) != 36427):
        raise ValueError(
            "V6.3 requires the original full-SPF V6.1 "
            "validation-selected CSS control checkpoint"
        )
    tr, val, old_val, prev_val, v6_seen = (
        v61_disjoint_spf_ablation_split_indices(
            len(ds), ratio=0.1, split_seed=36427
        )
    )
    if (list(map(int, ckpt["train_dialogue_indices"])) != tr
        or list(map(int, ckpt["valid_dialogue_indices"])) != val):
        raise ValueError("checkpoint split differs from V6.1 strict split")
    v52_train, v52_val, old = v52_disjoint_dialogue_split_indices(
        len(ds), ratio=0.1, split_seed=24117
    )
    if v52_val != prev_val or old != old_val:
        raise RuntimeError("mismatched legacy validation exclusions")
    # Exact V6 full TRAIN sample. No new 87-dialogue VALID sampling.
    selected = sorted(
        np.random.default_rng(73117).choice(
            v52_train, size=64, replace=False
        ).astype(int).tolist()
    )
    if (not set(selected).issubset(tr)
        or not set(selected).issubset(v6_seen)
        or set(selected) & (set(val) | set(old_val) | set(prev_val))):
        raise RuntimeError("V6.3 TRAIN sample is not confined to V6 seen")
    return cfg, selected, (tr, val, old_val, prev_val)


def grouped_summary(records, tag):
    return summarize_turns([
        row for row in records
        if tag == "all" or tag in row["tags"]
    ])


def run(args):
    checkpoint = read_checkpoint(args.checkpoint)
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    cfg, selected, (tr, val, old_val, v52_val) = (
        verify_source_checkpoint(checkpoint, ds)
    )
    loader = DataLoader(
        Subset(ds, selected), batch_size=args.batch_size,
        shuffle=False, collate_fn=ds.collate_fn, num_workers=0
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if model.last_gate.ablation_mode != "full":
        raise RuntimeError("not the original full SPF")
    model.eval()

    cache = {}
    worst_spf_diff, worst_logit_diff = 0., 0.
    rows = []

    def hook(spf, inputs, output):
        p = decompose_spfusion(spf, *inputs)
        err = (output-p["reconstructed"]).abs().max().item()
        cache.clear()
        cache.update(p)
        cache["spf_error"] = float(err)

    handle = model.last_gate.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for idx, batch in enumerate(loader, start=1):
                text, visual, audio, speaker, mask, label, vids = batch
                lengths = mask.sum(dim=1).long().tolist()
                if not lengths or min(lengths) < 1:
                    raise ValueError("empty training dialogue")
                outputs = model(
                    text.to(device), visual.to(device), audio.to(device),
                    mask.to(device),
                    speaker.permute(1, 0, 2).to(device), lengths
                )
                worst_spf_diff = max(worst_spf_diff, cache["spf_error"])
                logits = decompose_linear_logits(
                    model.all_output_layer,
                    cache["polynomial"], cache["residual"]
                )
                from_model = model.all_output_layer(
                    cache["reconstructed"]
                )
                max_abs = (
                    logits["full_logits"] - from_model
                ).abs().max().item()
                worst_logit_diff = max(worst_logit_diff, float(max_abs))
                if worst_spf_diff > 0.002 or worst_logit_diff > 0.002:
                    raise RuntimeError(
                        "source SPF/classifier decomposition inconsistent"
                    )
                if not torch.allclose(
                    F.log_softmax(logits["full_logits"], dim=-1),
                    outputs[3], rtol=1e-4, atol=1e-4
                ):
                    raise RuntimeError(
                        "logit decomposition fails original CSS log probabilities"
                    )
                ids = speaker.argmax(-1).transpose(0, 1).cpu()
                y = label.cpu()
                batch_rows = []
                for j, length in enumerate(lengths):
                    gold = y[j, :length].long().to(device)
                    metrics = per_turn_logit_statistics(
                        logits["full_logits"][j, :length],
                        logits["residual_only_logits"][j, :length],
                        logits["poly_logits"][j, :length],
                        logits["residual_logits_without_bias"][j, :length],
                        gold
                    )
                    tags = group_labels(
                        y[j, :length].tolist(),
                        ids[j, :length].tolist()
                    )
                    if len(metrics) != len(tags):
                        raise RuntimeError("misaligned tags/utterance count")
                    for m, tag in zip(metrics, tags):
                        m["tags"] = list(tag)
                    batch_rows.extend(metrics)
                rows.extend(batch_rows)
                batch_summary = summarize_turns(batch_rows)
                print(
                    f"batch={idx:02d} n={batch_summary['n']:>3d} "
                    f"top1_flips={batch_summary['flip_count']:>2d} "
                    f"centered_logit_ratio="
                    f"{batch_summary['mean_centered_poly_residual_ratio']:.4f} "
                    f"mean_abs_true_margin_delta="
                    f"{batch_summary['mean_abs_poly_true_margin_delta']:.4f} "
                    f"mean_prob_TV="
                    f"{batch_summary['mean_probability_total_variation']:.5f}",
                    flush=True
                )
    finally:
        handle.remove()

    if len(rows) != 668 or len(selected) != 64:
        raise RuntimeError(
            f"expected V6 original 64 inspected TRAIN dialogues / "
            f"668 utterances; saw {len(selected)} / {len(rows)}"
        )
    groups = {g: grouped_summary(rows, g) for g in GROUPS}
    classes = {
        name: summarize_turns([
            r for r in rows if r["label"] == k
        ]) for k, name in enumerate(CLASS_NAMES)
    }
    init_gate = 1.0/(1.0+math.exp(-1.0))
    learned_gates = [
        torch.sigmoid(p).detach().cpu().tolist()
        for p in model.last_gate.modality_gates
    ]
    deviation = max(
        abs(float(g)-init_gate)
        for row in learned_gates for g in row
    )
    report = {
        "protocol": {
            "dataset": "MELD",
            "model": "V6.1 original full SPF, frozen eval",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "selected_train_dialogue_ids": selected,
            "train_dialogues_total": len(tr),
            "previously_inspected_v6_train_dialogues_reused": len(selected),
            "new_validation_dialogues_excluded": len(val),
            "v52_validation_dialogues_excluded": len(v52_val),
            "original_validation_dialogues_excluded": len(old_val),
            "new_training_updates": 0,
            "official_test_evaluated": False,
            "validation_evaluated": False,
            "interpretation_limit": (
                "The classifier logit decomposition is algebraically exact "
                "for a linear final head. This is a descriptive measurement "
                "on ALREADY INSPECTED TRAIN dialogues, not an experiment "
                "comparing the marginal benefit across seeds or datasets. "
                "Top1 invariance does not imply zero probability or margin "
                "impact, and the audit cannot isolate training causality."
            )
        },
        "reconstruction_max_abs_error": {
            "spf": worst_spf_diff,
            "classifier_logits": worst_logit_diff
        },
        "spf_modality_gates": {
            "sigmoid_1_initial_gate": init_gate,
            "learned_per_order_text_audio_visual": learned_gates,
            "max_abs_distance_from_initial_gate_value": deviation,
            "gates_are_globally_learned_not_per_turn": True
        },
        "groups": groups,
        "per_class": classes
    }
    print("\nV6.3 SPF CLASS-LOGIT INFLUENCE | SAME V6 64 TRAIN DIALOGUES")
    print("No training, no old/new VALID or official TEST examined")
    print(f"reconstruction SPF/logit max error: "
          f"{worst_spf_diff:.6g} / {worst_logit_diff:.6g}")
    print(f"max global SPF gate distance from sigmoid(1): {deviation:.6f}")
    print("\nGROUP                     n flips  mean centered logit ratio "
          "mean abs margin delta  mean probability TV  "
          "margin better/worse fraction")
    for name, d in groups.items():
        if not d["n"]:
            continue
        print(
            f"{name:<24}{d['n']:>4} {d['flip_count']:>5}  "
            f"{d['mean_centered_poly_residual_ratio']:>9.4f} "
            f"{d['mean_abs_poly_true_margin_delta']:>22.4f} "
            f"{d['mean_probability_total_variation']:>19.5f}  "
            f"{d['fraction_true_margin_improved_by_poly']:.3f}/"
            f"{d['fraction_true_margin_harmed_by_poly']:.3f}"
        )
    print("\nCLASS      n  flips  logit_ratio  mean_margin_delta  "
          "mean_abs_margin_delta  prob_TV")
    for name, d in classes.items():
        if not d["n"]:
            print(name, "n=0")
            continue
        print(
            f"{name:<10}{d['n']:>3} {d['flip_count']:>5} "
            f"{d['mean_centered_poly_residual_ratio']:>12.4f} "
            f"{d['mean_signed_poly_true_margin_delta']:>18.4f} "
            f"{d['mean_abs_poly_true_margin_delta']:>22.4f} "
            f"{d['mean_probability_total_variation']:>8.5f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("\nSaved:", args.output)
    print("STOP: no parameter tuning on old or new VALID based on this audit.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path(
        "results/v61_spf_paired_split36427_trainseed10068/"
        "checkpoints/v61_full_valid_best.pt"
    ))
    p.add_argument(
        "--output", type=Path,
        default=Path("results/v63_spf_logit_margin_audit.json")
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if not args.checkpoint.is_file():
        p.error(f"missing checkpoint: {args.checkpoint}")
    run(args)


if __name__ == "__main__":
    main()

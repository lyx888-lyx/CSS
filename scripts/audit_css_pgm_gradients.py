#!/usr/bin/env python3
"""Read-only source-aligned CSS PGM gradient audit on TRAIN dialogues only.

NO training, optimizer.step, validation/test inference, threshold selection
or model modifications. Uses a trusted, previously VALID-selected CSS
checkpoint only as a frozen parameter snapshot. The sampled observations
are from the original MELD TRAIN subdivision, never the 115 VALID
dialogues and never official TEST.

This audit checks original PGM's EXACT parameter-selection ORDER and
loss definitions, including the non-detached fused KL target.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from model import MaskedKLDivLoss, MaskedNLLLoss, PGM
from research.pgm_diagnostics import (
    TASK_NAMES, gradient_geometry, safe_mean,
    summarise_batch_geometry, weighted_norm_shares
)
from scripts.compare_validation import read_checkpoint
from scripts.train_research import build_model, dialogue_split_indices


# SOURCE-ALIGNED: same order and module names as train_MELD.py
# train_or_eval_model() -> shared_params. DO NOT insert SPF here.
PARAM_GROUPS = (
    ("cross_modal_transformer", ("shared_transformer",)),
    ("modality_input_and_bias", (
        "textf_input", "acouf_input", "visuf_input",
        "text_to_audio", "text_to_vision",
        "bias_gate_audio", "bias_gate_vision",
    )),
    ("unimodal_context", ("t_t", "a_a", "v_v")),
    ("gates_and_reductions", (
        "t_t_gate", "t_gate", "a_a_gate", "a_gate",
        "v_v_gate", "v_gate",
        "features_reduce_t", "features_reduce_a", "features_reduce_v",
    )),
)
PAIR_KEYS = ("L1_L2", "L1_L3", "L2_L3")
EMOTIONS = (
    "Neutral", "Surprise", "Fear", "Sadness",
    "Joy", "Disgust", "Anger"
)


def select_source_aligned_parameters(model):
    """Return exact original PGM shared-param ordering, grouped for analysis."""
    groups = {}
    all_shared = []
    for group_name, modules in PARAM_GROUPS:
        chosen = []
        for module_name in modules:
            chosen.extend(list(getattr(model, module_name).parameters()))
        groups[group_name] = (len(all_shared), len(all_shared) + len(chosen))
        all_shared.extend(chosen)
    if any(not p.requires_grad for p in all_shared):
        raise ValueError("original CSS PGM selected parameters are unexpectedly frozen")
    if len({id(p) for p in all_shared}) != len(all_shared):
        raise ValueError("duplicate parameters in PGM selection")
    spf = list(model.last_gate.parameters())
    if {id(p) for p in spf} & {id(p) for p in all_shared}:
        raise ValueError("SPF should be excluded from original PGM projection")
    return all_shared, spf, groups


def flatten_aligned(gradient_tuple, param_tuple):
    """Preserve alignment when some loss does not touch a module."""
    if len(gradient_tuple) != len(param_tuple):
        raise ValueError("gradient/parameter length mismatch")
    return torch.cat([
        g.detach().reshape(-1) if g is not None else p.new_zeros(p.numel())
        for g, p in zip(gradient_tuple, param_tuple)
    ])


def checkpoint_and_dataset(checkpoint_path):
    ckpt = read_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    if cfg.get("dataset") != "MELD" or ckpt.get("selection") != "validation_weighted_f1":
        raise ValueError("requires strict VALIDATION-selected original MELD CSS checkpoint")
    if cfg.get("speaker_mode", "off") != "off":
        raise ValueError("V0 speaker modification is not original CSS")
    if cfg.get("precontext_mode", "off") != "off":
        raise ValueError("V1 precontext modification is not original CSS")
    if cfg.get("v3_supcon_weight", 0.0):
        raise ValueError("V3 auxiliary loss checkpoint not allowed")
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    train_idx, val_idx = dialogue_split_indices(len(ds), 0.1, 10068)
    if (list(map(int, ckpt["valid_dialogue_indices"])) != val_idx
            or list(map(int, ckpt["train_dialogue_indices"])) != train_idx
            or int(ckpt.get("split_seed", cfg.get("split_seed", -1))) != 10068):
        raise ValueError("checkpoint does not use the fixed 1037/115 MELD split")
    if len(train_idx) != 1037 or len(val_idx) != 115:
        raise ValueError("unexpected number of dialogue IDs")
    return ckpt, ds, train_idx, val_idx


def accumulate_class_statistics(stats, outputs, y, mask):
    """TRAIN-only per-class losses and logit gradient length, not PGM grads.

    Logit-gradient length is ||softmax(logits)-onehot(y)||_2 per turn
    for the fused NLL (before reduction). It is NOT a weight-gradient
    or a per-class PGM conflict measure.
    """
    with torch.no_grad():
        valid = mask.bool()
        p = outputs[4].detach()
        fused_nll = -outputs[3].detach().gather(
            -1, y.unsqueeze(-1)
        ).squeeze(-1)
        unimodal_nll = sum(
            -out.detach().gather(-1, y.unsqueeze(-1)).squeeze(-1)
            for out in outputs[:3]
        )
        kl_loss = sum(
            F.kl_div(
                h.detach(), outputs[8].detach(), reduction="none"
            ).sum(-1)
            for h in outputs[5:8]
        )
        onehot = F.one_hot(y.long(), num_classes=len(EMOTIONS))
        logit_grad_norm = torch.linalg.vector_norm(
            p - onehot.to(p.dtype), ord=2, dim=-1
        )
        correct = p.argmax(-1) == y
        for k in range(len(EMOTIONS)):
            keep = valid & (y == k)
            n = int(keep.sum().item())
            if not n:
                continue
            row = stats[k]
            row["n"] += n
            row["correct"] += int(correct[keep].sum().item())
            row["sum_L1"] += float(fused_nll[keep].sum().item())
            row["sum_L2"] += float(unimodal_nll[keep].sum().item())
            row["sum_L3"] += float(kl_loss[keep].sum().item())
            row["sum_fused_logit_gradient_norm"] += float(
                logit_grad_norm[keep].sum().item()
            )


def one_batch_diagnostic(model, pgm, batch, shared, spf, group_slices, device,
                         class_accumulator):
    text, visual, audio, speaker, mask, label, _ = batch
    lengths = mask.sum(-1).long().tolist()
    if min(lengths) < 1:
        raise ValueError("sampled an empty dialogue")
    y = label.to(device)
    umask = mask.to(device)
    outputs = model(
        text.to(device), visual.to(device), audio.to(device),
        umask, speaker.permute(1, 0, 2).to(device), lengths
    )
    losses = pgm._compute_task_losses(outputs, y.reshape(-1), umask)
    if len(losses) != 3:
        raise ValueError("original PGM must expose three losses")
    accumulate_class_statistics(
        class_accumulator, outputs, y, umask
    )

    # Exact source-aligned parameter order for original PGM: no SPF.
    # Additionally take SPF gradients to characterize the *excluded*
    # fusion layer. No graph/parameter changes are made.
    all_params = shared + spf
    per_task = []
    missing = {}
    for task_index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss, all_params,
            retain_graph=(task_index < 2), allow_unused=True,
            create_graph=False
        )
        per_task.append(gradients)
        missing[TASK_NAMES[task_index]] = {
            "shared": sum(g is None for g in gradients[:len(shared)]),
            "spf": sum(g is None for g in gradients[len(shared):]),
        }

    common = [
        flatten_aligned(grads[:len(shared)], shared)
        for grads in per_task
    ]
    common_geometry = gradient_geometry(common)

    # Original PGM applies SLSQP projection to UNIT gradients.
    # This is the actual source method, not a proxy optimizer.
    alpha = pgm._project(common).detach().cpu().tolist()
    if (any(not np.isfinite(v) for v in alpha)
            or any(v < -1e-5 or v > 1 + 1e-5 for v in alpha)
            or abs(sum(alpha) - 1.0) > 1e-3):
        raise ValueError("PGM produced invalid simplex weights")

    group_result = {}
    for group, (lo, hi) in group_slices.items():
        vectors = [
            flatten_aligned(grads[lo:hi], shared[lo:hi])
            for grads in per_task
        ]
        group_result[group] = gradient_geometry(vectors)
    spf_vectors = [
        flatten_aligned(grads[len(shared):], spf)
        for grads in per_task
    ]
    # L2 never uses last_gate, hence its SPF angle is undefined (null).
    group_result["SPF_excluded_from_PGM"] = gradient_geometry(spf_vectors)
    return {
        "utterances": int(umask.sum().item()),
        "losses": dict(zip(TASK_NAMES, [
            float(l.detach().item()) for l in losses
        ])),
        "pgm_weights": dict(zip(TASK_NAMES, [float(v) for v in alpha])),
        "weighted_raw_norm_shares_before_cancellation": weighted_norm_shares(
            common_geometry["norms"], alpha
        ),
        "pgm_shared": common_geometry,
        "parameter_groups": group_result,
        "unused_gradient_tensors": missing,
    }


def summarise_class_statistics(stats):
    out = {}
    for k, name in enumerate(EMOTIONS):
        row = stats[k]
        n = row["n"]
        out[name] = {
            "class_id": k,
            "n_train_sample": n,
            "train_accuracy_pct": (
                round(100 * row["correct"] / n, 2) if n else None
            ),
            "mean_per_utterance_losses": {
                f"L{j}": (row[f"sum_L{j}"] / n if n else None)
                for j in (1, 2, 3)
            },
            "fused_NLL_logit_gradient_norm": (
                row["sum_fused_logit_gradient_norm"] / n if n else None
            ),
        }
    return out


def summarize_batches(batches):
    parameter_keys = list(batches[0]["parameter_groups"])
    groups = {
        name: summarise_batch_geometry([
            b["parameter_groups"][name] for b in batches
        ]) for name in parameter_keys
    }
    return {
        "batches": len(batches),
        "utterances": sum(b["utterances"] for b in batches),
        "mean_pgm_weights": {
            name: safe_mean([b["pgm_weights"][name] for b in batches])
            for name in TASK_NAMES
        },
        "mean_loss_values": {
            name: safe_mean([b["losses"][name] for b in batches])
            for name in TASK_NAMES
        },
        "mean_weighted_raw_norm_share": {
            name: safe_mean([
                b["weighted_raw_norm_shares_before_cancellation"][name]
                for b in batches
            ]) for name in TASK_NAMES
        },
        "pgm_shared": summarise_batch_geometry([
            b["pgm_shared"] for b in batches
        ]),
        "parameter_groups": groups,
    }


def run(args):
    ckpt, ds, train_idx, valid_idx = checkpoint_and_dataset(
        args.checkpoint
    )
    cfg = ckpt["config"]
    rng = np.random.default_rng(args.sample_seed)
    max_dialogues = args.batch_size * args.max_batches
    sampled = sorted(rng.choice(
        train_idx, size=min(len(train_idx), max_dialogues),
        replace=False
    ).astype(int).tolist())
    if set(sampled) & set(valid_idx):
        raise RuntimeError("sampled VALID dialogues in TRAIN audit")
    loader = DataLoader(
        Subset(ds, sampled),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=ds.collate_fn, num_workers=0
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    # Train mode uses original stochastic dropout; NO updates occur.
    # Reseed AFTER construction for deterministic dropout stream.
    torch.manual_seed(args.dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.dropout_seed)
    model.train() if args.mode == "train" else model.eval()
    pgm = PGM(MaskedNLLLoss(), MaskedKLDivLoss())
    shared, spf, group_slices = select_source_aligned_parameters(model)
    class_stats = defaultdict(lambda: {
        "n": 0, "correct": 0, "sum_L1": 0.0, "sum_L2": 0.0,
        "sum_L3": 0.0, "sum_fused_logit_gradient_norm": 0.0
    })

    all_batches = []
    for i, batch in enumerate(loader, start=1):
        diag = one_batch_diagnostic(
            model, pgm, batch, shared, spf, group_slices,
            device, class_stats
        )
        all_batches.append(diag)
        c = diag["pgm_shared"]["cosines"]
        weights = diag["pgm_weights"]
        print(
            f"batch={i:02d} n={diag['utterances']:3d} "
            f"cos12={c['L1_L2']:+.3f} "
            f"cos13={c['L1_L3']:+.3f} "
            f"cos23={c['L2_L3']:+.3f} "
            f"alpha=[{weights['L1']:.3f},{weights['L2']:.3f},"
            f"{weights['L3']:.3f}]",
            flush=True
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = summarize_batches(all_batches)
    print("\nPGM INPUT GRADIENT GEOMETRY (selected shared parameters)")
    for pair, row in summary["pgm_shared"]["pairwise"].items():
        print(
            f"{pair}: mean_cos={row['mean_cosine']:+.4f} "
            f"negative={row['conflicting_batches']}/"
            f"{row['measured_batches']}"
        )
    w = summary["mean_pgm_weights"]
    print("Mean PGM alpha:",
          " ".join(f"{name}={w[name]:.4f}" for name in TASK_NAMES))
    effective = summary["mean_weighted_raw_norm_share"]
    print("Mean pre-cancellation alpha*||grad|| shares:",
          " ".join(f"{name}={effective[name]:.4f}" for name in TASK_NAMES))
    print("\nPER-PARAMETER-GROUP COSINE MEANS (undefined -> NA):")
    for name, obj in summary["parameter_groups"].items():
        cosines = obj["pairwise"]
        print(
            f"{name}: " + " ".join(
                f"{k}={v['mean_cosine']:.3f}"
                if v["mean_cosine"] is not None else f"{k}=NA"
                for k, v in cosines.items()
            )
        )
    print("\nSAMPLED TRAIN-CLASS LOSS PROFILE (not VALID; not class-gradient angles):")
    for name, row in summarise_class_statistics(class_stats).items():
        vals = row["mean_per_utterance_losses"]
        fmt = lambda v: f"{v:.3f}" if v is not None else "NA"
        print(
            f"{name:<10} n={row['n_train_sample']:>4d} "
            f"acc={fmt(row['train_accuracy_pct'])} "
            f"L1={fmt(vals['L1'])} L2={fmt(vals['L2'])} "
            f"L3={fmt(vals['L3'])} "
            f"logit_grad_norm={fmt(row['fused_NLL_logit_gradient_norm'])}"
        )

    report = {
        "protocol": {
            "dataset": "MELD",
            "split_seed": 10068,
            "train_dialogues_available": len(train_idx),
            "valid_dialogues_excluded": len(valid_idx),
            "selected_train_dialogue_indices": sampled,
            "sample_seed": args.sample_seed,
            "dropout_seed": args.dropout_seed,
            "model_mode": args.mode,
            "optimizer_steps": 0,
            "validation_predictions_evaluated": False,
            "official_test_evaluated": False,
            "checkpoint": str(args.checkpoint),
            "checkpoint_selected_epoch": int(ckpt["epoch"]),
            "observational_only": (
                "Frozen checkpoint on sampled TRAIN batches. Mean cosine/"
                "negative fraction describes conflict; it cannot prove "
                "PGM causes errors or is suboptimal. TRAIN dropout mode "
                "is not an exact historical training trajectory."
            ),
            "pgm_shared_modules_in_source_order": [
                module for _, modules in PARAM_GROUPS for module in modules
            ],
            "pgm_excluded_modules_of_interest": [
                "last_gate (SPFusion)",
                "t_output_layer", "a_output_layer",
                "v_output_layer", "all_output_layer",
                "speaker_embeddings",
            ],
            "kl_fused_target_detached": False,
            "spf_L2_unconnected_is_expected": True,
        },
        "summary": summary,
        "per_class_train_diagnostics": summarise_class_statistics(class_stats),
        "batches": all_batches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print("\nSaved:", args.output)
    print("No model weights modified; no VALID or TEST dialogues scored.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/v5_css_pgm_gradient_audit.json")
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=73011)
    p.add_argument("--dropout-seed", type=int, default=61027)
    p.add_argument("--mode", choices=("train", "eval"), default="train")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.max_batches < 1:
        p.error("batch-size and max-batches must be positive")
    if not args.checkpoint.is_file():
        p.error(f"checkpoint does not exist: {args.checkpoint}")
    print(
        "V5 ORIGINAL CSS PGM AUDIT | sampled TRAIN only | "
        "zero training updates | VALID/TEST untouched",
        flush=True
    )
    run(args)


if __name__ == "__main__":
    main()

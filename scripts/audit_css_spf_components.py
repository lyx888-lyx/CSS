#!/usr/bin/env python3
"""V6: read-only CSS SPF polynomial-vs-residual diagnostic on disjoint TRAIN.

Uses ONLY dialogues in the V5.2 original-CSS control's 933 TRAIN
dialogues (excludes old 115 VALID and new 104 VALID). Loads the V5.2
original CSS VALID-selected checkpoint as a frozen snapshot. No training,
no gradient, no optimizer, no official TEST or held-out VALID scoring.

Counterfactual scores (dropping a component at inference) are
distribution-shifted probes; NOT fair trained-ablation comparisons.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.spf_diagnostics import (
    component_statistics, decompose_spfusion, summarize_predictions
)
from scripts.compare_validation import group_labels, read_checkpoint
from scripts.train_research import (
    build_model, v52_disjoint_dialogue_split_indices
)

NAMES = ("Neutral", "Surprise", "Fear", "Sadness",
         "Joy", "Disgust", "Anger")
GROUPS = (
    "all", "first", "stable", "changed",
    "stable_after_other", "changed_after_other"
)


def verify_original_css_checkpoint(ckpt, dataset):
    cfg = ckpt["config"]
    if (
        cfg.get("dataset") != "MELD"
        or cfg.get("v52_pgm_mode") != "original"
        or cfg.get("v52_clean_split") is not True
        or cfg.get("speaker_mode", "off") != "off"
        or cfg.get("precontext_mode", "off") != "off"
        or ckpt.get("selection") != "validation_weighted_f1"
    ):
        raise ValueError(
            "V6 expects ORIGINAL CSS V5.2 control, new disjoint validation "
            "split, with no experimental modules or optimizer variant"
        )
    split_seed = int(cfg.get("split_seed", -1))
    if split_seed != 24117:
        raise ValueError("expected preregistered V5.2 split seed 24117")
    tr, val, old_val = v52_disjoint_dialogue_split_indices(
        len(dataset), ratio=0.1, split_seed=split_seed
    )
    if (
        list(map(int, ckpt["train_dialogue_indices"])) != tr
        or list(map(int, ckpt["valid_dialogue_indices"])) != val
    ):
        raise ValueError("checkpoint train/validation IDs do not match")
    if (len(tr), len(val), len(old_val)) != (933, 104, 115):
        raise ValueError("unexpected original MELD dialogue partition")
    if (set(tr) & set(val)) or (set(tr) & set(old_val)):
        raise RuntimeError("TRAIN intersects a holdout cohort")
    return cfg, tr, val, old_val


def one_utterance_row(label, full, no_poly, no_resid,
                      poly, resid):
    p = poly.float()
    r = resid.float()
    pn = float(torch.linalg.vector_norm(p).item())
    rn = float(torch.linalg.vector_norm(r).item())
    cos = float(
        (p * r).sum().item() / max(pn * rn, 1e-8)
    )
    return {
        "true": int(label),
        "full": int(full),
        "no_poly": int(no_poly),
        "no_resid": int(no_resid),
        "poly_norm": pn,
        "residual_norm": rn,
        "poly_residual_cosine": max(-1.0, min(1.0, cos)),
    }


def summarize_rows(rows):
    if not rows:
        return {"n": 0, "observed": False}
    labels = [r["true"] for r in rows]
    full = [r["full"] for r in rows]
    no_poly = [r["no_poly"] for r in rows]
    no_resid = [r["no_resid"] for r in rows]
    paired = summarize_predictions(labels, full, no_poly, no_resid)
    paired.update({
        "observed": True,
        "mean_poly_norm": float(np.mean([
            r["poly_norm"] for r in rows
        ])),
        "mean_residual_norm": float(np.mean([
            r["residual_norm"] for r in rows
        ])),
        "mean_poly_to_residual_norm_ratio": float(np.mean([
            r["poly_norm"] / max(r["residual_norm"], 1e-8)
            for r in rows
        ])),
        "mean_poly_residual_cosine": float(np.mean([
            r["poly_residual_cosine"] for r in rows
        ])),
        "fraction_poly_norm_greater_residual": float(np.mean([
            r["poly_norm"] > r["residual_norm"] for r in rows
        ])),
    })
    return paired


def run(args):
    ckpt = read_checkpoint(args.checkpoint)
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    cfg, tr, fresh_val, old_val = verify_original_css_checkpoint(ckpt, ds)

    rng = np.random.default_rng(args.sample_seed)
    n_dialogues = min(len(tr), args.batch_size * args.max_batches)
    selected = sorted(map(int, rng.choice(
        tr, size=n_dialogues, replace=False
    ).tolist()))
    if (set(selected) & set(fresh_val)) or (set(selected) & set(old_val)):
        raise RuntimeError("sample selection leaks a held-out dialogue")

    loader = DataLoader(
        Subset(ds, selected), batch_size=args.batch_size,
        shuffle=False, collate_fn=ds.collate_fn, num_workers=0
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    holder = {}
    max_error = 0.0
    per_utterance = []
    order_term_samples = defaultdict(list)
    batch_components = []

    def observe_spf(mod, inputs, output):
        # Model's own SPF FORWARD has already completed;
        # only observe and independently reconstruct its components.
        t, a, v = inputs
        result = decompose_spfusion(mod, t, a, v)
        error = float(
            (result["reconstructed"] - output).abs().max().item()
        )
        if not np.isfinite(error) or error > 0.002:
            raise RuntimeError(
                f"SPF reconstruction mismatch: max_abs_error={error}"
            )
        holder.clear()
        holder.update(result)
        holder["error"] = error

    handle = model.last_gate.register_forward_hook(observe_spf)
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader, start=1):
                text, visual, audio, speaker, mask, label, _ = batch
                valid_lens = mask.sum(-1).long().tolist()
                if min(valid_lens) < 1:
                    raise ValueError("empty sampled train dialogue")
                umask = mask.to(device)
                outputs = model(
                    text.to(device), visual.to(device), audio.to(device),
                    umask,
                    speaker.permute(1, 0, 2).to(device),
                    valid_lens,
                )
                pol = holder["polynomial"]
                res = holder["residual"]
                reconstructed = holder["reconstructed"]
                max_error = max(max_error, holder["error"])
                full_logits = model.all_output_layer(reconstructed)
                expected_pred = outputs[4].argmax(-1)
                if not torch.equal(full_logits.argmax(-1), expected_pred):
                    raise RuntimeError(
                        "reconstructed SPF path does not match original CSS "
                        "fusion predictions"
                    )
                # All three heads share one PRE-TRAINED classifier.
                # Only inputs change; there is NO reoptimization.
                full_pred = full_logits.argmax(-1).cpu()
                no_poly_pred = model.all_output_layer(res).argmax(-1).cpu()
                no_resid_pred = model.all_output_layer(pol).argmax(-1).cpu()

                comps = component_statistics(pol, res, umask)
                batch_components.append(comps)
                for order, term in enumerate(holder["order_terms"]):
                    selected_terms = term[umask.bool()]
                    order_term_samples[order].append(float(
                        selected_terms.abs().mean().item()
                    ))

                yy = label.cpu()
                sp = speaker.argmax(-1).transpose(0, 1).cpu()
                poly_cpu = pol.detach().cpu()
                res_cpu = res.detach().cpu()
                for b, length in enumerate(valid_lens):
                    labels = yy[b, :length].tolist()
                    tags = group_labels(
                        labels, sp[b, :length].tolist()
                    )
                    for t in range(length):
                        record = one_utterance_row(
                            labels[t], full_pred[b, t],
                            no_poly_pred[b, t], no_resid_pred[b, t],
                            poly_cpu[b, t], res_cpu[b, t]
                        )
                        record["tags"] = tags[t]
                        per_utterance.append(record)
                print(
                    f"batch={i:02d} dialogues={len(valid_lens)} "
                    f"n={comps['n']:3d} "
                    f"poly_norm={comps['mean_poly_norm']:.3f} "
                    f"residual_norm={comps['mean_residual_norm']:.3f} "
                    f"ratio={comps['mean_poly_to_residual_norm_ratio']:.3f} "
                    f"reconstruction_error={holder['error']:.7f}",
                    flush=True
                )
    finally:
        handle.remove()

    grouped = {
        group: summarize_rows([
            r for r in per_utterance
            if group == "all" or group in r["tags"]
        ])
        for group in GROUPS
    }
    by_class = {
        NAMES[k]: summarize_rows([
            r for r in per_utterance if r["true"] == k
        ])
        for k in range(len(NAMES))
    }
    gates = [
        [float(x) for x in torch.sigmoid(param).detach().cpu().tolist()]
        for param in model.last_gate.modality_gates
    ]

    result = {
        "protocol": {
            "dataset": "MELD", "model": "original CSS V5.2 control",
            "checkpoint": str(args.checkpoint),
            "selected_epoch": int(ckpt["epoch"]),
            "old_valid_dialogues_excluded": len(old_val),
            "new_valid_dialogues_excluded": len(fresh_val),
            "eligible_train_dialogues": len(tr),
            "sampled_train_dialogues": len(selected),
            "sampled_train_dialogue_indices": selected,
            "sample_seed": args.sample_seed,
            "model_mode": "eval", "optimizer_steps": 0,
            "validation_evaluated": False,
            "official_test_evaluated": False,
            "warning": (
                "All numbers are on sampled TRAIN utterances. "
                "Inferred outcomes after removing a component at inference "
                "are distribution-shifted probes, NOT trained ablations. "
                "Large polynomial norm is NOT evidence of usefulness, and "
                "small norm is NOT evidence of irrelevance."
            ),
        },
        "spf": {
            "poly_order": model.last_gate.poly_order,
            "rank": model.last_gate.rank,
            "max_reconstruction_abs_error": max_error,
            "global_modality_gates_per_order_text_audio_visual": gates,
            "gates_vary_per_utterance": False,
            "mean_absolute_order_factor_per_batch": {
                str(order + 1): float(np.mean(values))
                for order, values in order_term_samples.items()
            },
        },
        "all_groups": grouped,
        "per_class": by_class,
        "batch_component_stats": batch_components,
    }

    print("\nV6 SPF POLYNOMIAL vs RESIDUAL | sampled TRAIN only | TEST/VALID untouched")
    print("Source fact: SPF modality_gates are learned GLOBAL per polynomial order,")
    print("             NOT dynamically estimated from each utterance.")
    print("Gates by order [text,audio,visual]:")
    for i, item in enumerate(gates):
        print(f"  order_{i+1}: " + ", ".join(f"{x:.4f}" for x in item))
    print(f"Max forward reconstruction error: {max_error:.7g}")
    print("\nGROUP      N  full_acc  without_poly_acc  without_resid_acc"
          "  remove_poly_fixed/broken")
    for name, row in grouped.items():
        if not row["observed"]:
            print(name, "N=0")
            continue
        print(
            f"{name:<20}{row['n']:>4} "
            f"{row['full_accuracy_pct']:>8.2f} "
            f"{row['without_polynomial_accuracy_pct']:>17.2f} "
            f"{row['without_residual_accuracy_pct']:>18.2f} "
            f"{row['without_polynomial_fixed']:>4}/"
            f"{row['without_polynomial_broken']:<4}"
        )
    print("\nCLASS      N  poly_norm  residual_norm  ratio  full_acc"
          "  no_poly_acc  no_residual_acc")
    for name, row in by_class.items():
        if not row["observed"]:
            print(f"{name:<10} N=0")
            continue
        print(
            f"{name:<10}{row['n']:>3} "
            f"{row['mean_poly_norm']:>10.3f} "
            f"{row['mean_residual_norm']:>13.3f} "
            f"{row['mean_poly_to_residual_norm_ratio']:>6.3f} "
            f"{row['full_accuracy_pct']:>8.2f} "
            f"{row['without_polynomial_accuracy_pct']:>11.2f} "
            f"{row['without_residual_accuracy_pct']:>15.2f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print("Saved:", args.output)
    print("WARNING: component removal is a frozen-model distribution-shifted probe")
    print("         and cannot replace a matched retrained SPF ablation.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/v6_css_spf_audit.json"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=73117)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.max_batches < 1:
        p.error("batch-size and max-batches must be positive")
    if not args.checkpoint.is_file():
        p.error(f"missing checkpoint: {args.checkpoint}")
    run(args)


if __name__ == "__main__":
    main()

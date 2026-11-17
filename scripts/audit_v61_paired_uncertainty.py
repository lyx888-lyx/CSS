#!/usr/bin/env python3
"""V6.2 descriptive paired dialogue bootstrap for two EXISTING checkpoints.

Runs inference on ONLY the 87 already-inspected V6.1 VALID dialogues,
never trains or changes either model; no official TEST. This is a
post hoc *uncertainty summary*, not an opportunity for model selection,
threshold tuning, or publication-level confirmation.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.spf_paired_uncertainty import (
    GROUPS, bootstrap_dialogues, ensure_paired_dialogues,
    per_class_paired, score_group
)
from scripts.compare_validation import group_labels, read_checkpoint
from scripts.train_research import (
    build_model, v61_disjoint_spf_ablation_split_indices
)


def checked_checkpoints(full_file, residual_file):
    first = read_checkpoint(full_file)
    second = read_checkpoint(residual_file)
    c1, c2 = first["config"], second["config"]
    for label, ckpt, cfg, mode in (
        ("full", first, c1, "full"),
        ("residual", second, c2, "residual_only"),
    ):
        if (cfg.get("dataset") != "MELD"
                or cfg.get("v61_clean_split") is not True
                or cfg.get("v61_spf_mode") != mode
                or cfg.get("v52_pgm_mode", "original") != "original"
                or cfg.get("speaker_mode") != "off"
                or cfg.get("precontext_mode", "off") != "off"
                or int(cfg.get("split_seed", -1)) != 36427
                or ckpt.get("selection") != "validation_weighted_f1"):
            raise ValueError(f"invalid V6.1 {label} source checkpoint")
    for name in (
        "train_dialogue_indices", "valid_dialogue_indices", "train_seed"
    ):
        if first[name] != second[name]:
            raise ValueError(f"mismatched training protocol: {name}")
    if first["train_seed"] != 10068:
        raise ValueError("unexpected experiment training seed")
    return first, second


@torch.no_grad()
def infer_rows(checkpoint, dataset, indices, device):
    cfg = checkpoint["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=int(cfg["batch_size"]),
        shuffle=False, collate_fn=dataset.collate_fn, num_workers=0
    )
    records = {}
    for batch in loader:
        text, visual, audio, speaker, mask, label, vids = batch
        lengths = mask.sum(-1).long().tolist()
        outputs = model(
            text.to(device), visual.to(device), audio.to(device),
            mask.to(device), speaker.permute(1, 0, 2).to(device),
            lengths
        )
        pred = outputs[4].argmax(-1).cpu().numpy()
        y = label.cpu().numpy()
        speaker_ids = speaker.argmax(-1).transpose(0, 1).cpu().numpy()
        for j, n in enumerate(lengths):
            vid = str(vids[j])
            if not n or vid in records:
                raise ValueError("duplicate or zero-length validation dialogue")
            labels = y[j, :n].astype(int).tolist()
            sp = speaker_ids[j, :n].astype(int).tolist()
            tags = group_labels(labels, sp)
            records[vid] = {
                "dialogue_id": vid, "labels": labels,
                "speaker_ids": sp,
                "tags": [list(v) for v in tags],
                "prediction": pred[j, :n].astype(int).tolist(),
            }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def evaluate(args):
    full, residual = checked_checkpoints(
        args.full_checkpoint, args.residual_checkpoint
    )
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    train, new_val, old_val, v52_val, v6_seen = (
        v61_disjoint_spf_ablation_split_indices(
            len(ds), 0.1, 36427
        )
    )
    if (full["valid_dialogue_indices"] != new_val
            or full["train_dialogue_indices"] != train):
        raise ValueError("checkpoint does not match the exact clean V6.1 split")
    if ((set(new_val) & set(old_val))
            or (set(new_val) & set(v52_val))
            or (set(new_val) & set(v6_seen))):
        raise RuntimeError("V6.1 validation cohort contaminated by past audits")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    original = infer_rows(full, ds, new_val, device)
    alternative = infer_rows(residual, ds, new_val, device)
    if set(original) != set(alternative):
        raise RuntimeError("not all paired validation dialogues exist")

    records = []
    for vid in original:
        a, b = original[vid], alternative[vid]
        for key in ("labels", "speaker_ids", "tags"):
            if a[key] != b[key]:
                raise RuntimeError(f"unaligned prediction cohort: {vid}/{key}")
        records.append({
            "dialogue_id": vid,
            "labels": a["labels"],
            "tags": a["tags"],
            "full": a["prediction"],
            "residual": b["prediction"],
        })
    ensure_paired_dialogues(records)
    if len(records) != 87:
        raise ValueError("unexpected V6.1 clean validation dialogue count")

    groups = {name: score_group(records, name) for name in GROUPS}
    by_class = per_class_paired(records)
    print(
        "V6.2 REPEATED ANALYSIS OF EXISTING V6.1 VALID ONLY "
        "| NO TRAINING / NO TEST"
    )
    print("Group                    n    full F1    residual F1  "
          "delta pp  fixed broken")
    for name, val in groups.items():
        print(
            f"{name:<24} {val['n']:>3d} "
            f"{val['full_f1']:>9.2f}  {val['residual_f1']:>11.2f} "
            f" {val['delta_residual_minus_full']:>+7.2f} "
            f"{val['residual_fixed']:>5d} {val['residual_broken']:>6d}"
        )
    # Verify predictions reproduce original W-F1 before resampling.
    if (abs(groups["all"]["full_f1"] - 76.14) > .025
            or abs(groups["all"]["residual_f1"] - 76.43) > .025
            or groups["all"]["n"] != 875):
        raise RuntimeError(
            "recomputed V6.1 scores do not match user-reported results"
        )
    print(
        "\nPaired bootstrap over WHOLE DIALOGUES "
        f"(n_dialogues={len(records)}, replicates={args.replicates})"
    )
    boot = bootstrap_dialogues(
        records, reps=args.replicates, seed=args.bootstrap_seed
    )
    for name, row in boot.items():
        bounds = row["ci95_pp"]
        if bounds is None:
            print(name, "no replicates")
            continue
        print(
            f"{name:<24} 95% percentile CI of delta: "
            f"[{bounds[0]:+.2f}, {bounds[1]:+.2f}]pp "
            f"includes_zero={row['interval_includes_zero']}"
        )
    print("\nCLASS SUPPORT AND PAIRED FIX/BREAK")
    for name, row in by_class.items():
        print(
            f"{name:<10} n={row['support']:>3d} "
            f"original_correct={row['full_correct']:>3d} "
            f"residual_correct={row['residual_correct']:>3d} "
            f"fixed={row['residual_fixed']:>3d} "
            f"broken={row['residual_broken']:>3d}"
        )
    result = {
        "protocol": {
            "dataset": "MELD",
            "checkpoint_full": str(args.full_checkpoint),
            "checkpoint_residual": str(args.residual_checkpoint),
            "train_seed": 10068, "split_seed": 36427,
            "n_old_validation_dialogues_excluded": len(old_val),
            "n_v52_validation_dialogues_excluded": len(v52_val),
            "n_v6_seen_dialogues_excluded_from_new_validation": len(v6_seen),
            "validation_dialogues_revisited": len(records),
            "bootstrap_unit": "paired_whole_dialogue",
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.replicates,
            "optimizer_steps": 0,
            "official_test_evaluated": False,
            "warning": (
                "Descriptive paired bootstrap of the ALREADY INSPECTED "
                "V6.1 validation; NOT an independent holdout nor "
                "a source of tuning suggestions. Bootstrap intervals "
                "do not account for training-seed or split variation."
            ),
        },
        "group_scores": groups,
        "cluster_bootstrap": boot,
        "per_class_paired": by_class,
        "paired_dialogues": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("\nSaved:", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/v61_spf_paired_split36427_trainseed10068")
    parser.add_argument(
        "--full-checkpoint", type=Path,
        default=root / "checkpoints/v61_full_valid_best.pt"
    )
    parser.add_argument(
        "--residual-checkpoint", type=Path,
        default=root / "checkpoints/v61_residual_only_valid_best.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "v62_paired_uncertainty.json"
    )
    parser.add_argument("--replicates", type=int, default=1500)
    parser.add_argument("--bootstrap-seed", type=int, default=64217)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.full_checkpoint.is_file() or not args.residual_checkpoint.is_file():
        parser.error("both V6.1 trusted saved checkpoint files are required")
    if args.replicates < 100:
        parser.error("bootstrap replicates must be >=100")
    evaluate(args)


if __name__ == "__main__":
    main()

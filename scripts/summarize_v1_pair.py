#!/usr/bin/env python3
"""Summarize paired V1 validation-selected checkpoints; no TEST evaluation."""
import argparse
import json
from pathlib import Path

import torch


def load_best(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    # PyTorch's restricted weights-only unpickler cannot deserialize all
    # NumPy scalar values saved by the original CSS training metrics (e.g.
    # numpy.core.multiarray.scalar). This file comes from this local run, so
    # full unpickling is acceptable here. NEVER run this on untrusted .pt files.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("selection") != "validation_weighted_f1":
        raise ValueError(f"{path} is not a validation-selected checkpoint")
    cfg = ckpt["config"]
    return {
        "epoch": ckpt["epoch"],
        "validation_weighted_f1": ckpt["validation_f1"],
        "validation_accuracy": ckpt["validation_accuracy"],
        "train_seed": cfg["seed"],
        "split_seed": cfg.get("split_seed", cfg["seed"]),
        "training_dialogues": len(ckpt["train_dialogue_indices"]),
        "validation_dialogues": len(ckpt["valid_dialogue_indices"]),
        "validation_indices": ckpt["valid_dialogue_indices"],
        "precontext_mode": cfg.get("precontext_mode", "off"),
        "speaker_mode": cfg["speaker_mode"],
        "checkpoint": str(path),
    }


def summarize(run_dir):
    a = load_best(run_dir / "checkpoints/css_valid_best.pt")
    b = load_best(run_dir / "checkpoints/v1_precontext_valid_best.pt")
    if (a["train_seed"], a["split_seed"], a["validation_indices"]) != (
        b["train_seed"], b["split_seed"], b["validation_indices"]
    ):
        raise ValueError("CSS and V1 did not use the exact same train seed and validation dialogues")
    if (a["precontext_mode"], b["precontext_mode"]) != ("off", "both"):
        raise ValueError("Unexpected CSS/V1 architecture modes")
    if a["speaker_mode"] != "off" or b["speaker_mode"] != "off":
        raise ValueError("V0 speaker-history branch must be off in this comparison")

    delta = round(b["validation_weighted_f1"] - a["validation_weighted_f1"], 4)
    data = {
        "selection_protocol": "independent dialogue-level validation",
        "test_set_evaluated": False,
        "css": {k: v for k, v in a.items() if k != "validation_indices"},
        "v1": {k: v for k, v in b.items() if k != "validation_indices"},
        "delta_validation_weighted_f1_pp": delta,
        "validation_indices_match": True,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return "\n".join([
        "MELD | STRICT DIALOGUE-VALIDATION PAIR | TEST UNTOUCHED",
        "=======================================================",
        f"Train seed: {a['train_seed']} | split seed: {a['split_seed']}",
        f"Train dialogues: {a['training_dialogues']} | validation dialogues: {a['validation_dialogues']}",
        f"CSS baseline: VALID Weighted F1={a['validation_weighted_f1']:.2f}% "
        f"ACC={a['validation_accuracy']:.2f}% best_epoch={a['epoch']}",
        f"V1 precontext: VALID Weighted F1={b['validation_weighted_f1']:.2f}% "
        f"ACC={b['validation_accuracy']:.2f}% best_epoch={b['epoch']}",
        f"V1-CSS: VALID Weighted F1={delta:+.2f} percentage points",
        f"CSS checkpoint: {a['checkpoint']}",
        f"V1 checkpoint:  {b['checkpoint']}",
        f"Detailed training logs: {run_dir}/css.log ; {run_dir}/v1.log",
        f"JSON summary: {run_dir}/summary.json",
        "These are VALIDATION metrics. The official TEST was not evaluated.",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    print(summarize(args.run_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V1 self/other history-path attribution on an existing VALIDATION checkpoint.

Uses forward hooks to zero only each module's self_diff / other_diff output.
This leaves attention masks, context summaries, learned gates and ALL trained
weights unchanged. No gradients, optimizer updates or official TEST samples.

Models:
  CSS      = separately trained matched baseline
  both     = fully trained V1, unaltered
  self     = same V1 checkpoint, only OTHER projected delta zeroed
  other    = same V1 checkpoint, only SELF projected delta zeroed
  neither  = same V1 checkpoint, both projected deltas zeroed

The last three cases are INFERENCE-ONLY interventions: they are NOT
independently retrained architectural ablations and cannot establish causality.
Read only trusted, locally generated checkpoint files.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataloader import IEMOCAPDataset, MELDDataset
from scripts.compare_validation import (
    compute_report, predict_valid, read_checkpoint
)
from scripts.diagnose_v1_errors import (
    correctness_changes, GROUPS
)
from scripts.train_research import build_model


MODES = ("both", "self", "other", "neither")
DISPLAY = {
    "css": "CSS baseline (independently trained)",
    "both": "V1 full (original checkpoint)",
    "self": "V1 self contribution only",
    "other": "V1 other contribution only",
    "neither": "V1 without either contribution",
}


@contextmanager
def temporarily_ablate_history_paths(model, mode):
    """Remove feature contributions, preserving learned gate calculations.

    Important distinction from changing PreContextSpeakerShift.mode:
    toggling that attribute also changes context inputs to the gate.
    Hooking the two final projection outputs isolates contributions while
    leaving the other branch's learned gate untouched.
    """
    if mode not in MODES:
        raise ValueError(f"expected one of {MODES}, got {mode!r}")
    blocks = model.precontext_shift
    if blocks is None or not isinstance(blocks, torch.nn.ModuleDict):
        raise ValueError("expected a model with trained V1 ModuleDict")
    handles = []

    def zero_projection(_module, _inputs, output):
        return torch.zeros_like(output)

    try:
        if mode != "both":
            for block in blocks.values():
                if mode in ("other", "neither"):
                    handles.append(
                        block.self_diff.register_forward_hook(zero_projection)
                    )
                if mode in ("self", "neither"):
                    handles.append(
                        block.other_diff.register_forward_hook(zero_projection)
                    )
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def validate_checkpoints(css, v1):
    a, b = css["config"], v1["config"]
    for name in (
        "dataset", "seed", "classes", "batch_size",
        "hidden_dim", "rank", "heads", "order", "temp", "lr",
        "valid_ratio", "dropout",
    ):
        if a[name] != b[name]:
            raise ValueError(f"checkpoints differ on {name}")
    if a.get("split_seed", a["seed"]) != b.get("split_seed", b["seed"]):
        raise ValueError("validation split seeds differ")
    if css["valid_dialogue_indices"] != v1["valid_dialogue_indices"]:
        raise ValueError("validation dialogue indices differ")
    if a.get("precontext_mode", "off") != "off":
        raise ValueError("CSS checkpoint must have precontext-mode off")
    if b.get("precontext_mode", "off") != "both":
        raise ValueError("V1 checkpoint must have precontext-mode both")
    if a["speaker_mode"] != "off" or b["speaker_mode"] != "off":
        raise ValueError("V0 speaker-history module must be disabled")
    if css.get("selection") != "validation_weighted_f1" or (
        v1.get("selection") != "validation_weighted_f1"
    ):
        raise ValueError("both checkpoints must be validation-selected")


def build_valid_loader(config, dialogue_indices):
    if config["dataset"] == "MELD":
        dataset = MELDDataset(
            "data/meld_multimodal_features.pkl", train=True
        )
    elif config["dataset"] == "IEMOCAP":
        dataset = IEMOCAPDataset(train=True)
    else:
        raise ValueError("unsupported dataset")
    if not dialogue_indices or any(
        i < 0 or i >= len(dataset) for i in dialogue_indices
    ):
        raise ValueError("validation dialogue indices are missing or invalid")
    return DataLoader(
        Subset(dataset, dialogue_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )


@torch.no_grad()
def run(css_path, v1_path, device):
    css_ckpt = read_checkpoint(css_path)
    v1_ckpt = read_checkpoint(v1_path)
    validate_checkpoints(css_ckpt, v1_ckpt)
    cfg = css_ckpt["config"]
    loader = build_valid_loader(cfg, css_ckpt["valid_dialogue_indices"])

    css_model = build_model(cfg).to(device)
    css_model.load_state_dict(css_ckpt["model_state_dict"], strict=True)
    y, css_pred, tags = predict_valid(css_model, loader, device)
    del css_model, css_ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    v1_model = build_model(v1_ckpt["config"]).to(device)
    v1_model.load_state_dict(v1_ckpt["model_state_dict"], strict=True)
    v1_model.eval()
    del v1_ckpt

    predictions = {"css": css_pred}
    for mode in MODES:
        with temporarily_ablate_history_paths(v1_model, mode):
            yr, pr, tr = predict_valid(v1_model, loader, device)
        if y != yr or tags != tr:
            raise RuntimeError(f"validation samples misaligned for {mode}")
        predictions[mode] = pr
    del v1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = {
        "metadata": {
            "dataset": cfg["dataset"],
            "train_seed": int(cfg["seed"]),
            "split_seed": int(cfg.get("split_seed", cfg["seed"])),
            "validation_dialogues": len(loader.dataset),
            "validation_utterances": len(y),
            "test_evaluated": False,
            "css_checkpoint": str(css_path),
            "v1_checkpoint": str(v1_path),
            "warning": (
                "self/other/neither are inference-only path ablations of "
                "a trained V1, not retrained models. Do not interpret their "
                "scores as valid alternatives to an independently trained "
                "architecture or as evidence of inter-speaker causation."
            ),
        },
        "variants": {},
    }

    for name in ("css",) + MODES:
        pred = predictions[name]
        scores = compute_report(y, pred, tags, cfg["classes"])
        versus_css = correctness_changes(y, css_pred, pred, tags)
        versus_v1 = correctness_changes(y, predictions["both"], pred, tags)
        report["variants"][name] = {
            "description": DISPLAY[name],
            "scores": scores,
            "versus_css": versus_css,
            "versus_full_v1": versus_v1,
        }

    for name in ("css",) + MODES:
        item = report["variants"][name]
        sc = item["scores"]
        print(
            f"{name:<8} all={sc['all']['weighted_f1']:.2f}% "
            f"stable={sc['stable']['weighted_f1']:.2f}% "
            f"changed={sc['changed']['weighted_f1']:.2f}% "
            f"after_other_changed={sc['changed_after_other']['weighted_f1']:.2f}% "
            f"changed_predictions_vs_full_v1="
            f"{item['versus_full_v1']['predicted_label_changed']}",
            flush=True,
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--css", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/v1_history_paths.json"))
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()
    for path in (args.css, args.v1):
        if not path.is_file():
            p.error(f"checkpoint not found: {path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    report = run(args.css, args.v1, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved validation-only history-path diagnostic: {args.output}")


if __name__ == "__main__":
    main()

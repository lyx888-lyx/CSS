#!/usr/bin/env python3
"""Read-only paired V2 validation diagnostics; no model retraining or TEST.

Compare SAME validation utterances across current, EMA, adaptive,
adaptive_other and the *trained* adaptive_other model with only its
learned other_projection output zeroed via temporary forward hook.

The hook is an inference-time feature-path intervention, NOT a
freshly retrained ablation. Other-speaker presence metadata and
classifier parameters remain unchanged. Do not call the difference
an effect of interpersonal emotional causation.

Only load locally generated, trusted PyTorch checkpoints.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.v2_state_model import V2SpeakerStateProbe
from scripts.compare_validation import compute_report, group_labels
from scripts.diagnose_v1_errors import correctness_changes


MODES = ("current", "ema", "adaptive", "adaptive_other")
DIAGNOSTIC_MODES = MODES + ("adaptive_other_zero_external_vector",)
GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")


@contextmanager
def temporarily_remove_external_vector(model):
    """Zero projected other-speaker vector; restore hook even on exception."""
    if model.mode != "adaptive_other":
        raise ValueError("requires a trained adaptive_other model")
    def zero_output(_module, _input, output):
        return torch.zeros_like(output)
    handle = model.other_projection.register_forward_hook(zero_output)
    try:
        yield
    finally:
        handle.remove()


def same_protocol(checkpoints):
    first = checkpoints["current"]
    a = first["config"]
    idx = first["valid_dialogue_indices"]
    if not idx or a.get("max_train_batches", 0) != 0:
        raise ValueError("expected complete training with validation indices")
    for mode in MODES:
        ckpt = checkpoints[mode]
        conf = ckpt["config"]
        if conf["mode"] != mode:
            raise ValueError(f"wrong mode stored in checkpoint: {mode}")
        if conf.get("max_train_batches", 0) != 0:
            raise ValueError("cannot compare smoke-only checkpoints")
        for key in ("seed", "split_seed", "batch_size", "hidden_dim",
                    "dropout", "epochs", "lr", "weight_decay",
                    "valid_ratio", "dataset", "validation_dialogues"):
            if conf[key] != a[key]:
                raise ValueError(f"{mode}: different {key}")
        if ckpt["valid_dialogue_indices"] != idx:
            raise ValueError(f"{mode}: different validation dialogues")
        if ckpt.get("selection") != "validation_weighted_f1":
            raise ValueError(f"{mode}: must use VALIDATION-only selection")
    if a["dataset"] != "MELD":
        raise ValueError("this script targets MELD")
    return idx


def rebuild(ckpt, device):
    conf = ckpt["config"]
    model = V2SpeakerStateProbe(
        mode=conf["mode"], hidden_dim=conf["hidden_dim"],
        dropout=conf["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval()


@torch.no_grad()
def predictions(model, valid_loader, device):
    y, pred, tags = [], [], []
    for batch in valid_loader:
        text, visual, audio, qmask, umask, labels, _ = batch
        inputs = (
            text.to(device), visual.to(device), audio.to(device),
            qmask.to(device), umask.to(device),
        )
        prediction = model(*inputs).argmax(-1).cpu()
        lengths = umask.long().sum(-1).tolist()
        speaker = qmask.argmax(-1).transpose(0, 1)
        for b, n in enumerate(lengths):
            if n < 1:
                raise ValueError("empty validation dialogue")
            yy = labels[b, :n].long().tolist()
            pp = prediction[b, :n].tolist()
            sp = speaker[b, :n].tolist()
            y.extend(yy)
            pred.extend(pp)
            tags.extend(group_labels(yy, sp))
    return y, pred, tags


def compare_predictions(y, tags, predicted):
    if set(predicted) != set(DIAGNOSTIC_MODES):
        raise ValueError("exactly five prediction sets expected")
    result = {"variants": {}, "paired": {}}
    for name in DIAGNOSTIC_MODES:
        pred = predicted[name]
        if len(y) != len(pred):
            raise ValueError(f"{name} predictions misaligned")
        result["variants"][name] = compute_report(y, pred, tags, 7)
    for source, target in (
        ("current", "ema"),
        ("ema", "adaptive"),
        ("ema", "adaptive_other"),
        ("adaptive", "adaptive_other"),
        ("adaptive_other", "adaptive_other_zero_external_vector"),
    ):
        source_pred, target_pred = predicted[source], predicted[target]
        changes = {}
        for group in GROUPS:
            indices = [
                i for i, tt in enumerate(tags)
                if group == "all" or group in tt
            ]
            changes[group] = correctness_changes(
                y, source_pred, target_pred, tags, indices=indices
            )
        result["paired"][source + " -> " + target] = changes
    return result


def run(suite_dir, device):
    checkpoints = {}
    for mode in MODES:
        path = suite_dir / mode / "valid_best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[mode] = torch.load(
            path, map_location="cpu", weights_only=False
        )
    valid_indices = same_protocol(checkpoints)
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    if any(int(i) < 0 or int(i) >= len(ds) for i in valid_indices):
        raise ValueError("validation indices outside MELD TRAIN dataset")
    loader = DataLoader(
        Subset(ds, valid_indices),
        batch_size=checkpoints["current"]["config"]["batch_size"],
        shuffle=False, collate_fn=ds.collate_fn, num_workers=0,
    )
    y_ref, tags_ref, preds = None, None, {}
    model_metadata = {}
    for mode in MODES:
        model = rebuild(checkpoints[mode], device)
        y, p, tags = predictions(model, loader, device)
        if y_ref is None:
            y_ref, tags_ref = y, tags
        elif y != y_ref or tags != tags_ref:
            raise RuntimeError("models have different validation utterance ordering")
        preds[mode] = p
        model_metadata[mode] = {
            "best_epoch": checkpoints[mode]["epoch"],
        }
        if mode == "adaptive_other":
            with torch.no_grad():
                model_metadata[mode]["learned_decay"] = float(
                    torch.nn.functional.softplus(model.raw_decay).item()
                )
                model_metadata[mode]["learned_other_gain"] = float(
                    model.raw_other_gain.sigmoid().item()
                )
                model_metadata[mode]["other_projection_frobenius_norm"] = float(
                    model.other_projection.weight.float().norm().item()
                )
            with temporarily_remove_external_vector(model):
                y0, p0, t0 = predictions(model, loader, device)
            if y0 != y_ref or t0 != tags_ref:
                raise RuntimeError("intervention changed utterance ordering")
            preds["adaptive_other_zero_external_vector"] = p0
        del model, checkpoints[mode]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = compare_predictions(y_ref, tags_ref, preds)
    result["metadata"] = {
        "dataset": "MELD", "validation_dialogues": len(valid_indices),
        "validation_utterances": len(y_ref), "test_evaluated": False,
        "models": model_metadata,
        "meaning": (
            "Paired matched-validation error migration across separately "
            "trained checkpoints; zero external vector is an inference-only "
            "feature-path intervention. Not a retrained ablation, not causal."
        ),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    report = run(args.suite, device)
    print("V2 SAME-VALIDATION PAIRED ERROR DIAGNOSIS | TEST UNTOUCHED")
    print("Model                                Overall   Stable   Changed  Changed-after-other")
    for name in DIAGNOSTIC_MODES:
        s = report["variants"][name]
        print(
            f"{name:<40}"
            f"{s['all']['weighted_f1']:>6.2f}% "
            f"{s['stable']['weighted_f1']:>8.2f}% "
            f"{s['changed']['weighted_f1']:>8.2f}% "
            f"{s['changed_after_other']['weighted_f1']:>17.2f}%"
        )
    print("\nPaired error migration: fixed vs broken against source model")
    print("Pair                                       fixed broken net changed_preds")
    for name, g in report["paired"].items():
        x = g["all"]
        print(
            f"{name:<44}{x['fixed']:>4} "
            f"{x['broken']:>6} {x['net_correct_change']:>+4} "
            f"{x['predicted_label_changed']:>12}"
        )
    ext = report["metadata"]["models"]["adaptive_other"]
    print(f"Learned adaptive-other decay={ext['learned_decay']:.6f}; "
          f"other gain={ext['learned_other_gain']:.6f}; "
          f"projection weight norm={ext['other_projection_frobenius_norm']:.6f}")
    path = args.output or args.suite / "paired_errors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print("Saved:", path)
    print("Caution: the external-vector intervention still retains "
          "other-speaker availability metadata. It is NOT a complete "
          "deletion of other-speaker history.")


if __name__ == "__main__":
    main()

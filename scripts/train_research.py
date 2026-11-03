#!/usr/bin/env python3
"""Strict train/validation-selected CSS experiments.

The original CSS source selects the best checkpoint using TEST weighted F1.
This research runner instead splits the original TRAIN dialogues into train
and validation sets, selects only using validation, and leaves TEST untouched
until an explicit --evaluate-test / --eval-only action.

NOTE: No individual CSS utterance's labels from the held-out set are used
to make inputs, and split granularity is whole dialogues. The underlying
original CSS encoder remains bidirectional within each dialogue.
"""
import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.utils.data import DataLoader, Subset

from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedKLDivLoss, MaskedNLLLoss, Transformer_Based_Model


DEFAULTS = {
    "IEMOCAP": dict(seed=61078, epochs=200, lr=1e-4,
                    order=3, temp=1, audio=1582, classes=6, speakers=2),
    "MELD": dict(seed=10068, epochs=100, lr=5e-6,
                 order=4, temp=4, audio=300, classes=7, speakers=9),
}


def dialogue_split_indices(n_dialogues, ratio, split_seed):
    """Stable dialogue-level indices independent of model initialization seed.

    One Dataset item is one complete conversation. The split is fully
    determined by its ordering, ratio, and split_seed. No test dialogue
    indices are included.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("validation ratio must be strictly between 0 and 1")
    if n_dialogues < 2:
        raise ValueError("need at least two training dialogues")
    shuffled = np.random.default_rng(split_seed).permutation(n_dialogues)
    count = max(1, int(round(n_dialogues * ratio)))
    if count >= n_dialogues:
        raise ValueError("validation split leaves no training dialogues")
    valid_indices = sorted(int(x) for x in shuffled[:count])
    train_indices = sorted(int(x) for x in shuffled[count:])
    assert not set(train_indices).intersection(valid_indices)
    return train_indices, valid_indices


def get_loaders(dataset, batch_size, ratio, split_seed, train_seed=None):
    if dataset == "IEMOCAP":
        trainset = IEMOCAPDataset()
        testset = IEMOCAPDataset(train=False)
    else:
        path = "data/meld_multimodal_features.pkl"
        trainset = MELDDataset(path, train=True)
        testset = MELDDataset(path, train=False)

    train_indices, valid_indices = dialogue_split_indices(
        len(trainset), ratio, split_seed
    )
    # Shuffle training dialogues independently of the validation split.
    # If train_seed is omitted, retain legacy shuffle seed behavior.
    shuffle_seed = split_seed if train_seed is None else train_seed

    train_loader = DataLoader(
        Subset(trainset, train_indices), batch_size=batch_size, shuffle=True,
        collate_fn=trainset.collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(shuffle_seed + 1000)
    )
    valid_loader = DataLoader(
        Subset(trainset, valid_indices), batch_size=batch_size, shuffle=False,
        collate_fn=trainset.collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        collate_fn=testset.collate_fn, num_workers=0
    )
    return train_loader, valid_loader, test_loader, train_indices, valid_indices


def build_model(config):
    model = Transformer_Based_Model(
        config["dataset"], config["temp"], 1024, 342,
        config["audio"], config["heads"], config["rank"], config["order"],
        n_classes=config["classes"], hidden_dim=config["hidden_dim"],
        n_speakers=config["speakers"], dropout=config["dropout"],
        speaker_history_mode=(
            None if config["speaker_mode"] == "off" else config["speaker_mode"]
        ),
        speaker_history_decay=config["time_decay"],
        speaker_history_lag=config["max_lag"],
        precontext_shift_mode=(
            None if config.get("precontext_mode", "off") == "off"
            else config["precontext_mode"]
        ),
    )
    return model


def make_config(args):
    d = DEFAULTS[args.dataset]
    return {
        "dataset": args.dataset,
        "seed": d["seed"] if args.seed is None else args.seed,
        "split_seed": (d["seed"] if args.seed is None else args.seed)
                      if args.split_seed is None else args.split_seed,
        "epochs": d["epochs"] if args.epochs is None else args.epochs,
        "lr": d["lr"] if args.lr is None else args.lr,
        "order": d["order"] if args.order is None else args.order,
        "temp": d["temp"] if args.temp is None else args.temp,
        "audio": d["audio"], "classes": d["classes"], "speakers": d["speakers"],
        "batch_size": args.batch_size, "hidden_dim": args.hidden_dim,
        "heads": args.heads, "rank": args.rank, "dropout": args.dropout,
        "l2": args.l2, "valid_ratio": args.valid_ratio,
        "speaker_mode": args.speaker_mode, "time_decay": not args.no_time_decay,
        "max_lag": args.max_lag,
        "precontext_mode": getattr(args, "precontext_mode", "off"),
        "v11_consistency_lambda": getattr(args, "v11_consistency_lambda", 0.0),
    }


def evaluate_test(model, loader, trainer, ckpt_path):
    # The evaluation routine computes ordinary forward predictions only.
    acc, label, pred, mask, fscore = trainer.evaluate_loaded_model(model, loader)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Final held-out TEST | Accuracy: {acc:.2f}% | Weighted F1: {fscore:.2f}%")
    print(classification_report(
        label, pred, sample_weight=mask, digits=4, zero_division=0
    ))
    print(confusion_matrix(label, pred, sample_weight=mask))
    return {"test_accuracy": acc, "test_weighted_f1": fscore}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DEFAULTS, default="MELD")
    p.add_argument("--seed", type=int, default=None,
                   help="Model initialization, shuffle and stochastic training seed")
    p.add_argument("--split-seed", type=int, default=None,
                   help="Dialogue split seed; defaults to --seed for backward compatibility")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--temp", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--speaker-mode", choices=("off", "self", "other", "both"),
                   default="off")
    p.add_argument("--precontext-mode", choices=("off", "self", "other", "both"),
                   default="off", help="V1 experimental pre-context difference module")
    p.add_argument("--v11-consistency-lambda", type=float, default=0.0,
                   help="Train-only V1.1 consistency on stable-after-other returns; zero disables")
    p.add_argument("--no-time-decay", action="store_true")
    p.add_argument("--max-lag", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--evaluate-test", action="store_true",
                   help="Explicitly evaluate held-out test once after selection")
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    checkpoint = None
    if args.eval_only:
        if args.checkpoint is None:
            p.error("--eval-only requires --checkpoint")
        # Only open checkpoints you created or obtained from a trusted source.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" not in checkpoint or "model_state_dict" not in checkpoint:
            p.error("expected research checkpoint with config and model_state_dict")
        config = checkpoint["config"]
    else:
        config = make_config(args)
        if config["epochs"] < 1:
            p.error("epochs must be positive")
        if config["max_lag"] is not None and config["max_lag"] < 1:
            p.error("max-lag must be positive")

    if (config["speaker_mode"] != "off"
        and config.get("precontext_mode", "off") != "off"):
        p.error("V0 speaker-mode and V1 precontext-mode must not both be enabled")
    coefficient = config.get("v11_consistency_lambda", 0.0)
    if coefficient < 0:
        p.error("--v11-consistency-lambda must be nonnegative")
    if coefficient > 0 and (config["speaker_mode"] != "off"
                            or config.get("precontext_mode", "off") != "other"):
        p.error("V1.1 consistency requires --speaker-mode off --precontext-mode other")

    dataset = config["dataset"]
    trainer = importlib.import_module(
        "train_IEMOCAP" if dataset == "IEMOCAP" else "train_MELD"
    )
    trainer.set_seed(config["seed"])
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    trainer.cuda = use_cuda
    # Original train_or_eval_model uses args.tensorboard as a module-level variable.
    trainer.args = argparse.Namespace(
        tensorboard=False,
        v11_consistency_lambda=config.get("v11_consistency_lambda", 0.0),
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, train_idx, valid_idx = get_loaders(
        dataset, config["batch_size"], config["valid_ratio"],
        config.get("split_seed", config["seed"]), train_seed=config["seed"]
    )
    model = build_model(config).to(device)
    if checkpoint is None:
        # Experimental layers consume RNG state during initialization. Restore
        # identical training RNG streams across CSS-off and CSS-on runs with
        # the same train seed. Initial common CSS weights remain identical.
        trainer.set_seed(config["seed"])
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_test(model, test_loader, trainer, args.checkpoint)
        return

    print(json.dumps({"config": config,
                      "train_dialogues": len(train_idx),
                      "valid_dialogues": len(valid_idx),
                      "train_seed": config["seed"],
                      "split_seed": config.get("split_seed", config["seed"]),
                      "test_selection": "never"},
                     indent=2, ensure_ascii=False))
    if dataset == "IEMOCAP":
        weights = torch.tensor(
            [1/0.086747, 1/0.144406, 1/0.227883,
             1/0.160585, 1/0.127711, 1/0.252668],
            device=device, dtype=torch.float32
        )
        loss_fn = MaskedNLLLoss(weights)
    else:
        loss_fn = MaskedNLLLoss()
    kl_fn = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["l2"])

    run_name = args.run_name or (
        f'{dataset.lower()}_{config["speaker_mode"]}_'
        f'{config.get("precontext_mode", "off")}_seed{config["seed"]}'
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir / f"{run_name}_valid_best.pt"
    best_score = float("-inf")
    best_acc = float("-inf")
    best_epoch = None

    for epoch in range(config["epochs"]):
        train_loss, train_acc, _, _, _, train_f1 = trainer.train_or_eval_model(
            model, loss_fn, kl_fn, train_loader, epoch,
            optimizer=optimizer, train=True
        )
        valid_acc, _, _, _, valid_f1 = trainer.evaluate_loaded_model(
            model, valid_loader
        )
        is_best = valid_f1 > best_score or (
            valid_f1 == best_score and valid_acc > best_acc
        )
        if is_best:
            best_score, best_acc, best_epoch = valid_f1, valid_acc, epoch + 1
            torch.save({
                "epoch": best_epoch,
                "validation_f1": best_score,
                "validation_accuracy": best_acc,
                "model_state_dict": model.state_dict(),
                "config": config,
                "train_dialogue_indices": train_idx,
                "valid_dialogue_indices": valid_idx,
                "selection": "validation_weighted_f1",
                "split_seed": config.get("split_seed", config["seed"]),
                "train_seed": config["seed"],
            }, path)
        print(f"epoch={epoch+1} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.2f} train_f1={train_f1:.2f} "
              f"valid_acc={valid_acc:.2f} valid_f1={valid_f1:.2f} "
              f"best_valid_f1={best_score:.2f} best_epoch={best_epoch}",
              flush=True)

    print(f"Selected epoch={best_epoch} by VALID weighted F1={best_score:.2f}%.")
    print(f"Checkpoint: {path}")
    print("Held-out TEST not examined during training.")
    if args.evaluate_test:
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        evaluate_test(model, test_loader, trainer, path)


if __name__ == "__main__":
    main()

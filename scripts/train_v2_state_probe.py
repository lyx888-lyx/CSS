#!/usr/bin/env python3
"""Low-cost, fair MELD V2 speaker-state experiment (not a CSS reproduction).

Each run:
- same current-only text/audio/visual feature encoder and readout topology
- same 1037/115 TRAIN/VALID whole-dialogue split; no official TEST access
- same training seed, batches, epochs, loss, optimizer for all modes
- states reset at each dialogue and advance in chronological turn order
- validation labels used ONLY after prediction, for model selection/analysis
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.v2_state_model import MODES, V2SpeakerStateProbe
from scripts.compare_validation import group_labels, compute_report
from scripts.train_research import dialogue_split_indices


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(ds, split_seed, train_seed, batch_size, valid_ratio):
    train_idx, valid_idx = dialogue_split_indices(
        len(ds), valid_ratio, split_seed
    )
    train = DataLoader(
        Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
        collate_fn=ds.collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(train_seed + 1000),
    )
    valid = DataLoader(
        Subset(ds, valid_idx), batch_size=batch_size, shuffle=False,
        collate_fn=ds.collate_fn, num_workers=0,
    )
    return train, valid, train_idx, valid_idx


def to_device(batch, device):
    text, visual, audio, qmask, umask, labels, vids = batch
    return (
        text.to(device), visual.to(device), audio.to(device),
        qmask.to(device), umask.to(device), labels.to(device), vids
    )


def train_one_epoch(model, loader, optimizer, device, max_batches=0):
    model.train()
    total_loss, total_n, correct = 0.0, 0, 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        text, visual, audio, qmask, umask, labels, _ = to_device(batch, device)
        logits = model(text, visual, audio, qmask, umask)
        valid = umask.bool()
        loss = nn.functional.cross_entropy(logits[valid], labels[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = int(valid.sum().item())
        total_n += n
        total_loss += float(loss.detach().item()) * n
        correct += int((logits.detach().argmax(-1)[valid] == labels[valid]).sum().item())
    if total_n == 0:
        raise ValueError("training loader had no valid utterances")
    return total_loss / total_n, 100.0 * correct / total_n, total_n


@torch.no_grad()
def evaluate(model, loader, device, n_classes=7):
    model.eval()
    all_y, all_p, all_tags = [], [], []
    for batch in loader:
        text, visual, audio, qmask, umask, labels, _ = to_device(batch, device)
        logits = model(text, visual, audio, qmask, umask)
        preds = logits.argmax(-1).cpu().numpy()
        ys = labels.cpu().numpy()
        speakers = qmask.argmax(-1).transpose(0, 1).cpu().numpy()
        lengths = umask.sum(-1).long().tolist()
        for b, n in enumerate(lengths):
            if n < 1:
                raise ValueError("empty validation dialogue")
            y = ys[b, :n].tolist()
            p = preds[b, :n].tolist()
            sp = speakers[b, :n].tolist()
            all_y.extend(y)
            all_p.extend(p)
            # Gold labels are used ONLY AFTER model prediction.
            all_tags.extend(group_labels(y, sp))
    report = compute_report(all_y, all_p, all_tags, n_classes)
    return report, len(all_y)


def run(args):
    set_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.max_train_batches < 0:
        raise ValueError("--max-train-batches must be nonnegative")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    ds = MELDDataset(str(args.data), train=True)
    train, valid, train_idx, valid_idx = create_dataloaders(
        ds, args.split_seed, args.seed, args.batch_size, args.valid_ratio
    )
    model = V2SpeakerStateProbe(
        mode=args.mode, hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "valid_best.pt"
    metadata = {
        "model": "V2SpeakerStateProbe",
        "dataset": "MELD",
        "mode": args.mode,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "valid_ratio": args.valid_ratio,
        "max_train_batches": args.max_train_batches,
        "train_dialogues": len(train_idx),
        "validation_dialogues": len(valid_idx),
        "test_evaluated": False,
        "history": "strictly earlier speaker utterances, no gold-label inputs",
        "backbone": "independent current-utterance multimodal encoder, NOT CSS",
    }
    print(json.dumps(metadata, indent=2), flush=True)
    if args.max_train_batches:
        print("SMOKE-ONLY: training used a subset of batches. "
              "Do NOT compare smoke scores as a full model.", flush=True)
    best_f1, best_acc, best_epoch = -1.0, -1.0, 0
    trace = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, n_seen = train_one_epoch(
            model, train, optimizer, device,
            max_batches=args.max_train_batches,
        )
        report, n_valid = evaluate(model, valid, device)
        f1 = report["all"]["weighted_f1"]
        acc = report["all"]["accuracy"]
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1, best_acc, best_epoch = f1, acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": metadata,
                "train_dialogue_indices": train_idx,
                "valid_dialogue_indices": valid_idx,
                "epoch": best_epoch,
                "validation_weighted_f1": best_f1,
                "selection": "validation_weighted_f1",
            }, checkpoint)
            best_report = report
        trace.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 2), "train_utterances": n_seen,
            "validation_weighted_f1": f1,
            "validation_accuracy": acc,
        })
        print(
            f"mode={args.mode} epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"valid_f1={f1:.2f}% valid_acc={acc:.2f}% "
            f"best_valid_f1={best_f1:.2f}% best_epoch={best_epoch}",
            flush=True,
        )

    result = {
        "config": metadata, "selected_epoch": best_epoch,
        "validation_utterances": n_valid,
        "validation": best_report, "epoch_trace": trace,
        "checkpoint": str(checkpoint),
        "warning": (
            "One-seed pilot of an independent, lightweight model. "
            "Never compare directly with independently trained CSS "
            "as architecture-matched evidence. Model selection used "
            "validation only; official TEST untouched."
        ),
    }
    path = args.out_dir / "summary.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(
        f"SELECTED mode={args.mode} epoch={best_epoch} "
        f"VALID weighted F1={best_f1:.2f}% "
        f"changed={best_report['changed']['weighted_f1']:.2f}% "
        f"stable={best_report['stable']['weighted_f1']:.2f}%",
        flush=True,
    )
    print(f"Saved summary: {path}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Official MELD TEST untouched.", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--data", type=Path,
                   default=Path("data/meld_multimodal_features.pkl"))
    p.add_argument("--seed", type=int, default=10068)
    p.add_argument("--split-seed", type=int, default=10068)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="Optional small train-batch limit for SMOKE ONLY")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cpu", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()

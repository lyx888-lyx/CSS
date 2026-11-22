#!/usr/bin/env python3
"""TRAIN-only fixed-epoch CSS leave-one-session-out confusion audit.

Each fold trains original CSS on three IEMOCAP TRAIN sessions and evaluates
ONLY the remaining TRAIN session, exactly once after epoch 200. No TEST
Dataset(train=False), TEST model selection, new architecture, or gold history
labels in inference. This is exploratory given previously inspected Ses04.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedKLDivLoss, MaskedNLLLoss
from research.session_confusion_stability import (
    SESSIONS, split_by_session, validate_train_only, fold_indices,
    summarize_fold, aggregate_folds,
)
from scripts.run_v12_author_protocol import create_model, CLASS_WEIGHTS

DEFAULT_OUTPUT = Path("results/css_session_confusion_stability")
FINAL_EPOCH = 200
BATCH_SIZE = 32


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_partition():
    # Original pickle may physically contain other split arrays, but only
    # the official TRAIN key list is traversed; no TEST IDs are instantiated,
    # evaluated, or used to select the model.
    ds = IEMOCAPDataset(train=True)
    metadata = validate_train_only(ds.keys, ds.videoLabels)
    groups = split_by_session(ds.keys)
    return ds, groups, metadata


def setup_train(seed, no_cuda):
    trainer.set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    )
    trainer.cuda = device.type == "cuda"
    trainer.args = argparse.Namespace(tensorboard=False)
    model = create_model("original").to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    weights = torch.tensor([1.0 / x for x in CLASS_WEIGHTS],
                           dtype=torch.float32, device=device)
    return device, model, optimizer, MaskedNLLLoss(weights), MaskedKLDivLoss()


def train_fold(args, ds, groups, meta):
    held_out = args.fold if args.fold is not None else "Ses01"
    train_indices, valid_indices = fold_indices(groups, held_out)
    root = args.output_dir / f"seed{args.seed}"
    report_path = root / f"{held_out}.json"
    checkpoint_path = root / f"{held_out}_epoch200.pt"
    if not args.smoke and (report_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite frozen fold: {report_path} or {checkpoint_path}"
        )

    device, model, optimizer, loss_fn, kl_fn = setup_train(args.seed, args.no_cuda)
    train_loader = DataLoader(
        Subset(ds, train_indices), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=ds.collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    epochs = 1 if args.smoke else FINAL_EPOCH
    for epoch in range(epochs):
        loss, acc, _, _, _, f1 = trainer.train_or_eval_model(
            model, loss_fn, kl_fn, train_loader, epoch,
            optimizer=optimizer, train=True
        )
        print(f"{held_out} train epoch={epoch+1:03d}/{epochs} "
              f"loss={loss:.4f} ACC={acc:.2f}% W-F1={f1:.2f}%",
              flush=True)

    if args.smoke:
        print("SMOKE PASS: original CSS one epoch on TRAIN-only; "
              "no held-out or official TEST scoring.")
        return

    # Held-out VALID session is accessed just once, after fixed epoch 200.
    valid_loader = DataLoader(
        Subset(ds, valid_indices), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=ds.collate_fn, num_workers=0
    )
    _, labels, predictions, masks, _ = trainer.evaluate_loaded_model(
        model, valid_loader
    )
    mask = np.asarray(masks, dtype=bool)
    y = np.asarray(labels, dtype=np.int64)[mask]
    pred = np.asarray(predictions, dtype=np.int64)[mask]
    ids = [str(ds.keys[i]) for i in valid_indices]
    lengths = [len(ds.videoLabels[k]) for k in ids]
    expected_y = np.concatenate([
        np.asarray(ds.videoLabels[k], dtype=np.int64) for k in ids
    ])
    if len(y) != sum(lengths) or not np.array_equal(y, expected_y):
        raise RuntimeError("incorrect held-out predictions/labels alignment")
    summary = summarize_fold(y, pred)
    if summary["utterances"] != meta["utterances_by_session"][held_out]:
        raise RuntimeError("held-out utterance coverage mismatch")

    training_sessions = [s for s in SESSIONS if s != held_out]
    train_turns = sum(meta["utterances_by_session"][s]
                      for s in training_sessions)
    report = {
        "protocol": {
            "dataset": "IEMOCAP-6", "official_train_only": True,
            "official_test_scored": False, "training_sessions": training_sessions,
            "training_utterances": train_turns,
            "held_out_session": held_out,
            "seed": args.seed, "epochs": FINAL_EPOCH,
            "optimizer": "Adam lr=1e-4 wd=1e-5",
            "batch_size": BATCH_SIZE,
            "original_css_pgm_spf": True,
            "checkpoint_selection": "fixed_epoch_200_no_heldout_selection",
        },
        "held_out_session": held_out,
        "checkpoint_selection": "fixed_epoch_200_no_heldout_selection",
        "held_out_dialogue_ids": ids,
        "held_out_dialogue_lengths": lengths,
        "labels": y.astype(int).tolist(),
        "predictions": pred.astype(int).tolist(),
        "summary": summary,
    }
    root.mkdir(parents=True, exist_ok=True)
    # Checkpoint provenance; save only after full training & held-out checks.
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": args.seed,
        "epoch": FINAL_EPOCH,
        "held_out_session": held_out,
        "training_sessions": training_sessions,
        "selection": "fixed_epoch_200_no_heldout_selection",
    }, checkpoint_path)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"HELD-OUT {held_out}: N={len(y)} "
          f"ACC={summary['accuracy']:.2f}% "
          f"W-F1={summary['weighted_f1']:.2f}% "
          f"Macro-F1={summary['macro_f1']:.2f}%", flush=True)
    for name, item in summary["pairs"].items():
        print(f"  {name}: {item['a_to_b']} / {item['b_to_a']} "
              f"directional confusions; "
              f"pair-support rate={item['pair_errors_per_pair_support_pct']}%")
    print("Saved:", report_path)


def aggregate(args):
    root = args.output_dir / f"seed{args.seed}"
    summary_path = root / "cross_session_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite previous aggregate: {summary_path}")
    folds = {}
    for held_out in SESSIONS:
        path = root / f"{held_out}.json"
        checkpoint = root / f"{held_out}_epoch200.pt"
        report = json.loads(path.read_text(encoding="utf-8"))
        p = report["protocol"]
        if (p.get("official_train_only") is not True or
                p.get("official_test_scored") is not False or
                p.get("original_css_pgm_spf") is not True or
                p.get("seed") != args.seed or
                p.get("epochs") != FINAL_EPOCH or
                p.get("held_out_session") != held_out or
                p.get("checkpoint_selection") !=
                "fixed_epoch_200_no_heldout_selection" or
                p.get("training_sessions") !=
                [s for s in SESSIONS if s != held_out]):
            raise ValueError(f"untrusted/incompatible fold metadata: {held_out}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != report.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint changed or missing: {held_out}")
        folds[held_out] = report
    result = aggregate_folds(folds)
    result["protocol"]["seed"] = args.seed
    result["checkpoint_sha256_by_fold"] = {
        s: folds[s]["checkpoint_sha256"] for s in SESSIONS
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    summary = result["pooled_out_of_fold"]
    print(f"OOF TRAIN-only N={summary['utterances']} "
          f"ACC={summary['accuracy']:.2f}% "
          f"W-F1={summary['weighted_f1']:.2f}% "
          f"Macro-F1={summary['macro_f1']:.2f}%")
    for pair, details in result["pair_stability"].items():
        print(f"{pair}: bidirectional_nonzero_folds="
              f"{details['bidirectional_nonzero_folds']}/4")
        for session in SESSIONS:
            print(f"  {session} directional rates (%): "
                  f"{details['per_session_directed_rates_pct'][session]}")
    print("WARNING: previously inspected Session4 selected the candidate pairs; "
          "this is exploratory, not independent confirmation.")
    print("Saved:", summary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--fold", choices=SESSIONS)
    modes.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int, default=61078)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args)
        return
    ds, groups, metadata = load_train_partition()
    if args.preflight:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print("PREFLIGHT PASS: 120 TRAIN dialogues, 5810 utterances; no TEST inference")
        return
    train_fold(args, ds, groups, metadata)


if __name__ == "__main__":
    main()

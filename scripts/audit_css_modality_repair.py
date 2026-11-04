#!/usr/bin/env python3
"""V3 read-only CSS modality disagreement audit — NO training, NO TEST.

2026 research motivation: SURE (ICASSP, uncertainty-aware experts),
DnR (WACV, complementary modality evidence), EMART (ACL, role-aware
fusion), and HiRoC (2026, disagreement correction prior art).

This is a SCREENING experiment, not a new paper method. It uses
the stored CSS VALIDATION-selected checkpoint and its MELD dialogue
indices. It never trains weights, tunes thresholds, or reads MELD TEST.
All true validation labels are used ONLY AFTER all predictions are
collected, to score the frozen, predetermined rule and find an
UPPER BOUND on modal error recoverability.

NOTE: CSS modality heads are NOT independent classifiers; they were
jointly trained with PGM. Their errors may be strongly correlated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from scripts.compare_validation import (
    compute_report, group_labels, read_checkpoint,
)
from scripts.diagnose_v1_errors import correctness_changes
from scripts.train_research import build_model


GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")
HEADS = ("text", "audio", "visual")


def fixed_consensus_repair(fused, heads):
    """Predetermined, label-free conservative two-modal vote.

    The one prespecified pilot rule is:
    - Fused top probability strictly less than 0.60.
    - At least TWO of text/audio/visual unimodal argmax predictions
      agree on some class DIFFERENT from fused argmax.
    - Mean probability assigned to the consensus label by the agreeing
      heads exceeds fused top-class probability by >0.05.

    Then replace ONLY the argmax class prediction, not probabilities.
    The thresholds are fixed before reading validation results, NOT
    selected via validation tuning. No future, speaker, label or
    emotion-change annotations are used.
    """
    fused = np.asarray(fused, dtype=np.float64)
    heads = np.asarray(heads, dtype=np.float64)
    if fused.ndim != 2 or heads.shape != (len(fused), 3, fused.shape[-1]):
        raise ValueError("expected fused=[N,C], heads=[N,3,C]")
    if not np.isfinite(fused).all() or not np.isfinite(heads).all():
        raise ValueError("nonfinite probability vectors")
    if (np.abs(fused.sum(-1) - 1) > 1e-3).any():
        raise ValueError("fused probabilities must sum to 1")
    if (np.abs(heads.sum(-1) - 1) > 1e-3).any():
        raise ValueError("unimodal probabilities must sum to 1")
    base = fused.argmax(-1).astype(np.int64)
    output = base.copy()
    triggered = np.zeros(len(fused), dtype=bool)
    conditions = {
        "low_fused_confidence": np.zeros(len(fused), dtype=bool),
        "two_modal_agreement_with_fused_disagreement": np.zeros(
            len(fused), dtype=bool
        ),
        "margin_satisfied": np.zeros(len(fused), dtype=bool),
    }
    head_predictions = heads.argmax(-1)
    for i in range(len(fused)):
        base_prob = float(fused[i, base[i]])
        low_conf = base_prob < 0.60
        conditions["low_fused_confidence"][i] = low_conf
        classes, counts = np.unique(head_predictions[i], return_counts=True)
        candidates = classes[(counts >= 2) & (classes != base[i])]
        if len(candidates) == 0:
            continue
        conditions["two_modal_agreement_with_fused_disagreement"][i] = True
        # 3 modalities, so at most one class has a 2+ majority.
        consensus = int(candidates[0])
        who = head_predictions[i] == consensus
        consensus_confidence = float(heads[i, who, consensus].mean())
        margin_ok = consensus_confidence > base_prob + 0.05
        conditions["margin_satisfied"][i] = margin_ok
        if low_conf and margin_ok:
            output[i] = consensus
            triggered[i] = True
    return output, triggered, conditions


def correctable_opportunities(labels, fused_pred, head_pred, tags):
    """Diagnostic ORACLE opportunities, never a deployable predictor."""
    labels = np.asarray(labels, dtype=np.int64)
    fused_pred = np.asarray(fused_pred, dtype=np.int64)
    head_pred = np.asarray(head_pred, dtype=np.int64)
    if (labels.shape != fused_pred.shape or
            head_pred.shape != (len(labels), 3) or
            len(tags) != len(labels)):
        raise ValueError("mismatched predictions, labels, tags")
    correct = head_pred == labels[:, None]
    wrong = fused_pred != labels
    vote_gold = correct.sum(-1)
    head_majority = (
        (head_pred[:, 0] == head_pred[:, 1])
        | (head_pred[:, 0] == head_pred[:, 2])
        | (head_pred[:, 1] == head_pred[:, 2])
    )
    majority_label = np.where(
        head_pred[:, 0] == head_pred[:, 1],
        head_pred[:, 0],
        np.where(
            head_pred[:, 0] == head_pred[:, 2],
            head_pred[:, 0], head_pred[:, 1],
        ),
    )
    out = {}
    for group in GROUPS:
        chosen = np.array([
            group == "all" or group in tag for tag in tags
        ], dtype=bool)
        n = int(chosen.sum())
        n_wrong = int((chosen & wrong).sum())
        one_right = int((chosen & wrong & (vote_gold >= 1)).sum())
        two_right = int((chosen & wrong & (vote_gold >= 2)).sum())
        unsafe = int(
            (chosen & ~wrong & head_majority &
             (majority_label != fused_pred)).sum()
        )
        out[group] = {
            "n": n,
            "fused_wrong": n_wrong,
            "wrong_with_any_correct_modal_head": one_right,
            "wrong_with_two_or_more_correct_modal_heads": two_right,
            "fused_correct_but_two_heads_agree_on_wrong_label": unsafe,
            "share_of_errors_with_any_correct_head": (
                round(100 * one_right / n_wrong, 2)
                if n_wrong else None
            ),
        }
    return out


def paired_score_and_moves(labels, tags, before, after, n_classes=7):
    scores = compute_report(labels, after, tags, n_classes)
    moves = {}
    for group in GROUPS:
        indices = [
            i for i, tag in enumerate(tags)
            if group == "all" or group in tag
        ]
        moves[group] = correctness_changes(
            labels, before, after, tags, indices=indices
        )
    return scores, moves


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, device):
    ckpt = read_checkpoint(checkpoint_path)
    conf = ckpt["config"]
    if (conf.get("dataset") != "MELD"
            or conf.get("speaker_mode", "off") != "off"
            or conf.get("precontext_mode", "off") != "off"):
        raise ValueError(
            "Expected original CSS MELD research checkpoint without V0/V1"
        )
    if ckpt.get("selection") != "validation_weighted_f1":
        raise ValueError("Checkpoint must be VALIDATION-selected; TEST selection forbidden")
    indices = ckpt["valid_dialogue_indices"]
    if len(indices) != 115:
        raise ValueError("Expected original fixed 115-dialogue MELD validation split")
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    if any(int(i) < 0 or int(i) >= len(ds) for i in indices):
        raise ValueError("Invalid validation dialogue indices")
    model = build_model(conf).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=conf["batch_size"], shuffle=False,
        collate_fn=ds.collate_fn, num_workers=0,
    )
    all_y, all_tags, fused_probs, heads_probs = [], [], [], []
    for batch in loader:
        text, visual, audio, qmask, umask, label, _ = batch
        lengths = umask.sum(-1).long().tolist()
        preds = model(
            text.to(device), visual.to(device), audio.to(device),
            umask.to(device), qmask.permute(1, 0, 2).to(device), lengths,
        )
        fused = preds[4].detach().cpu().numpy()       # B,L,C
        separate = torch.stack([
            preds[0].exp(), preds[1].exp(), preds[2].exp()
        ], dim=2).detach().cpu().numpy()            # B,L,3,C
        speaker = qmask.argmax(-1).transpose(0, 1).cpu().numpy()
        y = label.cpu().numpy()
        for b, n in enumerate(lengths):
            if n < 1:
                raise ValueError("empty validation dialogue")
            labels = y[b, :n].astype(int).tolist()
            all_y.extend(labels)
            all_tags.extend(group_labels(labels, speaker[b, :n]))
            fused_probs.append(fused[b, :n])
            heads_probs.append(separate[b, :n])
    all_fused = np.concatenate(fused_probs)
    all_heads = np.concatenate(heads_probs)
    if len(all_y) != 1044:
        raise RuntimeError(
            f"Unexpected validation utterance count: {len(all_y)}"
        )
    return all_y, all_tags, all_fused, all_heads, {
        "checkpoint": str(checkpoint_path),
        "selected_epoch": ckpt["epoch"],
        "train_seed": conf["seed"],
        "split_seed": conf.get("split_seed", conf["seed"]),
        "validation_dialogues": len(indices),
        "validation_utterances": len(all_y),
    }


def run(checkpoint_path, device):
    y, tags, fused, heads, meta = evaluate_checkpoint(checkpoint_path, device)
    pred_fused = fused.argmax(-1)
    pred_head = heads.argmax(-1)
    repaired, triggered, conditions = fixed_consensus_repair(fused, heads)
    baseline = compute_report(y, pred_fused, tags, 7)
    head_scores = {
        name: compute_report(y, pred_head[:, j], tags, 7)
        for j, name in enumerate(HEADS)
    }
    repair_scores, repair_moves = paired_score_and_moves(
        y, tags, pred_fused, repaired,
    )
    diagnostics = correctable_opportunities(y, pred_fused, pred_head, tags)
    pred_changed = (pred_head != pred_fused[:, None]).sum(-1)
    metadata = {
        **meta, "test_evaluated": False,
        "method": "frozen CSS heads; no training or label input at prediction",
        "policy": (
            "Pre-fixed two-of-three modal argmax consensus different "
            "from fusion; fusion confidence <0.60 and majority "
            "mean confidence > fusion confidence +0.05"
        ),
        "warning": (
            "Label-based recoverability is an ORACLE diagnostic only, "
            "never a valid model prediction; all scoring uses only VALIDATION. "
            "Unimodal heads are jointly trained and not independent. "
            "One validation split, fixed gate; no generalization claim. "
            "Other published 2026 systems already use disagreement "
            "calibration, so this is NOT an original method."
        ),
    }
    return {
        "metadata": metadata,
        "scores": {
            "original_fused_css": baseline,
            "text_head": head_scores["text"],
            "audio_head": head_scores["audio"],
            "visual_head": head_scores["visual"],
            "frozen_consensus_repair": repair_scores,
        },
        "paired_repair_changes": repair_moves,
        "recovery_opportunities": diagnostics,
        "disagreement_diagnostics": {
            "at_least_one_modal_head_differs_from_fused": int(
                (pred_changed >= 1).sum()
            ),
            "two_or_three_modal_heads_differ_from_fused": int(
                (pred_changed >= 2).sum()
            ),
            "all_three_modal_heads_differ_from_fused": int(
                (pred_changed == 3).sum()
            ),
            "fused_mean_top_probability": float(fused.max(-1).mean()),
            "fused_median_top_probability": float(np.median(fused.max(-1))),
            "rule_triggered_n": int(triggered.sum()),
            "rule_triggered_share_percent": round(
                100 * triggered.mean(), 2
            ),
            "conditions_count": {
                key: int(value.sum())
                for key, value in conditions.items()
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/v3_css_modality_repair_audit.json"),
    )
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error("Checkpoint not found: " + str(args.checkpoint))
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    report = run(args.checkpoint, device)
    base = report["scores"]["original_fused_css"]
    repaired = report["scores"]["frozen_consensus_repair"]
    print("V3 CSS MODALITY REPAIRABILITY AUDIT | VALIDATION ONLY, TEST UNTOUCHED")
    print("Checkpoint:", args.checkpoint)
    print("Model        Overall W-F1  Stable W-F1  Changed W-F1  After-other-change")
    for name in ("original_fused_css", "text_head", "audio_head",
                 "visual_head", "frozen_consensus_repair"):
        d = report["scores"][name]
        print(
            f"{name:<25}"
            f"{d['all']['weighted_f1']:>7.2f}% "
            f"{d['stable']['weighted_f1']:>11.2f}% "
            f"{d['changed']['weighted_f1']:>12.2f}% "
            f"{d['changed_after_other']['weighted_f1']:>18.2f}%"
        )
    if abs(base["all"]["weighted_f1"] - 73.59) > 0.11:
        print(
            "WARNING: re-evaluated CSS validation F1 differs from expected "
            "73.59%; verify split/checkpoint before interpreting results."
        )
    print("\nWrong CSS examples with potentially correct unimodal evidence:")
    print("Group                  wrong  >=1 head right  >=2 heads right  unsafe wrong majority")
    for group in ("all", "stable", "changed", "changed_after_other"):
        d = report["recovery_opportunities"][group]
        print(
            f"{group:<21}"
            f"{d['fused_wrong']:>5}"
            f"{d['wrong_with_any_correct_modal_head']:>15}"
            f"{d['wrong_with_two_or_more_correct_modal_heads']:>18}"
            f"{d['fused_correct_but_two_heads_agree_on_wrong_label']:>22}"
        )
    print("\nFrozen consensus repair action:")
    print("Triggered:", report["disagreement_diagnostics"]["rule_triggered_n"])
    for group in ("all", "stable", "changed", "changed_after_other"):
        m = report["paired_repair_changes"][group]
        print(
            f"{group}: n={m['n']} "
            f"fixed={m['fixed']} broken={m['broken']} "
            f"net={m['net_correct_change']:+d} "
            f"prediction_changed={m['predicted_label_changed']}"
        )
    print(
        f"Weighted F1 delta (frozen rule vs CSS): "
        f"{repaired['all']['weighted_f1']-base['all']['weighted_f1']:+.2f} pp"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Saved:", args.output)
    print(
        "Decision gate: if few CSS errors are recoverable from modal "
        "heads, STOP this path. If opportunities exist, design a "
        "TRAIN-only confidence calibration and independent seeds; "
        "never fit a correction rule on VALID labels."
    )


if __name__ == "__main__":
    main()

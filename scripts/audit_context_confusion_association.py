#!/usr/bin/env python3
"""Analyze emotion confusions vs prior-own history using frozen TRAIN-only OOF.

This is READ-ONLY analysis of existing 4-fold predictions. It does NOT
train a model, open checkpoints, access IEMOCAP official TEST, or select epochs.
Gold previous emotions are used to label analysis groups AFTER prediction.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import IEMOCAPDataset
from research.session_confusion_stability import (
    SESSIONS, split_by_session, validate_train_only, aggregate_folds
)
from research.context_confusion_association import dialogue_observations, analyze

DEFAULT_INPUT = Path("results/css_session_confusion_stability/seed61078")


def load_and_verify(input_dir):
    ds = IEMOCAPDataset(train=True)
    validation = validate_train_only(ds.keys, ds.videoLabels)
    groups = split_by_session(ds.keys)
    fold_reports = {}
    for session in SESSIONS:
        path = input_dir / f"{session}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report.get("protocol", {})
        if (report.get("held_out_session") != session or
                report.get("checkpoint_selection") !=
                "fixed_epoch_200_no_heldout_selection" or
                protocol.get("official_train_only") is not True or
                protocol.get("official_test_scored") is not False or
                protocol.get("epochs") != 200 or
                protocol.get("held_out_session") != session or
                protocol.get("original_css_pgm_spf") is not True):
            raise ValueError(f"non-TRAIN / unsupported fold: {session}")
        expected = [str(ds.keys[i]) for i in groups[session]]
        if report["held_out_dialogue_ids"] != expected:
            raise ValueError(f"fold dialogue order or identity mismatch: {session}")
        expected_lengths = [len(ds.videoLabels[k]) for k in expected]
        if report["held_out_dialogue_lengths"] != expected_lengths:
            raise ValueError(f"fold utterance lengths mismatch: {session}")
        fold_reports[session] = report

    pooled = aggregate_folds(fold_reports)
    summary_path = input_dir / "cross_session_summary.json"
    prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (prev_summary["pooled_out_of_fold"] != pooled["pooled_out_of_fold"] or
            prev_summary["per_session"] != pooled["per_session"] or
            prev_summary["pair_stability"] != pooled["pair_stability"]):
        raise ValueError("frozen four-fold summary disagrees with fold predictions")
    saved_hashes = prev_summary.get("checkpoint_sha256_by_fold", {})
    if saved_hashes != {s: fold_reports[s].get("checkpoint_sha256")
                        for s in SESSIONS}:
        raise ValueError("checkpoint fingerprint provenance mismatch")

    rows = []
    for session in SESSIONS:
        fold = fold_reports[session]
        y = fold["labels"]
        p = fold["predictions"]
        offset = 0
        for did, length in zip(fold["held_out_dialogue_ids"],
                               fold["held_out_dialogue_lengths"]):
            labels = y[offset:offset + length]
            predictions = p[offset:offset + length]
            expected_labels = ds.videoLabels[did]
            if len(labels) != length or not np.array_equal(
                    np.asarray(labels, dtype=np.int64),
                    np.asarray(expected_labels, dtype=np.int64)):
                raise ValueError(f"TRAIN labels not aligned to OOF: {did}")
            speakers = ds.videoSpeakers[did]
            rows.extend(dialogue_observations(
                did, speakers, labels, predictions))
            offset += length
        if offset != len(y) or len(y) != len(p):
            raise ValueError(f"unconsumed fold predictions: {session}")
    if len(rows) != 5810:
        raise ValueError("OOF coverage is not exactly 5810 utterances")
    return validation, rows, saved_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "context_confusion_association.json"
    if output.exists():
        parser.error(f"Refusing to overwrite previous analysis: {output}")
    metadata, rows, hashes = load_and_verify(args.input_dir)
    results = analyze(rows)
    results["protocol"] = {
        "dataset": "IEMOCAP-6",
        "data": "official TRAIN only; Ses01-Ses04",
        "source": "frozen 4-fold fixed-epoch-200 OOF prediction JSON files",
        "test_loader_instantiated": False,
        "model_training_or_inference": False,
        "checkpoint_loaded": False,
        "gold_prior_emotion_used_in_model": False,
        "gold_prior_emotion_used_posthoc": True,
        "session4_previously_inspected": True,
        "interpretation": "descriptive association only, NOT causal",
        "train_data_checks": metadata,
        "checkpoint_sha256_by_fold": hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TRAIN-only CONTEXT x CONFUSION ASSOCIATION | 4-fold OOF")
    print("Coverage:", results["coverage"])
    for direction, item in results["directions"].items():
        print(f"\n{direction}: total={item['current_true_class_support']} "
              f"to_wrong_target={item['mistaken_as_target']} "
              f"prior_own={item['has_prior_own']}")
        for feature, res in item["features"].items():
            cells = res["pooled"]
            present, absent = cells["present"], cells["absent"]
            print(f"  {feature}: "
                  f"TRUE {present['mistakes']}/{present['n']} "
                  f"({present['error_pct']}%) vs "
                  f"FALSE {absent['mistakes']}/{absent['n']} "
                  f"({absent['error_pct']}%); "
                  f"gap={cells['risk_difference_pp']} pp; "
                  f"positive sessions "
                  f"{res['session_positive_gap_count']}/"
                  f"{res['session_comparable_count']}")
    print("\nWARNING: gold history labels used ONLY for analysis; "
          "context associations do not prove speaker influence. "
          "CSS is bidirectional; prior Session4 exposure remains.")
    print("Saved:", output)


if __name__ == "__main__":
    main()

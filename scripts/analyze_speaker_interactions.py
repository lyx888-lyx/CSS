#!/usr/bin/env python3
"""Finer speaker-return diagnostics on TRAIN dialogues of trusted CSS pickles.

WARNING: Python pickle can execute code. Only read official/trusted feature files.
All emotion labels in this analysis are used for research diagnostics only;
NEVER pass ground-truth test or previous-speaker emotion labels as model features.
The statistics are observational, not estimates of causal contagion.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

FEATURES = {"MELD": "meld_multimodal_features.pkl",
            "IEMOCAP": "iemocap_multimodal_features.pkl"}


def speakers_as_ids(raw):
    values = np.asarray(raw)
    if values.ndim == 2:
        return values.argmax(axis=1).astype(int).tolist()
    if values.ndim != 1:
        raise ValueError(f"unsupported speaker data shape {values.shape}")
    mapping = {}
    result = []
    for speaker in values.tolist():
        speaker = str(speaker)
        if speaker not in mapping:
            mapping[speaker] = len(mapping)
        result.append(mapping[speaker])
    return result


def lag_bucket(lag):
    if lag <= 4:
        return str(lag)
    if lag <= 8:
        return "5-8"
    return "9+"


def summary(counter):
    total = counter["total"]
    changed = counter["changed"]
    return {
        "samples": total,
        "emotion_changed": changed,
        "change_rate": round(changed / total, 4) if total else None,
    }


def run_analysis(speakers, labels, train_ids):
    overall = Counter()
    by_gap = {}
    by_other_emotion = {}
    other_agrees_previous = Counter()
    other_disagrees_previous = Counter()
    by_gap_when_disagree = {}

    for did in train_ids:
        sp = speakers_as_ids(speakers[did])
        ys = [int(y) for y in labels[did]]
        if len(sp) != len(ys):
            raise ValueError(f"{did} has mismatched speaker and label counts")
        last_own = {}
        for i, person in enumerate(sp):
            if person not in last_own:
                last_own[person] = i
                continue
            previous = last_own[person]
            gap = i - previous
            changed = int(ys[i] != ys[previous])
            overall["all_return_events"] += 1
            if gap == 1:
                overall["adjacent_same_speaker"] += 1
                overall["adjacent_changed"] += changed
            else:
                # By definition, no same-speaker turn exists inside the gap.
                # Therefore every intervening turn belongs to another speaker.
                overall["with_intervening_other"] += 1
                overall["with_intervening_other_changed"] += changed
                bucket = lag_bucket(gap)
                group = by_gap.setdefault(bucket, Counter())
                group["total"] += 1
                group["changed"] += changed

                # Immediately preceding utterance is by another speaker.
                other_label = ys[i - 1]
                other_bucket = by_other_emotion.setdefault(other_label, Counter())
                other_bucket["total"] += 1
                other_bucket["changed"] += changed
                if ys[previous] == other_label:
                    dst = other_agrees_previous
                else:
                    dst = other_disagrees_previous
                    dst["current_matches_last_other"] += int(ys[i] == other_label)
                    dst["current_keeps_previous_self"] += int(ys[i] == ys[previous])
                    gap_dst = by_gap_when_disagree.setdefault(bucket, Counter())
                    gap_dst["total"] += 1
                    gap_dst["changed"] += changed
                    gap_dst["current_matches_last_other"] += int(ys[i] == other_label)
                dst["total"] += 1
                dst["changed"] += changed
            last_own[person] = i

    return {
        "events": dict(overall),
        "gap_note": (
            "Gap=1 means adjacent turns by the same speaker; "
            "gap>=2 always contains another speaker. Therefore a naive "
            "with-other versus without-other contrast cannot disentangle "
            "speaker interaction from gap length."
        ),
        "speaker_return_change_rate_by_gap": {
            k: summary(v) for k, v in sorted(by_gap.items(),
                key=lambda x: {"2":2,"3":3,"4":4,"5-8":5,"9+":9}[x[0]])
        },
        "speaker_return_change_rate_by_last_other_label": {
            str(k): summary(v) for k, v in sorted(by_other_emotion.items())
        },
        "last_other_same_emotion_as_previous_self": summary(other_agrees_previous),
        "last_other_different_emotion_from_previous_self": {
            **summary(other_disagrees_previous),
            "current_matches_last_other": other_disagrees_previous["current_matches_last_other"],
            "current_keeps_previous_self": other_disagrees_previous["current_keeps_previous_self"],
            "match_other_rate": (
                round(other_disagrees_previous["current_matches_last_other"] /
                      other_disagrees_previous["total"], 4)
                if other_disagrees_previous["total"] else None
            ),
        },
        "last_other_different_by_gap": {
            k: {
                **summary(v),
                "match_other_rate": (
                    round(v["current_matches_last_other"] / v["total"], 4)
                    if v["total"] else None
                ),
            } for k, v in sorted(by_gap_when_disagree.items(),
                key=lambda x: {"2":2,"3":3,"4":4,"5-8":5,"9+":9}[x[0]])
        },
        "limitations": (
            "Labels are annotations, not an emotion-transmission measurement. "
            "Rates are affected by class imbalance, topic, turn-taking and "
            "temporal distance. Last-other agreement is not causal evidence."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=FEATURES, default="MELD")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    file = args.data_dir / FEATURES[args.dataset]
    if not file.is_file():
        ap.error(f"data file not found: {file}")
    with file.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    if len(raw) < 12:
        raise ValueError("unexpected CSS feature format")
    result = {
        "dataset": args.dataset,
        "split": "train",
        "analysis": run_analysis(raw[1], raw[2], raw[10]),
    }
    content = json.dumps(result, ensure_ascii=False, indent=2)
    print(content)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

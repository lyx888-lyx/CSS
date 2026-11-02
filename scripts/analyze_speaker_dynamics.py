#!/usr/bin/env python3
"""Explore within-speaker and cross-speaker histories in CSS datasets.

Run ONLY on trusted official preprocessed pickle files: unpickling arbitrary
files can execute malicious code.

This script uses labels for descriptive training-split analysis, NOT to add
ground-truth emotional states to the model inputs. Observed associations
must never be interpreted as evidence of causal emotional influence.
"""
import argparse
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

FILES = {
    "IEMOCAP": "iemocap_multimodal_features.pkl",
    "MELD": "meld_multimodal_features.pkl",
}


def speaker_ids_for_dialogue(entries):
    values = np.asarray(entries)
    if values.ndim == 2:
        return values.argmax(axis=-1).astype(int).tolist()
    if values.ndim != 1:
        raise ValueError(f"unexpected speaker encoding shape: {values.shape}")
    # IEMOCAP exposes M/F strings, but numeric IDs work as well.
    mapping = {}
    out = []
    for value in values.tolist():
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return out


def analyze(speakers, labels, dialogue_ids):
    counts = Counter()
    dialogue_lengths = []
    speaker_counts = []
    changed = {"with_other": [0, 0], "without_other": [0, 0]}

    for did in dialogue_ids:
        sid = speaker_ids_for_dialogue(speakers[did])
        emo = [int(v) for v in labels[did]]
        if len(sid) != len(emo):
            raise ValueError(f"speaker/label length mismatch for dialogue {did}")
        dialogue_lengths.append(len(sid))
        speaker_counts.append(len(set(sid)))
        last_by_speaker = {}
        for i, current_speaker in enumerate(sid):
            counts["utterances"] += 1
            has_own = current_speaker in last_by_speaker
            has_other = any(s != current_speaker for s in sid[:i])
            if has_own:
                counts["has_own_history"] += 1
            if has_other:
                counts["has_other_history"] += 1
            if has_own and has_other:
                counts["has_both_histories"] += 1
            if has_own:
                previous_index = last_by_speaker[current_speaker]
                other_intervened = any(
                    sid[k] != current_speaker
                    for k in range(previous_index + 1, i)
                )
                key = "with_other" if other_intervened else "without_other"
                changed[key][1] += 1
                if emo[i] != emo[previous_index]:
                    changed[key][0] += 1
            last_by_speaker[current_speaker] = i

    def rate(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else None

    return {
        "dialogues": len(dialogue_ids),
        "utterances": counts["utterances"],
        "dialogue_length_mean": round(statistics.mean(dialogue_lengths), 2) if dialogue_lengths else None,
        "dialogue_length_median": statistics.median(dialogue_lengths) if dialogue_lengths else None,
        "speakers_per_dialogue_mean": round(statistics.mean(speaker_counts), 2) if speaker_counts else None,
        "speakers_per_dialogue_max": max(speaker_counts, default=None),
        "utterances_with_prior_same_speaker": counts["has_own_history"],
        "utterances_with_prior_other_speaker": counts["has_other_history"],
        "utterances_with_both": counts["has_both_histories"],
        "own_emotion_change_after_intervening_other": {
            "changed": changed["with_other"][0],
            "eligible": changed["with_other"][1],
            "rate": rate(changed["with_other"]),
        },
        "own_emotion_change_without_intervening_other": {
            "changed": changed["without_other"][0],
            "eligible": changed["without_other"][1],
            "rate": rate(changed["without_other"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=FILES, default="MELD")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--split", choices=("train", "test"), default="train",
        help="Use train for exploratory research; test only after the hypothesis and model are frozen.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output file.")
    args = parser.parse_args()
    path = args.data_dir / FILES[args.dataset]
    if not path.is_file():
        parser.error(f"missing {path}; run scripts/check_data.py first")
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if len(obj) < 12:
        raise ValueError("unexpected CSS dataset pickle format")
    speakers, labels = obj[1], obj[2]
    train_ids, test_ids = obj[10], obj[11]
    selected = train_ids if args.split == "train" else test_ids
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "interpretation": (
            "Descriptive correlations only. The data do not identify who caused "
            "whose emotional state, and emotion labels are not model input."
        ),
        "statistics": analyze(speakers, labels, selected),
    }
    data = json.dumps(result, ensure_ascii=False, indent=2)
    print(data)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

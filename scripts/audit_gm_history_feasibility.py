#!/usr/bin/env python3
"""Zero-training MELD feasibility audit for classic speaker-specific GM(1,1).

The classic model normally needs at least four observed scalar history
points. A forecast at current dialogue turn therefore needs >=4 PRIOR
utterances from that SAME speaker in the SAME conversation.

This script uses only speaker identities/lengths from the MELD TRAIN
partition and its fixed, dialogue-level VALIDATION split. It never
loads the official TEST partition or uses true emotion labels.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dataloader import MELDDataset
from scripts.train_research import dialogue_split_indices


def audit_dialogues(dialogues, minimum_history=4):
    """Input: list of arrays of per-turn speaker IDs, one array per dialogue."""
    if minimum_history < 1:
        raise ValueError("minimum_history must be >=1")
    utterances = 0
    elig_count = 0
    repeat_count = 0
    with_other_between_count = 0
    eligible_with_other_between = 0
    histories = Counter()
    track_lengths = Counter()
    num_tracks = 0
    for raw in dialogues:
        speakers = np.asarray(raw)
        if speakers.ndim != 1 or speakers.size == 0:
            raise ValueError("each dialogue must be non-empty 1D speaker IDs")
        num_tracks += len(set(speakers.tolist()))
        for n in Counter(speakers.tolist()).values():
            track_lengths[n if n < 6 else "6+"] += 1
        previous = {}
        last = {}
        for t, speaker in enumerate(speakers.tolist()):
            old = previous.get(speaker, 0)
            histories[old if old < 6 else "6+"] += 1
            if old:
                repeat_count += 1
                other_between = t - last[speaker] > 1
                with_other_between_count += int(other_between)
            else:
                other_between = False
            if old >= minimum_history:
                elig_count += 1
                eligible_with_other_between += int(other_between)
            previous[speaker] = old + 1
            last[speaker] = t
            utterances += 1

    return {
        "dialogues": len(dialogues),
        "utterances": utterances,
        "speaker_dialogue_tracks": num_tracks,
        "mean_utterances_per_speaker_track": (
            round(utterances / num_tracks, 3) if num_tracks else None
        ),
        "gm_minimum_history": minimum_history,
        "eligible_prediction_turns": elig_count,
        "eligible_prediction_share_percent": (
            round(100 * elig_count / utterances, 2) if utterances else 0
        ),
        "insufficient_history_turns": utterances - elig_count,
        "speaker_return_turns": repeat_count,
        "speaker_return_with_other_between": with_other_between_count,
        "eligible_with_other_between": eligible_with_other_between,
        "speaker_track_lengths": {
            str(k): v for k, v in sorted(
                track_lengths.items(), key=lambda pair: (str(pair[0]) == "6+", str(pair[0]))
            )
        },
        "prior_same_speaker_turns_per_target": {
            str(k): v for k, v in sorted(
                histories.items(), key=lambda pair: (str(pair[0]) == "6+", str(pair[0]))
            )
        },
    }


def collect_dialogues(dataset, indices):
    for ix in indices:
        key = dataset.keys[int(ix)]
        one_hot = np.asarray(dataset.videoSpeakers[key])
        if one_hot.ndim != 2:
            raise ValueError(f"expected one-hot speakers [L,S], got {one_hot.shape} for {key}")
        yield one_hot.argmax(axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/meld_multimodal_features.pkl"))
    p.add_argument("--split-seed", type=int, default=10068)
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--minimum-history", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/grey_history_feasibility.json"))
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f"not found: {args.data}")
    dataset = MELDDataset(str(args.data), train=True)
    train_idx, valid_idx = dialogue_split_indices(
        len(dataset), args.validation_ratio, args.split_seed
    )
    reports = {
        "description": (
            "MELD original TRAIN partition only; fixed train/validation dialogue split. "
            "History eligibility is based solely on earlier same-speaker turns. "
            "No true emotion labels or official TEST were used."
        ),
        "split_seed": args.split_seed,
        "minimum_history": args.minimum_history,
        "train": audit_dialogues(
            list(collect_dialogues(dataset, train_idx)),
            minimum_history=args.minimum_history,
        ),
        "validation": audit_dialogues(
            list(collect_dialogues(dataset, valid_idx)),
            minimum_history=args.minimum_history,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for label in ("train", "validation"):
        x = reports[label]
        print(f"== {label.upper()} ==")
        print(
            f"Dialogues: {x['dialogues']}; utterances: {x['utterances']}; "
            f"speaker-dialogue tracks: {x['speaker_dialogue_tracks']}; "
            f"mean turns/track: {x['mean_utterances_per_speaker_track']}"
        )
        print(
            f"Targets with >= {args.minimum_history} PREVIOUS same-speaker turns "
            f"(classical GM forecast eligible): "
            f"{x['eligible_prediction_turns']}/{x['utterances']} "
            f"({x['eligible_prediction_share_percent']:.2f}%)"
        )
        print("Speaker track length counts:", x["speaker_track_lengths"])
        print("Prior same-speaker turn counts:", x["prior_same_speaker_turns_per_target"])
        print(
            f"Return after one or more other-speaker turns: "
            f"{x['speaker_return_with_other_between']} "
            f"(GM-eligible such targets: {x['eligible_with_other_between']})"
        )
    print(f"Saved no-training/no-TEST feasibility report: {args.output}")


if __name__ == "__main__":
    main()

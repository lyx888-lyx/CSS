#!/usr/bin/env python3
"""Validate the official CSS feature pickles without training.

WARNING: pickle.load executes code from untrusted files. Only inspect files
obtained from the paper's official data source or another trusted source.
"""
import argparse
import pickle
from pathlib import Path

DATASETS = {
    "IEMOCAP": {"filename": "iemocap_multimodal_features.pkl", "entries": 12,
                "text": 1024, "audio": 1582, "visual": 342, "classes": 6, "speakers": 2},
    "MELD": {"filename": "meld_multimodal_features.pkl", "entries": 13,
             "text": 1024, "audio": 300, "visual": 342, "classes": 7, "speakers": 9},
}


def feature_shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    if not value:
        return (0,)
    return (len(value), len(value[0]))


def check_one(label, config, directory):
    path = directory / config["filename"]
    print(f"\n[{label}] {path}")
    if not path.is_file():
        print("  MISSING: download the preprocessed features from the official README.")
        return False
    with path.open("rb") as file:
        # Use latin1 to preserve compatibility with older IEMOCAP pickle files.
        obj = pickle.load(file, encoding="latin1")
    if not isinstance(obj, (tuple, list)) or len(obj) != config["entries"]:
        print(f"  ERROR: expected {config['entries']} pickle entries, "
              f"got {len(obj) if hasattr(obj, '__len__') else type(obj)}")
        return False
    ids, speakers, labels, text, _, _, _, audio, visual, _, train, test = obj[:12]
    train_ids, test_ids = list(train), list(test)
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        print("  ERROR: empty split or overlapping train/test dialogue IDs")
        return False
    failures = []
    all_classes = set()
    for vid in train_ids + test_ids:
        if any(vid not in part for part in (speakers, labels, text, audio, visual)):
            failures.append(f"{vid}: missing features/labels")
            continue
        length = len(labels[vid])
        all_classes.update(int(v) for v in labels[vid])
        if not length or len(speakers[vid]) != length:
            failures.append(f"{vid}: label/speaker length mismatch")
        for name, collection in (("text", text), ("audio", audio), ("visual", visual)):
            shape = feature_shape(collection[vid])
            if shape != (length, config[name]):
                failures.append(f"{vid}: {name} shape {shape}, "
                                f"expected ({length}, {config[name]})")
        if label == "MELD":
            shape = feature_shape(speakers[vid])
            if shape != (length, config["speakers"]):
                failures.append(f"{vid}: speaker shape {shape}, "
                                f"expected ({length}, {config['speakers']})")
    if not all_classes.issubset(set(range(config["classes"]))):
        failures.append(f"labels outside [0, {config['classes'] - 1}]: {sorted(all_classes)}")
    print(f"  dialogues: train={len(train_ids)}, test={len(test_ids)}")
    print(f"  modalities: text={config['text']}, audio={config['audio']}, "
          f"visual={config['visual']}; labels={sorted(all_classes)}")
    if failures:
        for msg in failures[:12]:
            print(f"  ERROR: {msg}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} additional errors")
        return False
    print("  OK: split, labels, speaker features and modality dimensions")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["IEMOCAP", "MELD", "all"], default="all")
    args = parser.parse_args()
    datasets = DATASETS.items() if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]
    ok = all([check_one(name, cfg, args.data_dir) for name, cfg in datasets])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

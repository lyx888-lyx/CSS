"""Validation-only pairing and summary tests, no MELD data or GPU required."""
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.summarize_v1_pair import summarize


def save_dummy(path, mode, validation_indices=None, seed=10068):
    valid = [1, 3] if validation_indices is None else validation_indices
    torch.save({
        "selection": "validation_weighted_f1",
        "epoch": 52,
        "validation_f1": 73.50 if mode == "off" else 74.00,
        "validation_accuracy": 74.50,
        "model_state_dict": {"toy": torch.ones(1)},
        "config": {
            "seed": seed,
            "split_seed": 10068,
            "precontext_mode": mode,
            "speaker_mode": "off",
        },
        "train_dialogue_indices": [0, 2, 4],
        "valid_dialogue_indices": valid,
    }, path)


class TestV1PairSummary(unittest.TestCase):
    def test_outputs_summary_with_validation_not_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "checkpoints").mkdir()
            save_dummy(d / "checkpoints/css_valid_best.pt", "off")
            save_dummy(d / "checkpoints/v1_precontext_valid_best.pt", "both")
            result = summarize(d)
            self.assertIn("TEST UNTOUCHED", result)
            self.assertIn("+0.50 percentage points", result)
            self.assertTrue((d / "summary.json").exists())

    def test_rejects_mismatched_validation_dialogues(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "checkpoints").mkdir()
            save_dummy(d / "checkpoints/css_valid_best.pt", "off")
            save_dummy(
                d / "checkpoints/v1_precontext_valid_best.pt",
                "both", validation_indices=[1, 2]
            )
            with self.assertRaisesRegex(ValueError, "exact same"):
                summarize(d)


if __name__ == "__main__":
    unittest.main()

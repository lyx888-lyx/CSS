"""Synthetic regression tests for V2 cross-run validation protocol."""
import unittest

from research.v2_state_model import MODES
from scripts.summarize_v2_state_suite import validate_and_summarize


def sample_report(mode, score=50.0):
    config = dict(
        mode=mode, dataset="MELD", seed=10068, split_seed=10068,
        valid_ratio=0.1, batch_size=32, epochs=12, hidden_dim=64,
        dropout=0.15, lr=0.001, weight_decay=0.0001,
        max_train_batches=0, validation_dialogues=115,
        train_dialogues=1037, test_evaluated=False,
    )
    groups = ("all", "first", "stable", "changed",
              "stable_after_other", "changed_after_other")
    return {
        "config": config,
        "valid_dialogue_indices": [0, 4, 16],
        "validation_utterances": 1044,
        "selected_epoch": 6,
        "checkpoint": "test/" + mode,
        "validation": {
            group: {
                "n": 10, "weighted_f1": score,
                "macro_f1": score, "accuracy": score,
            } for group in groups
        },
    }


class TestV2SuiteComparison(unittest.TestCase):
    def setUp(self):
        self.reports = {mode: sample_report(mode) for mode in MODES}

    def test_common_split_and_fair_protocol(self):
        self.reports["adaptive_other"]["validation"]["all"]["weighted_f1"] = 54.5
        s = validate_and_summarize(self.reports)
        self.assertEqual(s["protocol"]["test_evaluated"], False)
        self.assertEqual(len(s["rows"]), 5)
        self.assertEqual(s["rows"][-1]["delta_vs_current_pp"]["all"], 4.5)

    def test_rejects_split_mismatch(self):
        self.reports["ema"]["valid_dialogue_indices"] = [3, 4, 16]
        with self.assertRaisesRegex(ValueError, "validation dialogue indices"):
            validate_and_summarize(self.reports)

    def test_rejects_seed_mismatch(self):
        self.reports["last"]["config"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "differs from current on seed"):
            validate_and_summarize(self.reports)

    def test_rejects_test_access(self):
        self.reports["adaptive"]["config"]["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "official TEST"):
            validate_and_summarize(self.reports)

    def test_rejects_missing_arm(self):
        del self.reports["ema"]
        with self.assertRaisesRegex(ValueError, "exactly five modes"):
            validate_and_summarize(self.reports)


if __name__ == "__main__":
    unittest.main()

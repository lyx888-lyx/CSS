"""No-data regression test for the validation-only history probe."""
import unittest

from scripts.compare_validation import group_labels
from scripts.probe_speaker_history import change_report


class HistoryProbeTest(unittest.TestCase):
    def test_turning_off_history_counts_changed_predictions(self):
        labels = [0, 6, 6, 0]
        tags = group_labels(labels, [0, 1, 0, 1])
        report = change_report(
            labels, [0, 6, 6, 0], [0, 6, 0, 0], tags, 7
        )
        self.assertEqual(report["total_validation_utterances"], 4)
        self.assertEqual(
            report["predictions_changed_when_module_removed"], 1
        )
        self.assertEqual(report["changed_prediction_percent"], 25.0)
        self.assertEqual(
            report["with_module"]["all"]["weighted_f1"], 100.0
        )
        self.assertLess(
            report["same_checkpoint_without_module_at_inference"]["all"]["weighted_f1"],
            100.0
        )


if __name__ == "__main__":
    unittest.main()

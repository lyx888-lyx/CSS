"""Unit tests for paired correctness migration; no MELD data or GPU."""
import unittest

from scripts.diagnose_v1_errors import (
    correctness_changes, per_group_changes
)
from scripts.compare_validation import group_labels


class TestV1PairedErrors(unittest.TestCase):
    def test_exact_fixed_broken_and_prediction_changes(self):
        # CSS correct -> V1 wrong, CSS wrong -> V1 correct,
        # both wrong with different predictions, both correct.
        y = [0, 1, 2, 3]
        css_pred = [0, 0, 0, 3]
        v1_pred = [1, 1, 1, 3]
        tags = [("first",)] * len(y)
        r = correctness_changes(y, css_pred, v1_pred, tags)
        self.assertEqual(r["n"], 4)
        self.assertEqual(r["css_correct"], 2)
        self.assertEqual(r["v1_correct"], 2)
        self.assertEqual(r["fixed"], 1)
        self.assertEqual(r["broken"], 1)
        self.assertEqual(r["both_correct"], 1)
        self.assertEqual(r["both_wrong"], 1)
        self.assertEqual(r["both_wrong_changed_answer"], 1)
        self.assertEqual(r["predicted_label_changed"], 3)
        self.assertEqual(r["net_correct_change"], 0)

    def test_group_partition_and_nested_after_other(self):
        y = [0, 1, 1, 1, 1, 0]
        sp = [0, 1, 0, 0, 1, 0]
        tags = group_labels(y, sp)
        css_pred = y.copy()
        v1_pred = y.copy()
        v1_pred[2] = 0
        v1_pred[3] = 0
        report = per_group_changes(y, css_pred, v1_pred, tags, 7)
        groups = report["groups"]
        self.assertEqual(groups["all"]["n"], 6)
        self.assertEqual(groups["first"]["n"], 2)
        self.assertEqual(groups["stable"]["n"], 2)
        self.assertEqual(groups["changed"]["n"], 2)
        self.assertEqual(groups["changed_after_other"]["n"], 2)
        self.assertEqual(groups["stable_after_other"]["n"], 1)
        self.assertEqual(groups["all"]["broken"], 2)
        self.assertEqual(groups["changed_after_other"]["broken"], 1)
        self.assertEqual(groups["stable"]["broken"], 1)
        self.assertEqual(
            groups["all"]["n"],
            groups["first"]["n"] + groups["stable"]["n"]
            + groups["changed"]["n"]
        )
        self.assertEqual(
            sum(x["n"] for x in report["by_true_emotion_class"].values()),
            6
        )

    def test_mismatched_predictions_raise(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            correctness_changes([0], [0], [], [("first",)])


if __name__ == "__main__":
    unittest.main()

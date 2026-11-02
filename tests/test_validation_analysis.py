"""Pure validation transition-report tests; no dataset bytes or GPU required."""
import unittest

from scripts.compare_validation import compute_report, group_labels


class TestValidationReports(unittest.TestCase):
    def test_first_changed_stable_and_other(self):
        # A neutral, B angry, A angry, A angry, B angry, A neutral.
        speakers = [0, 1, 0, 0, 1, 0]
        ys = [0, 6, 6, 6, 6, 0]
        tags = group_labels(ys, speakers)
        self.assertEqual(tags[0], ("first",))
        self.assertEqual(tags[1], ("first",))
        self.assertEqual(tags[2], ("changed", "changed_after_other"))
        self.assertEqual(tags[3], ("stable",))
        self.assertEqual(tags[4], ("stable", "stable_after_other"))
        self.assertEqual(tags[5], ("changed", "changed_after_other"))
        result = compute_report(ys, ys, tags, 7)
        self.assertEqual(result["all"]["n"], 6)
        self.assertEqual(result["first"]["n"], 2)
        self.assertEqual(result["changed"]["n"], 2)
        self.assertEqual(result["stable"]["n"], 2)
        self.assertEqual(result["changed_after_other"]["n"], 2)
        self.assertEqual(result["stable_after_other"]["n"], 1)
        self.assertEqual(result["all"]["weighted_f1"], 100.0)
        self.assertEqual(result["changed"]["accuracy"], 100.0)

    def test_wrong_predictions_reduce_scores(self):
        ys = [0, 1, 0]
        tags = group_labels(ys, [0, 1, 0])
        pred = [0, 0, 1]
        report = compute_report(ys, pred, tags, 7)
        self.assertLess(report["all"]["weighted_f1"], 100)
        self.assertLess(report["changed_after_other"]["accuracy"], 100)


if __name__ == "__main__":
    unittest.main()

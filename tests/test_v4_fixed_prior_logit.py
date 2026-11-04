"""Pure unit tests for V4 fixed train-prior correction. No MELD/GPU."""
import unittest

import numpy as np

from scripts.audit_v4_fixed_prior_logit import (
    FIXED_TAU, fixed_logit_adjustment, paired_report
)
from scripts.compare_validation import group_labels


class TestV4FixedPrior(unittest.TestCase):
    def test_no_adjustment_is_identical_to_css_argmax(self):
        p = np.array([
            [0.70, 0.10, 0.04, 0.04, 0.04, 0.04, 0.04],
            [0.08, 0.64, 0.06, 0.06, 0.06, 0.05, 0.05],
        ], dtype=float)
        priors = np.array([70, 10, 5, 5, 5, 3, 2], dtype=float) / 100
        np.testing.assert_array_equal(
            fixed_logit_adjustment(p, priors, tau=0),
            p.argmax(-1)
        )

    def test_rare_class_can_override_marginal_majority(self):
        # With fixed tau=0.15, prior correction may recover a
        # low-probability rare label when the fused output is close.
        p = np.array([[0.51, 0.02, 0.43, 0.01, 0.01, 0.01, 0.01]])
        priors = np.array([0.7, 0.04, 0.02, 0.04, 0.1, 0.02, 0.08])
        self.assertEqual(int(p.argmax(-1)[0]), 0)
        self.assertEqual(int(fixed_logit_adjustment(p, priors)[0]), 2)
        self.assertEqual(FIXED_TAU, 0.15)

    def test_confident_majority_stays(self):
        p = np.array([[0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
        priors = np.array([0.7, 0.04, 0.02, 0.04, 0.1, 0.02, 0.08])
        self.assertEqual(int(fixed_logit_adjustment(p, priors)[0]), 0)

    def test_requires_normalized_nonzero_prior(self):
        p = np.full((2, 7), 1.0 / 7)
        with self.assertRaises(ValueError):
            fixed_logit_adjustment(p, np.array([0, 1, 1, 1, 1, 1, 1]) / 6)
        with self.assertRaises(ValueError):
            fixed_logit_adjustment(p, np.array([1] * 7) / 10)
        with self.assertRaises(ValueError):
            fixed_logit_adjustment(p, np.array([1] * 7) / 7, tau=-0.1)

    def test_repair_counts_do_not_hide_new_errors(self):
        labels = [0, 2, 5, 0]
        previous = [0, 0, 5, 0]  # one wrong
        current = [2, 2, 5, 0]   # fixed 1, broke 1
        tags = group_labels(labels, [0, 1, 0, 1])
        _, _, report = paired_report(labels, previous, current, tags)
        self.assertEqual(report["all"]["fixed"], 1)
        self.assertEqual(report["all"]["broken"], 1)
        self.assertEqual(report["all"]["net_correct_change"], 0)


if __name__ == "__main__":
    unittest.main()

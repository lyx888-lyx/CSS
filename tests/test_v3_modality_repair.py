"""No-GPU unit tests for read-only V3 CSS modality disagreement audit."""
import unittest

import numpy as np

from scripts.audit_css_modality_repair import (
    correctable_opportunities, fixed_consensus_repair,
    paired_score_and_moves,
)
from scripts.compare_validation import group_labels


class TestV3ModalityRepair(unittest.TestCase):
    def test_two_heads_override_low_confidence_fused_with_margin(self):
        # One low-confidence fusion wrong; two strong modality heads
        # independently agree on the alternative class.
        fused = np.array([[0.48, 0.30, 0.22]])
        heads = np.array([[
            [0.04, 0.90, 0.06],
            [0.08, 0.84, 0.08],
            [0.70, 0.20, 0.10],
        ]])
        out, triggered, stats = fixed_consensus_repair(fused, heads)
        self.assertEqual(out.tolist(), [1])
        self.assertEqual(triggered.tolist(), [True])
        self.assertEqual(stats["low_fused_confidence"].tolist(), [True])

    def test_confident_fusion_not_overridden(self):
        fused = np.array([[0.85, 0.10, 0.05]])
        heads = np.array([[
            [0.02, 0.95, 0.03],
            [0.04, 0.93, 0.03],
            [0.90, 0.05, 0.05],
        ]])
        out, triggered, _ = fixed_consensus_repair(fused, heads)
        self.assertEqual(out.tolist(), [0])
        self.assertEqual(triggered.tolist(), [False])

    def test_no_agreeing_modal_heads_no_override(self):
        fused = np.array([[0.42, 0.35, 0.23]])
        heads = np.array([[
            [0.05, 0.91, 0.04],
            [0.03, 0.08, 0.89],
            [0.83, 0.12, 0.05],
        ]])
        out, triggered, stats = fixed_consensus_repair(fused, heads)
        self.assertEqual(out.tolist(), [0])
        self.assertFalse(triggered[0])
        self.assertFalse(stats["two_modal_agreement_with_fused_disagreement"][0])

    def test_rejects_non_probabilities(self):
        fused = np.array([[2.0, 0.0]])
        heads = np.full((1, 3, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            fixed_consensus_repair(fused, heads)

    def test_recovery_oracle_is_diagnostic_only(self):
        # y=1 fused=0 wrong and two modal heads correctly predict 1;
        # next y=0 fused=0 right but two heads incorrectly predict 1.
        y = [1, 0]
        pred = [0, 0]
        heads = np.array([[1, 1, 0], [1, 1, 0]])
        tags = group_labels(y, [0, 1])
        d = correctable_opportunities(y, pred, heads, tags)
        self.assertEqual(d["all"]["n"], 2)
        self.assertEqual(d["all"]["fused_wrong"], 1)
        self.assertEqual(d["all"]["wrong_with_any_correct_modal_head"], 1)
        self.assertEqual(d["all"]["wrong_with_two_or_more_correct_modal_heads"], 1)
        self.assertEqual(d["all"]["fused_correct_but_two_heads_agree_on_wrong_label"], 1)

    def test_paired_analysis_counts_both_directions(self):
        y = [1, 0, 2, 1]
        pred = [0, 0, 2, 1]
        repaired = [1, 1, 2, 0]
        tags = group_labels(y, [0, 1, 0, 1])
        _, moves = paired_score_and_moves(y, tags, pred, repaired, 3)
        self.assertEqual(moves["all"]["fixed"], 1)
        self.assertEqual(moves["all"]["broken"], 2)
        self.assertEqual(moves["all"]["net_correct_change"], -1)
        self.assertEqual(moves["all"]["predicted_label_changed"], 3)


if __name__ == "__main__":
    unittest.main()

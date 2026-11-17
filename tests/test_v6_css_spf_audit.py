"""Synthetic CPU tests for V6 CSS SPF forward decomposition."""
import unittest

import torch

from model import SPFusion
from research.spf_diagnostics import (
    decompose_spfusion, component_statistics, summarize_predictions
)
from scripts.train_research import v52_disjoint_dialogue_split_indices


class TestV6SPFAudit(unittest.TestCase):
    def test_reconstructed_output_matches_original_spf(self):
        torch.manual_seed(20261117)
        module = SPFusion(input_dim=8, rank=4, poly_order=3,
                          output_dim=8).eval()
        t = torch.randn(2, 4, 8)
        a = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        with torch.no_grad():
            original = module(t, a, v)
            components = decompose_spfusion(module, t, a, v)
        torch.testing.assert_close(
            original, components["reconstructed"],
            rtol=1e-5, atol=1e-5
        )
        self.assertEqual(len(components["order_terms"]), 3)
        self.assertEqual(len(components["gates_by_order"]), 3)
        self.assertEqual(components["polynomial"].shape, original.shape)
        self.assertEqual(components["residual"].shape, original.shape)

    def test_gate_parameters_are_global_not_utterance_conditioned(self):
        torch.manual_seed(111)
        module = SPFusion(input_dim=4, rank=3, poly_order=2, output_dim=4)
        t = torch.randn(3, 5, 4)
        result = decompose_spfusion(module, t, t, t)
        for gate in result["gates_by_order"]:
            self.assertEqual(gate.shape, (3,))
            self.assertTrue((gate > 0).all().item())
            self.assertTrue((gate < 1).all().item())

    def test_component_mask_counts_only_valid_dialogue_turns(self):
        torch.manual_seed(8)
        p, r = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        mask = torch.tensor([
            [True, True, False],
            [True, False, False]
        ])
        stats = component_statistics(p, r, mask)
        self.assertEqual(stats["n"], 3)
        self.assertGreater(stats["mean_residual_norm"], 0)
        self.assertGreaterEqual(stats["fraction_poly_norm_greater_residual"], 0)
        self.assertLessEqual(stats["fraction_poly_norm_greater_residual"], 1)

    def test_paired_prediction_removal_counts_both_fixed_and_broken(self):
        gold = [0, 1, 2, 3]
        full = [0, 2, 2, 3]
        without_poly = [1, 1, 3, 3]
        without_resid = [0, 1, 2, 2]
        report = summarize_predictions(
            gold, full, without_poly, without_resid
        )
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["without_polynomial_fixed"], 1)
        self.assertEqual(report["without_polynomial_broken"], 2)
        self.assertEqual(report["without_residual_fixed"], 1)
        self.assertEqual(report["without_residual_broken"], 1)

    def test_old_and_new_valid_dialogues_excluded(self):
        train, fresh_val, old_val = v52_disjoint_dialogue_split_indices(
            1152, 0.1, 24117
        )
        self.assertEqual((len(train), len(fresh_val), len(old_val)),
                         (933, 104, 115))
        self.assertFalse(set(train) & set(fresh_val))
        self.assertFalse(set(train) & set(old_val))


if __name__ == "__main__":
    unittest.main()

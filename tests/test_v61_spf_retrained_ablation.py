"""V6.1 synthetic CPU checks for strict retrained SPF ablation."""
import unittest

import torch

from model import SPFusion
from research.spf_diagnostics import decompose_spfusion
from scripts.train_research import (
    v52_disjoint_dialogue_split_indices,
    v61_disjoint_spf_ablation_split_indices
)


class TestV61SPFRetrainedAblation(unittest.TestCase):
    def test_original_mode_preserves_complete_spf(self):
        torch.manual_seed(116)
        spf = SPFusion(input_dim=8, rank=3, poly_order=4, output_dim=8)
        t, a, v = [torch.randn(2, 5, 8) for _ in range(3)]
        self.assertEqual(spf.ablation_mode, "full")
        out = spf(t, a, v)
        decomposed = decompose_spfusion(spf, t, a, v)
        torch.testing.assert_close(out, decomposed["reconstructed"])

    def test_residual_only_exactly_skips_polynomial_even_at_train_time(self):
        torch.manual_seed(117)
        spf = SPFusion(input_dim=8, rank=3, poly_order=4, output_dim=8)
        t, a, v = [torch.randn(2, 5, 8, requires_grad=True)
                   for _ in range(3)]
        spf.ablation_mode = "residual_only"
        spf.train()
        output = spf(t, a, v)
        reference = spf.residual_proj(torch.cat([t, a, v], dim=-1))
        torch.testing.assert_close(output, reference)
        output.sum().backward()
        self.assertIsNotNone(spf.residual_proj.weight.grad)
        self.assertIsNone(spf.output_proj.grad)
        self.assertIsNone(spf.output_bias.grad)
        self.assertTrue(
            all(p.grad is None for p in spf.weight_list_text)
        )
        self.assertTrue(
            all(p.grad is None for p in spf.modality_gates)
        )

    def test_switch_does_not_alter_state_dict_or_initial_parameters(self):
        torch.manual_seed(118)
        full = SPFusion(input_dim=8, rank=4, poly_order=3, output_dim=8)
        torch.manual_seed(118)
        residual = SPFusion(input_dim=8, rank=4, poly_order=3, output_dim=8)
        residual.ablation_mode = "residual_only"
        self.assertEqual(set(full.state_dict()), set(residual.state_dict()))
        for key in full.state_dict():
            torch.testing.assert_close(
                full.state_dict()[key], residual.state_dict()[key]
            )

    def test_three_cohort_exclusions_and_v6_probe_exclusion(self):
        train, new_val, old_val, v52_val, v6_seen = (
            v61_disjoint_spf_ablation_split_indices(
                1152, 0.1, split_seed=36427
            )
        )
        self.assertEqual((len(train), len(new_val)), (846, 87))
        self.assertEqual((len(old_val), len(v52_val)), (115, 104))
        self.assertEqual(len(v6_seen), 65)
        self.assertFalse(set(new_val) & set(old_val))
        self.assertFalse(set(new_val) & set(v52_val))
        self.assertFalse(set(new_val) & set(v6_seen))
        self.assertFalse(set(train) & (set(old_val) | set(v52_val)))
        self.assertEqual(len(set(train+new_val+old_val+v52_val)), 1152)
        self.assertTrue(set(v6_seen).issubset(set(train)))

    def test_split_is_deterministic_and_new_val_differs_from_v52(self):
        first = v61_disjoint_spf_ablation_split_indices(
            1152, 0.1, 36427
        )
        second = v61_disjoint_spf_ablation_split_indices(
            1152, 0.1, 36427
        )
        self.assertEqual(first, second)
        _, older_val, _ = v52_disjoint_dialogue_split_indices(
            1152, 0.1, 24117
        )
        self.assertFalse(set(first[1]) & set(older_val))


if __name__ == "__main__":
    unittest.main()

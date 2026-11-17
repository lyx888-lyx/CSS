"""V6.3 synthetic CPU tests: exact SPF-to-linear-head logit attribution."""
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model import SPFusion
from research.spf_diagnostics import decompose_spfusion
from research.spf_logit_diagnostics import (
    decompose_linear_logits, per_turn_logit_statistics,
    summarize_turns, true_class_margin
)
from scripts.train_research import (
    v52_disjoint_dialogue_split_indices,
    v61_disjoint_spf_ablation_split_indices
)


class TestV63SPFLogits(unittest.TestCase):
    def test_exact_original_spf_and_classifier_logit_decomposition(self):
        torch.manual_seed(20261117)
        spf = SPFusion(input_dim=8, rank=4, poly_order=4,
                       output_dim=8).eval()
        head = torch.nn.Linear(8, 7).eval()
        t, a, v = [torch.randn(2, 3, 8) for _ in range(3)]
        with torch.no_grad():
            source_features = spf(t, a, v)
            d = decompose_spfusion(spf, t, a, v)
            z = decompose_linear_logits(
                head, d["polynomial"], d["residual"]
            )
            torch.testing.assert_close(
                head(source_features), z["full_logits"],
                atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                head(d["residual"]), z["residual_only_logits"],
                atol=1e-5, rtol=1e-5
            )
            self.assertEqual(z["poly_logits"].shape, (2, 3, 7))

    def test_margin_exact_and_true_vs_rival(self):
        z = torch.tensor([[2., 1., 0.], [0., 4., 1.]])
        y = torch.tensor([0, 1])
        self.assertEqual(true_class_margin(z, y).tolist(), [1., 3.])
        with self.assertRaises(ValueError):
            true_class_margin(z, torch.tensor([2, 3]))
        with self.assertRaises(ValueError):
            true_class_margin(z, torch.tensor([1]))

    def test_logits_with_uniform_class_offset_cannot_change_decision(self):
        head = torch.nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([
                [1., 0.], [1., 0.], [1., 0.]
            ]))
            head.bias.copy_(torch.tensor([1., 0., -1.]))
        p = torch.tensor([[[2., 0.]]])
        r = torch.tensor([[[1., 0.]]])
        d = decompose_linear_logits(head, p, r)
        rows = per_turn_logit_statistics(
            d["full_logits"].reshape(1, 3),
            d["residual_only_logits"].reshape(1, 3),
            d["poly_logits"].reshape(1, 3),
            d["residual_logits_without_bias"].reshape(1, 3),
            torch.tensor([0])
        )
        self.assertEqual(rows[0]["full_pred"], rows[0]["residual_pred"])
        self.assertAlmostEqual(rows[0]["centered_poly_logit_norm"], 0.0)
        self.assertAlmostEqual(rows[0]["probability_total_variation"], 0.0,
                               places=6)
        self.assertAlmostEqual(rows[0]["poly_true_margin_delta"], 0.0,
                               places=6)

    def test_margin_can_change_with_no_top1_flip(self):
        logits_full = torch.tensor([[3.0, 2.0, 0.0]])
        logits_res = torch.tensor([[2.0, 1.5, 0.0]])
        poly = logits_full - logits_res
        rows = per_turn_logit_statistics(
            logits_full, logits_res, poly, logits_res,
            torch.tensor([0])
        )
        summary = summarize_turns(rows)
        self.assertEqual(summary["flip_count"], 0)
        self.assertEqual(summary["n"], 1)
        self.assertGreater(summary["mean_abs_poly_true_margin_delta"], 0)
        self.assertGreater(summary["mean_probability_total_variation"], 0)
        self.assertEqual(summary["fraction_true_margin_improved_by_poly"], 1)

    def test_fixed_V6_training_samples_are_inside_V61_training_cohort(self):
        v52_train, v52_val, old_val = (
            v52_disjoint_dialogue_split_indices(1152, 0.1, 24117)
        )
        v61_train, v61_val, old_again, v52_again, seen = (
            v61_disjoint_spf_ablation_split_indices(
                1152, 0.1, 36427
            )
        )
        selected = set(np.random.default_rng(73117).choice(
            v52_train, size=64, replace=False
        ).tolist())
        self.assertEqual(len(selected), 64)
        self.assertTrue(selected.issubset(v61_train))
        self.assertTrue(selected.issubset(seen))
        self.assertFalse(selected & set(v61_val))
        self.assertEqual(old_val, old_again)
        self.assertEqual(v52_val, v52_again)

    def test_all_global_spf_gates_start_at_sigmoid_one(self):
        spf = SPFusion(input_dim=4, rank=2, poly_order=4, output_dim=4)
        initial = 1.0/(1.0+math.exp(-1.0))
        for v in spf.modality_gates:
            for x in torch.sigmoid(v).tolist():
                self.assertAlmostEqual(x, initial, places=6)


if __name__ == "__main__":
    unittest.main()

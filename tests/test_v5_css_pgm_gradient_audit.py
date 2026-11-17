"""CPU synthetic tests for observational CSS-PGM gradient audit."""
import unittest

import torch
from torch import nn

from research.pgm_diagnostics import (
    gradient_geometry, summarise_batch_geometry,
    weighted_norm_shares
)
from scripts.audit_css_pgm_gradients import (
    flatten_aligned, select_source_aligned_parameters,
    summarise_class_statistics,
)


class TestPGMGeometry(unittest.TestCase):
    def test_negative_orthogonal_and_positive_gradients(self):
        g = gradient_geometry([
            torch.tensor([1.0, 0.0]),
            torch.tensor([-2.0, 0.0]),
            torch.tensor([0.0, 3.0]),
        ])
        self.assertAlmostEqual(g["cosines"]["L1_L2"], -1.0)
        self.assertAlmostEqual(g["cosines"]["L1_L3"], 0.0)
        self.assertAlmostEqual(g["cosines"]["L2_L3"], 0.0)
        self.assertAlmostEqual(g["norms"]["L2"], 2.0)

    def test_zero_gradient_is_undefined_not_zero_cosine(self):
        g = gradient_geometry([
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([4.0, 0.0]),
        ])
        self.assertIsNone(g["cosines"]["L1_L2"])
        self.assertIsNone(g["cosines"]["L2_L3"])
        self.assertAlmostEqual(g["cosines"]["L1_L3"], 1.0)

    def test_rejects_misaligned_gradients(self):
        with self.assertRaises(ValueError):
            gradient_geometry([
                torch.ones(3), torch.ones(4), torch.ones(3)
            ])
        with self.assertRaises(ValueError):
            gradient_geometry([
                torch.ones(3), torch.ones(3), torch.tensor([1.0, float("nan"), 0])
            ])

    def test_weighted_raw_norm_shares_not_just_alpha(self):
        d = weighted_norm_shares(
            {"L1": 10.0, "L2": 1.0, "L3": 1.0},
            [1/3, 1/3, 1/3]
        )
        self.assertAlmostEqual(d["L1"], 10/12)
        self.assertAlmostEqual(sum(d.values()), 1.0)

    def test_batch_geometry_with_partial_zero(self):
        a = gradient_geometry([
            torch.tensor([1., 0.]),
            torch.tensor([-1., 0.]),
            torch.tensor([0., 0.])
        ])
        b = gradient_geometry([
            torch.tensor([1., 0.]),
            torch.tensor([1., 0.]),
            torch.tensor([0., 2.])
        ])
        agg = summarise_batch_geometry([a, b])
        self.assertEqual(agg["pairwise"]["L1_L2"]["conflicting_batches"], 1)
        self.assertEqual(agg["pairwise"]["L1_L2"]["measured_batches"], 2)
        self.assertEqual(agg["pairwise"]["L1_L3"]["measured_batches"], 1)

    def test_flattens_missing_gradients_as_aligned_zeros(self):
        p1 = nn.Parameter(torch.zeros(2))
        p2 = nn.Parameter(torch.zeros(3))
        vec = flatten_aligned(
            (torch.tensor([2.0, 3.0]), None), [p1, p2]
        )
        self.assertEqual(vec.tolist(), [2., 3., 0., 0., 0.])

    def test_selected_modules_match_original_pgm_source(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                from scripts.audit_css_pgm_gradients import PARAM_GROUPS
                for _, group in PARAM_GROUPS:
                    for name in group:
                        setattr(self, name, nn.Linear(2, 2))
                self.last_gate = nn.Linear(2, 2)
                self.all_output_layer = nn.Linear(2, 2)

        model = ToyModel()
        selected, spf, mapping = select_source_aligned_parameters(model)
        self.assertEqual(len(mapping), 4)
        self.assertEqual(len({id(p) for p in selected}), len(selected))
        self.assertFalse({id(p) for p in selected} & {id(p) for p in spf})
        self.assertFalse(
            {id(p) for p in selected} &
            {id(p) for p in model.all_output_layer.parameters()}
        )

    def test_empty_rare_class_reporting_is_null(self):
        from collections import defaultdict
        stat = defaultdict(lambda: {
            "n": 0, "correct": 0,
            "sum_L1": 0.0, "sum_L2": 0.0, "sum_L3": 0.0,
            "sum_fused_logit_gradient_norm": 0.0
        })
        result = summarise_class_statistics(stat)
        self.assertEqual(result["Fear"]["n_train_sample"], 0)
        self.assertIsNone(result["Fear"]["mean_per_utterance_losses"]["L1"])
        self.assertIsNone(result["Disgust"]["fused_NLL_logit_gradient_norm"])


if __name__ == "__main__":
    unittest.main()

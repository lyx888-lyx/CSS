"""CPU tests for V5.2 mathematically matched PGM direction update.

No MELD pickle or model checkpoint needed.
"""
import unittest

import torch

from model import PGM
from research.pgm_direction_consistent import (
    DirectionAlignedPGM, derive_direction_aligned_coefficients
)


class TestV52PGMDirectionAlignment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)

    def test_exact_shared_direction_and_magnitude_match(self):
        p = torch.nn.Parameter(torch.tensor([0.2, -0.5, 0.3]))
        tasks = [
            2.0*p[0] + 0.4*p[1] + 0.01*p[2],
            -0.3*p[0] + 7.0*p[1] + 0.2*p[2],
            0.05*p[0] + 0.1*p[1] + 0.2*p[2]
        ]
        raw_task_grads = [
            torch.autograd.grad(l, p, retain_graph=True)[0].detach()
            for l in tasks
        ]
        alpha = torch.tensor([0.34, 0.26, 0.40])
        coeff, meta = derive_direction_aligned_coefficients(
            raw_task_grads, alpha
        )
        self.assertFalse(coeff.requires_grad)
        self.assertFalse(any(x.requires_grad for x in raw_task_grads))
        original = torch.stack(
            [a*g for a, g in zip(alpha, raw_task_grads)]
        ).sum(0)
        unit = torch.stack([
            a*g/(g.norm()+1e-8)
            for a, g in zip(alpha, raw_task_grads)
        ]).sum(0)
        expected = unit * original.norm()/unit.norm()
        new_loss = sum(c*l for c, l in zip(coeff, tasks))
        actual = torch.autograd.grad(new_loss, p)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(
            float(actual.norm()), float(original.norm()), places=5
        )
        self.assertAlmostEqual(
            float(torch.dot(actual, expected)/(actual.norm()*expected.norm())),
            1.0, places=6
        )
        self.assertGreater(meta["direction_angle_deg"], 0)

    def test_equal_task_norms_yield_original_coefficient(self):
        grads = [
            torch.tensor([2.0, 0, 0]),
            torch.tensor([0, 2.0, 0]),
            torch.tensor([0, 0, 2.0])
        ]
        alpha = torch.tensor([0.3, 0.4, 0.3])
        coeff, diag = derive_direction_aligned_coefficients(grads, alpha)
        torch.testing.assert_close(coeff, alpha, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(diag["direction_angle_deg"], 0.0, places=4)

    def test_scale_invariance_of_task_gradient(self):
        # Doubling a task loss multiplies its raw norm and changes the
        # global matching scale, but the normalized target direction
        # remains defined by alpha and per-task unit gradients.
        grads = [
            torch.tensor([2., 1., 0.]),
            torch.tensor([-1., 3., 0.]),
            torch.tensor([0., 1., 1.])
        ]
        a = torch.tensor([.3, .3, .4])
        coeff_a, first = derive_direction_aligned_coefficients(grads, a)
        doubled = [grads[0], 4*grads[1], grads[2]]
        coeff_b, second = derive_direction_aligned_coefficients(doubled, a)
        update_a = sum(c*g for c, g in zip(coeff_a, grads))
        update_b = sum(c*g for c, g in zip(coeff_b, doubled))
        # The global matching scale changes, but the direction
        # depends only on alpha and the three UNIT task gradients.
        torch.testing.assert_close(
            update_a / update_a.norm(), update_b / update_b.norm(),
            rtol=1e-6, atol=1e-6
        )
        self.assertGreater(
            second["original_task_gradient_norms"][1],
            first["original_task_gradient_norms"][1]
        )

    def test_invalid_norm_alpha_or_dimensions_fail(self):
        valid = [
            torch.tensor([1., 0., 0.]),
            torch.tensor([0., 1., 0.]),
            torch.tensor([0., 0., 1.])
        ]
        a = torch.tensor([0.3, 0.3, 0.4])
        for bad_grads, bad_alpha in (
            ([torch.zeros(3), valid[1], valid[2]], a),
            ([valid[0], valid[1], torch.ones(4)], a),
            (valid, torch.tensor([0.2, 0.2, 0.2])),
            (valid, torch.tensor([0.3, -0.2, 0.9]))
        ):
            with self.subTest(bad_grads=bad_grads, bad_alpha=bad_alpha):
                with self.assertRaises(ValueError):
                    derive_direction_aligned_coefficients(
                        bad_grads, bad_alpha
                    )

    def test_original_pgm_alpha_is_reused(self):
        x = torch.nn.Parameter(torch.tensor([0.2, 0.4, 0.7]))
        class MiniPGM(DirectionAlignedPGM):
            def _compute_task_losses(self, outputs, labels_, umask):
                z = outputs
                return [
                    2*z[0] + z[2],
                    z[0] + 3*z[1],
                    0.01*z[1] + 0.1*z[2]
                ]
        pgm = MiniPGM(None, None)
        tasks = pgm._compute_task_losses(x, None, None)
        grads = pgm._compute_gradients(tasks, [x])
        expected_alpha = pgm._project(grads).clone()
        total_loss, actual_alpha, returned = pgm(x, None, None, [x])
        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertEqual(len(returned), 3)
        torch.testing.assert_close(actual_alpha, expected_alpha, atol=1e-5, rtol=1e-5)
        self.assertIsNotNone(pgm.last_diagnostics)


if __name__ == "__main__":
    unittest.main()

"""Synthetic math/experimental-protocol tests for grey forecasting baselines."""
import unittest

import numpy as np

from scripts.benchmark_gm_probabilities import (
    ar1_forecast, ema_forecast, gm11_next_scalar,
    gm_forecast, is_gm_level_ratio_admissible,
    probability_normalize,
)


class TestGM11LocalBenchmark(unittest.TestCase):
    def test_constant_positive_series_one_step(self):
        x = np.array([0.3, 0.3, 0.3, 0.3])
        self.assertAlmostEqual(gm11_next_scalar(x), 0.3, places=6)
        self.assertTrue(is_gm_level_ratio_admissible(x))

    def test_positive_approximately_exponential_series(self):
        x = np.array([0.1, 0.11, 0.121, 0.1331])
        prediction = gm11_next_scalar(x)
        self.assertTrue(np.isfinite(prediction))
        self.assertLess(abs(prediction - 0.14641), 0.02)

    def test_nonadmissible_spike(self):
        self.assertFalse(is_gm_level_ratio_admissible([0.1, 0.1, 0.9, 0.9]))

    def test_nonnegative_requirement_and_minimum_length(self):
        for series in ([0.1, 0.2, 0.3], [-1.0, 0.2, 0.3, 0.4]):
            with self.subTest(series=series):
                with self.assertRaises(ValueError):
                    gm11_next_scalar(series)

    def test_all_methods_output_finite_simplex_probabilities(self):
        h = np.array([
            [0.75, 0.20, 0.05],
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.55, 0.35, 0.10],
        ])
        for forecast in (
            lambda x: ar1_forecast(x),
            lambda x: ema_forecast(x),
            lambda x: gm_forecast(x)[0],
            lambda x: gm_forecast(x, level_ratio_guard=True)[0],
        ):
            out = forecast(h)
            self.assertEqual(out.shape, (3,))
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertTrue(np.all(out > 0))
            self.assertAlmostEqual(float(out.sum()), 1.0)

    def test_guarded_gm_falls_back_for_unstable_component(self):
        h = np.array([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
        ])
        p, diag = gm_forecast(h, level_ratio_guard=True)
        self.assertEqual(diag["fit_fallbacks"], 2)
        self.assertEqual(diag["admissible_components"], 0)
        np.testing.assert_allclose(p, h[-1], atol=1e-7)

    def test_probability_normalization_handles_nonfinite(self):
        p = probability_normalize([np.nan, 0.3], [0.7, 0.3])
        np.testing.assert_allclose(p, [0.7, 0.3])

    def test_probability_normalization_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            probability_normalize([0.1, 0.9], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

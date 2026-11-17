"""No-data unit checks for the preregistered V7 3-seed summary."""
import unittest

from scripts.summarize_v7_seed_controls import (
    GROUPS, SEEDS, summarize
)


def fixture(seed, val):
    return {
        "seed": seed,
        "groups": {
            group: {"delta_residual_minus_full": val}
            for group in GROUPS
        }
    }


class TestV7ThreeSeedSummary(unittest.TestCase):
    def test_fixed_seed_order_and_mean_sd(self):
        rows = [
            fixture(10068, 0.29),
            fixture(21431, -0.25),
            fixture(31907, 0.53),
        ]
        result = summarize(rows)
        self.assertAlmostEqual(
            result["all"]["mean_delta_pp"], (0.29-0.25+0.53)/3
        )
        self.assertEqual(result["all"]["positive_seed_count"], 2)
        self.assertEqual(result["all"]["negative_seed_count"], 1)
        self.assertGreater(result["all"]["seed_sd_delta_pp"], 0)

    def test_wrong_or_cherrypicked_seed_list_rejected(self):
        with self.assertRaises(ValueError):
            summarize([
                fixture(10068, 0.2),
                fixture(777, 9.5),
                fixture(31907, 0.3),
            ])
        with self.assertRaises(ValueError):
            summarize([fixture(10068, 0.2)])


if __name__ == "__main__":
    unittest.main()

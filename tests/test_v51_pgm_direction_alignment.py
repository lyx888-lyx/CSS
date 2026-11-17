"""V5.1 pure-stdlib tests: exact angle from saved pairwise PGM geometry."""
import math
import unittest

from scripts.audit_pgm_direction_alignment import per_batch, summarize


def make_batch(norms=(1., 1., 1.),
               cos=(0., 0., 0.),
               alpha=(1/3, 1/3, 1/3)):
    return {
        "pgm_shared": {
            "norms": dict(zip(("L1", "L2", "L3"), norms)),
            "cosines": dict(zip(("L1_L2", "L1_L3", "L2_L3"), cos)),
        },
        "pgm_weights": dict(zip(("L1", "L2", "L3"), alpha)),
        "utterances": 8,
    }


class TestV51DirectionAlignment(unittest.TestCase):
    def test_identical_norms_identical_directions(self):
        row = per_batch(make_batch(
            norms=(4, 4, 4), cos=(0.2, 0.0, 0.1)
        ), 1)
        self.assertAlmostEqual(row[
            "angle_deg_between_PGM_target_and_actual_update"
        ], 0.0, places=5)
        for name in ("L1", "L2", "L3"):
            self.assertAlmostEqual(
                row["PGM_target_task_cosines"][name],
                row["actual_update_task_cosines"][name],
                places=6
            )

    def test_known_orthonormal_gradient_angle(self):
        row = per_batch(make_batch(norms=(2, 1, 1)), 2)
        # unit direction = (1/3,1/3,1/3); actual = (2/3,1/3,1/3)
        expected = math.degrees(
            math.acos(4.0 / (math.sqrt(3) * math.sqrt(6)))
        )
        self.assertAlmostEqual(
            row["angle_deg_between_PGM_target_and_actual_update"],
            expected, places=6
        )
        self.assertAlmostEqual(
            row["pre_cancellation_raw_magnitude_share"]["L1"],
            0.5
        )

    def test_collinear_gradients_not_necessarily_direction_mismatch(self):
        # If all unit task gradients point the SAME way, their
        # raw norm differences don't change the joint direction.
        row = per_batch(make_batch(
            norms=(1, 5, 20), cos=(1, 1, 1)
        ), 3)
        self.assertAlmostEqual(
            row["angle_deg_between_PGM_target_and_actual_update"], 0,
            places=5
        )

    def test_invalid_cosine_and_zero_grad_rejected(self):
        for invalid in (
            make_batch(norms=(0, 1, 1)),
            make_batch(cos=(None, 0, 0)),
            make_batch(cos=(float("nan"), 0, 0)),
            make_batch(alpha=(0.8, 0.8, -0.6)),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    per_batch(invalid)

    def test_summary_counts_batch_angles(self):
        rows = [
            per_batch(make_batch(norms=(1, 1, 1)), 1),
            per_batch(make_batch(norms=(12, 1, 1)), 2),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["batches"], 2)
        self.assertEqual(summary["batches_angle_gt_20_deg"], 1)
        self.assertGreater(summary["mean_angle_deg"], 0)


if __name__ == "__main__":
    unittest.main()

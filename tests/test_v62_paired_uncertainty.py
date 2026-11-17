"""Synthetic, zero-data tests for V6.2 paired-dialogue bootstrap."""
import unittest

from research.spf_paired_uncertainty import (
    bootstrap_dialogues, ensure_paired_dialogues,
    per_class_paired, score_group
)


def toy():
    return [
        {
            "dialogue_id": "d1",
            "labels": [0, 0, 1, 1],
            "tags": [["first"], ["stable"], ["first"], ["stable"]],
            "full": [0, 0, 0, 1],
            "residual": [0, 1, 1, 1],
        },
        {
            "dialogue_id": "d2",
            "labels": [0, 1, 1],
            "tags": [["first"], ["changed", "changed_after_other"],
                     ["stable", "stable_after_other"]],
            "full": [0, 0, 1],
            "residual": [0, 1, 0],
        },
        {
            "dialogue_id": "d3",
            "labels": [1, 0, 0],
            "tags": [["first"], ["changed"], ["stable"]],
            "full": [0, 0, 1],
            "residual": [1, 0, 0],
        },
    ]


class TestV62PairedUncertainty(unittest.TestCase):
    def test_paired_fix_break_bookkeeping(self):
        t = toy()
        ensure_paired_dialogues(t)
        s = score_group(t, "all")
        self.assertEqual(s["n"], 10)
        self.assertEqual(s["residual_fixed"], 4)
        self.assertEqual(s["residual_broken"], 2)
        self.assertAlmostEqual(
            s["delta_residual_minus_full"],
            s["residual_f1"] - s["full_f1"]
        )

    def test_changed_group_uses_tags_not_naive_previous_label(self):
        s = score_group(toy(), "changed_after_other")
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["residual_fixed"], 1)
        self.assertEqual(s["residual_broken"], 0)

    def test_pairwise_class_support_and_fixed_counts(self):
        c = per_class_paired(toy())
        self.assertEqual(c["Neutral"]["support"], 5)
        self.assertEqual(c["Surprise"]["support"], 5)
        self.assertEqual(c["Neutral"]["full_correct"], 4)
        self.assertEqual(c["Neutral"]["residual_correct"], 4)

    def test_repeatable_whole_dialogue_bootstrap(self):
        a = bootstrap_dialogues(toy(), reps=100, seed=22)
        b = bootstrap_dialogues(toy(), reps=100, seed=22)
        self.assertEqual(a, b)
        self.assertEqual(a["all"]["bootstrap_valid_replicates"], 100)
        self.assertEqual(len(a["all"]["ci95_pp"]), 2)

    def test_empty_group_returns_none(self):
        t = toy()
        res = score_group(t, "missing")
        self.assertEqual(res["n"], 0)
        self.assertIsNone(res["delta_residual_minus_full"])

    def test_duplicate_and_misaligned_dialogues_rejected(self):
        t = toy()
        with self.assertRaises(ValueError):
            ensure_paired_dialogues(t + [t[0]])
        broken = [dict(t[0], full=[0]), t[1]]
        with self.assertRaises(ValueError):
            ensure_paired_dialogues(broken)


if __name__ == "__main__":
    unittest.main()

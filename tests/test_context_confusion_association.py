"""Synthetic CPU unit tests for gold-label posthoc context confusion analysis.

Does not load IEMOCAP files, TEST data, checkpoints, or models.
"""
import unittest
import numpy as np

from research.context_confusion_association import (
    normalize_speakers, dialogue_observations, analyze
)


class TestContextConfusionAssociation(unittest.TestCase):
    def sample(self, session):
        # A-Frustrated -> B-Excited -> A-Angry(err as Frustrated)
        # -> A-Angry(correct) -> B-Excited -> A-Frustrated(err as Angry)
        return dialogue_observations(
            f"{session}M_impro00",
            ["M", "F", "M", "M", "F", "M"],
            [5, 4, 3, 3, 4, 5],
            [5, 4, 5, 3, 4, 3],
        )

    def test_transition_and_intervening_speaker(self):
        rows = self.sample("Ses01")
        self.assertIsNone(rows[0]["own_emotion_changed"])
        self.assertIsNone(rows[1]["other_intervened"])
        self.assertTrue(rows[2]["own_emotion_changed"])
        self.assertTrue(rows[2]["other_intervened"])
        self.assertEqual(rows[2]["prev_own_true_label"], 5)
        self.assertFalse(rows[3]["own_emotion_changed"])
        self.assertFalse(rows[3]["other_intervened"])
        self.assertTrue(rows[5]["own_emotion_changed"])
        self.assertTrue(rows[5]["other_intervened"])

    def test_fixed_current_class_direction_and_group_denominators(self):
        rows = [r for s in ("Ses01", "Ses02", "Ses03", "Ses04")
                for r in self.sample(s)]
        result = analyze(rows)
        angry = result["directions"]["angry->frustrated"]
        self.assertEqual(angry["current_true_class_support"], 8)
        self.assertEqual(angry["mistaken_as_target"], 4)
        self.assertEqual(angry["has_prior_own"], 8)
        changed = angry["features"]["own_emotion_changed"]
        self.assertEqual(changed["pooled"]["present"]["mistakes"], 4)
        self.assertEqual(changed["pooled"]["present"]["n"], 4)
        self.assertEqual(changed["pooled"]["absent"]["mistakes"], 0)
        self.assertEqual(changed["pooled"]["absent"]["n"], 4)
        self.assertEqual(changed["pooled"]["risk_difference_pp"], 100.0)
        self.assertEqual(changed["session_positive_gap_count"], 4)
        history_target = angry["features"]["previous_own_label_matches_wrong_target"]
        self.assertEqual(history_target["pooled"]["present"]["n"], 4)
        self.assertEqual(history_target["pooled"]["present"]["mistakes"], 4)

    def test_zero_group_does_not_impute_rate(self):
        rows = [r for s in ("Ses01", "Ses02", "Ses03", "Ses04")
                for r in self.sample(s)]
        result = analyze(rows)
        # For true happy, every synthetic example is absent,
        # so the comparison is undefined rather than falsely 0%.
        happy = result["directions"]["happy->excited"]
        self.assertEqual(happy["current_true_class_support"], 0)
        self.assertIsNone(happy["features"]["own_emotion_changed"]
                          ["pooled"]["present"]["error_pct"])
        self.assertIsNone(happy["features"]["own_emotion_changed"]
                          ["pooled"]["risk_difference_pp"])

    def test_reject_misalignment_test_ids_and_duplicate_turns(self):
        with self.assertRaises(ValueError):
            dialogue_observations("Ses05M", ["M"], [3], [3])
        with self.assertRaises(ValueError):
            dialogue_observations("Ses01M", ["M", "F"], [3], [3])
        with self.assertRaises(ValueError):
            dialogue_observations("Ses01M", ["M"], [3], [6])
        with self.assertRaises(ValueError):
            normalize_speakers(np.ones((2, 3)))
        rows = [r for s in ("Ses01", "Ses02", "Ses03", "Ses04")
                for r in self.sample(s)]
        with self.assertRaises(ValueError):
            analyze(rows + [rows[0]])

    def test_one_hot_speakers(self):
        encoded = np.asarray([[1, 0], [0, 1], [1, 0]])
        self.assertEqual(normalize_speakers(encoded), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()

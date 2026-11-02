"""Unit tests for finer speaker-return analysis. Run with unittest discover."""
import unittest

from scripts.analyze_speaker_interactions import run_analysis


class SpeakerReturnAnalysisTest(unittest.TestCase):
    def test_gap_and_last_other(self):
        # 0(A), 1(B), 2(A), 3(B), 4(A), 5(A)
        speakers = {"d": ["A", "B", "A", "B", "A", "A"]}
        labels = {"d": [0, 1, 1, 2, 2, 2]}
        out = run_analysis(speakers, labels, ["d"])
        self.assertEqual(out["events"]["all_return_events"], 4)
        self.assertEqual(out["events"]["with_intervening_other"], 3)
        self.assertEqual(out["events"]["adjacent_same_speaker"], 1)
        self.assertEqual(out["speaker_return_change_rate_by_gap"]["2"]["samples"], 3)
        diff = out["last_other_different_emotion_from_previous_self"]
        self.assertEqual(diff["samples"], 2)
        self.assertEqual(diff["current_matches_last_other"], 2)
        self.assertEqual(diff["current_keeps_previous_self"], 0)
        self.assertEqual(
            out["last_other_same_emotion_as_previous_self"]["samples"], 1
        )

    def test_one_hot_multispeaker(self):
        speakers = {"d": [[1,0,0],[0,1,0],[1,0,0]]}
        labels = {"d": [0,1,0]}
        out = run_analysis(speakers, labels, ["d"])
        self.assertEqual(out["events"]["with_intervening_other"], 1)
        self.assertEqual(out["events"]["all_return_events"], 1)


if __name__ == "__main__":
    unittest.main()

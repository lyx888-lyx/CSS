"""No-dataset CPU tests for GM(1,1) history coverage audit."""
import unittest

from scripts.audit_gm_history_feasibility import audit_dialogues


class TestGreyHistoryFeasibility(unittest.TestCase):
    def test_interleaved_speakers_have_two_eligible_targets(self):
        d = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        r = audit_dialogues([d], minimum_history=4)
        self.assertEqual(r["dialogues"], 1)
        self.assertEqual(r["speaker_dialogue_tracks"], 2)
        self.assertEqual(r["utterances"], 10)
        self.assertEqual(r["eligible_prediction_turns"], 2)
        self.assertEqual(r["eligible_with_other_between"], 2)
        self.assertEqual(r["speaker_return_with_other_between"], 8)
        self.assertEqual(r["speaker_return_turns"], 8)
        self.assertEqual(r["mean_utterances_per_speaker_track"], 5.0)

    def test_five_adjacent_utterances_have_one_gm_forecast_target(self):
        r = audit_dialogues([[0, 0, 0, 0, 0]], minimum_history=4)
        self.assertEqual(r["eligible_prediction_turns"], 1)
        self.assertEqual(r["eligible_with_other_between"], 0)
        self.assertEqual(r["insufficient_history_turns"], 4)

    def test_short_tracks_are_not_gm_eligible(self):
        r = audit_dialogues([[0, 1, 0], [1, 2, 2, 1]])
        self.assertEqual(r["eligible_prediction_turns"], 0)
        self.assertEqual(r["speaker_dialogue_tracks"], 4)
        self.assertEqual(r["utterances"], 7)

    def test_rejects_invalid_input(self):
        for dialogues, minimum in [([[]], 4), ([[1, 2]], 0)]:
            with self.subTest(dialogues=dialogues, minimum=minimum):
                with self.assertRaises(ValueError):
                    audit_dialogues(dialogues, minimum_history=minimum)


if __name__ == "__main__":
    unittest.main()

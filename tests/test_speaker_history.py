"""Tests for the first speaker-dynamics diagnostics and history masks."""
import unittest

import torch

from research.speaker_history import build_speaker_history_masks
from scripts.analyze_speaker_dynamics import analyze, speaker_ids_for_dialogue


class TestSpeakerHistory(unittest.TestCase):
    def test_past_only_and_two_types(self):
        speaker = torch.tensor([[0, 1, 0, 2, 0]])
        valid = torch.ones((1, 5), dtype=torch.bool)
        own, other = build_speaker_history_masks(speaker, valid)
        self.assertEqual(own.shape, (1, 5, 5))
        self.assertTrue(own[0, 4, 0].item())
        self.assertTrue(own[0, 4, 2].item())
        self.assertTrue(other[0, 4, 1].item())
        self.assertTrue(other[0, 4, 3].item())
        self.assertFalse(own[0, 1, 2].item())  # future
        self.assertFalse(other[0, 1, 1].item())  # current turn
        self.assertEqual(own.logical_and(other).sum().item(), 0)
        self.assertEqual(own.sum().item(), 3)
        self.assertEqual(other.sum().item(), 7)

    def test_padding_and_window(self):
        speaker = torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0]])
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        own, other = build_speaker_history_masks(speaker, valid, max_lag=1)
        self.assertFalse(own[0, 2, 0].item())  # distance 2
        self.assertTrue(other[0, 2, 1].item())
        self.assertFalse(own[:, 3, :].any().item())
        self.assertFalse(other[:, :, 3].any().item())
        self.assertTrue(own[1, 1, 0].item())

    def test_descriptive_change_counts(self):
        # A happy -> B angry -> C calm -> A angry -> A angry.
        speakers = {"d": ["A", "B", "C", "A", "A"]}
        labels = {"d": [0, 3, 2, 3, 3]}
        out = analyze(speakers, labels, ["d"])
        self.assertEqual(out["dialogues"], 1)
        self.assertEqual(out["utterances_with_both"], 2)
        self.assertEqual(
            out["own_emotion_change_after_intervening_other"]["rate"], 1.0
        )
        self.assertEqual(
            out["own_emotion_change_without_intervening_other"]["rate"], 0.0
        )

    def test_one_hot_speakers(self):
        self.assertEqual(
            speaker_ids_for_dialogue([[0, 1, 0], [1, 0, 0], [0, 1, 0]]),
            [1, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()

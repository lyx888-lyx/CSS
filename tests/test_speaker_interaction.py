"""CPU tests for optional speaker-history fusion."""
import unittest

import torch

from research.speaker_interaction import SpeakerHistoryFusion


class TestSpeakerHistoryFusion(unittest.TestCase):
    def test_shape_no_history_and_padding(self):
        torch.manual_seed(7)
        fusion = SpeakerHistoryFusion(8)
        x = torch.randn(2, 4, 8)
        speakers = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])
        valid = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        y = fusion(x, speakers, valid)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())
        # First real utterance has no past, so fusion is exactly identity.
        self.assertTrue(torch.equal(y[:, 0], x[:, 0]))
        self.assertTrue(torch.equal(y[1, 1:], x[1, 1:]))

    def test_no_future_access_with_fixed_features(self):
        torch.manual_seed(1)
        fusion = SpeakerHistoryFusion(8)
        x = torch.randn(1, 4, 8)
        speaker = torch.tensor([[0, 1, 0, 1]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        first = fusion(x, speaker, valid)
        x2 = x.clone()
        x2[:, 3] += 100
        second = fusion(x2, speaker, valid)
        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-6))
        # Only for THIS MODULE. Upstream CSS uses bidirectional attention.

    def test_backward_and_ablation_modes(self):
        for mode in ("both", "self", "other"):
            with self.subTest(mode=mode):
                fusion = SpeakerHistoryFusion(8, mode=mode, use_decay=False)
                x = torch.randn(1, 4, 8, requires_grad=True)
                y = fusion(
                    x, torch.tensor([[0, 1, 0, 1]]),
                    torch.ones(1, 4, dtype=torch.bool)
                )
                y.sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all().item())
                self.assertTrue(torch.isfinite(fusion.query.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()

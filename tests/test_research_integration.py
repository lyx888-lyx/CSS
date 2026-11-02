"""Model-level smoke tests for the opt-in CSS speaker history module."""
import unittest

import torch

from model import Transformer_Based_Model


def toy_model(mode=None):
    return Transformer_Based_Model(
        "MELD", 4, 12, 8, 10, 4, 2, 2,
        n_classes=7, hidden_dim=16, n_speakers=9,
        dropout=0.0, speaker_history_mode=mode
    )


class TestOptInIntegration(unittest.TestCase):
    def test_original_state_dict_is_compatible_when_disabled(self):
        torch.manual_seed(99)
        before = toy_model()
        after = toy_model(mode=None)
        after.load_state_dict(before.state_dict(), strict=True)
        self.assertIsNone(after.speaker_history)

    def test_speaker_history_full_forward_and_gradient(self):
        torch.manual_seed(11)
        model = toy_model(mode="both")
        x = torch.randn(4, 2, 12)
        v = torch.randn(4, 2, 8)
        a = torch.randn(4, 2, 10)
        valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
        speakers = torch.zeros(2, 4, 9)
        speakers[0, 0, 0] = 1
        speakers[0, 1, 1] = 1
        speakers[0, 2, 0] = 1
        speakers[0, 3, 2] = 1
        speakers[1, :, 1] = 1
        out = model(x, v, a, valid, speakers, [4, 2])
        self.assertEqual(out[4].shape, (2, 4, 7))
        out[3].sum().backward()
        self.assertIsNotNone(model.speaker_history.gate.weight.grad)
        self.assertTrue(torch.isfinite(model.speaker_history.gate.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()

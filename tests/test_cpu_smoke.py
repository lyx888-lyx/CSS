"""Fast CPU-only checks for CSS forward pass and device handling.

Run with: python -m unittest discover -s tests -v
"""
import unittest

import torch

from model import PGM, Transformer_Based_Model


class TestCpuSmoke(unittest.TestCase):
    def test_model_forward_on_cpu(self):
        torch.manual_seed(42)
        length, batch = 4, 2
        for name, speakers, classes in [("IEMOCAP", 2, 6), ("MELD", 9, 7)]:
            with self.subTest(dataset=name):
                model = Transformer_Based_Model(
                    name, 1, 12, 8, 10, 4, 2, 2,
                    n_classes=classes, hidden_dim=16,
                    n_speakers=speakers, dropout=0.0,
                ).cpu().eval()
                text = torch.randn(length, batch, 12)
                visual = torch.randn(length, batch, 8)
                audio = torch.randn(length, batch, 10)
                mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
                qmask = torch.zeros(batch, length, speakers)
                qmask[:, :, 0] = 1
                with torch.no_grad():
                    outputs = model(text, visual, audio, mask, qmask, [2, 4])
                self.assertEqual(len(outputs), 9)
                self.assertEqual(tuple(outputs[4].shape), (batch, length, classes))
                self.assertEqual(outputs[4].device.type, "cpu")
                self.assertTrue(torch.isfinite(outputs[4]).all().item())

    def test_pgm_projection_on_cpu(self):
        pgm = PGM(None, None)
        weights = pgm._project([
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ])
        self.assertEqual(weights.device.type, "cpu")
        self.assertTrue(torch.all(weights >= 0).item())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

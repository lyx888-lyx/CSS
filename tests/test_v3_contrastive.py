"""CPU-only checks for CSS V3 class-balanced train-only contrastive loss."""
import unittest

import torch

from research.v3_contrastive import class_balanced_supcon


class TestV3ClassBalancedSupCon(unittest.TestCase):
    def test_finite_gradients_valid_real_turns(self):
        torch.manual_seed(22)
        emb = torch.randn(2, 4, 16, requires_grad=True)
        labels = torch.tensor([[0, 1, 0, 1], [2, 2, 1, 0]])
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss, n = class_balanced_supcon(emb, labels, mask, temperature=0.2)
        self.assertEqual(n, 8)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0)
        loss.backward()
        self.assertIsNotNone(emb.grad)
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertGreater(float(emb.grad.abs().sum().item()), 0)

    def test_padding_is_excluded_and_has_zero_gradient(self):
        torch.manual_seed(11)
        emb = torch.randn(1, 4, 6, requires_grad=True)
        labels = torch.tensor([[0, 0, 1, 0]])
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        loss, n = class_balanced_supcon(emb, labels, mask)
        self.assertEqual(n, 2)
        loss.backward()
        self.assertEqual(float(emb.grad[0, 2:].abs().sum()), 0.0)

    def test_unique_label_batch_returns_differentiable_zero(self):
        emb = torch.randn(1, 3, 5, requires_grad=True)
        labels = torch.tensor([[0, 1, 2]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss, n = class_balanced_supcon(emb, labels, mask)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(float(loss.item()), 0)
        loss.backward()
        self.assertIsNotNone(emb.grad)

    def test_one_valid_turn_returns_differentiable_zero(self):
        emb = torch.randn(1, 2, 5, requires_grad=True)
        y = torch.tensor([[0, 0]])
        m = torch.tensor([[1, 0]], dtype=torch.bool)
        loss, n = class_balanced_supcon(emb, y, m)
        self.assertEqual(n, 0)
        loss.backward()
        self.assertIsNotNone(emb.grad)

    def test_no_future_gold_labels_used_at_inference(self):
        # This objective accepts labels, while model.forward does not.
        # Replacing masked padded labels must not affect the loss.
        emb = torch.randn(1, 4, 8)
        y = torch.tensor([[0, 0, 2, 4]])
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        first, n1 = class_balanced_supcon(emb, y, mask)
        second, n2 = class_balanced_supcon(
            emb, torch.tensor([[0, 0, 5, 6]]), mask
        )
        self.assertEqual(n1, n2)
        torch.testing.assert_close(first, second)

    def test_invalid_shapes_and_temperature(self):
        emb = torch.randn(1, 3, 4)
        y = torch.ones(1, 3, dtype=torch.long)
        mask = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            class_balanced_supcon(emb, y, mask, temperature=0)
        with self.assertRaises(ValueError):
            class_balanced_supcon(emb, y[:, :2], mask)


if __name__ == "__main__":
    unittest.main()

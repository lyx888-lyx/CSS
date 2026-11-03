"""CPU regression tests for the V1.1 TRAIN-ONLY intervention loss."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from research.precontext_shift import PreContextSpeakerShift
from research.v11_intervention import (
    masked_categorical_kl,
    measure_auxiliary_gradient_ratio,
    stable_after_other_training_mask,
    without_v1_other_residual,
    v11_training_consistency_loss,
)


class TinyV1(nn.Module):
    """Tiny end-to-end analogue: full/ablated V1 path with trainable logits."""

    def __init__(self):
        super().__init__()
        self.precontext_shift = nn.ModuleDict({
            name: PreContextSpeakerShift(8, mode="other")
            for name in ("text", "audio", "visual")
        })
        self.classifier = nn.Linear(8, 3)

    def forward(self, text, audio, visual, umask, qmask, lengths):
        speaker_ids = qmask.argmax(-1)
        values = (
            self.precontext_shift["text"](text, speaker_ids, umask.bool())
            + self.precontext_shift["audio"](audio, speaker_ids, umask.bool())
            + self.precontext_shift["visual"](visual, speaker_ids, umask.bool())
        ) / 3
        prob = self.classifier(values).softmax(-1)
        return None, None, None, None, prob


class TestV11Consistency(unittest.TestCase):
    def test_mask_marks_stable_after_other_only(self):
        labels = torch.tensor([[0, 1, 0, 0, 0, 1],
                               [2, 2, 2, 0, 0, 0]])
        speakers = torch.tensor([[0, 1, 0, 0, 1, 0],
                                 [0, 0, 1, 0, 1, 1]])
        valid = torch.tensor([[1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        observed = stable_after_other_training_mask(labels, speakers, valid)
        expected = torch.tensor([[0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(observed, expected))

    def test_zero_other_projection_gives_identity_and_restores_hooks(self):
        torch.manual_seed(11)
        model = TinyV1()
        x = torch.randn(1, 4, 8)
        speakers = torch.tensor([[0, 1, 0, 1]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        layer = model.precontext_shift["text"]
        normal = layer(x, speakers, valid)
        self.assertFalse(torch.equal(normal, x))
        with without_v1_other_residual(model):
            out = layer(x, speakers, valid)
            self.assertTrue(torch.equal(out, x))
        self.assertEqual(len(layer.other_diff._forward_hooks), 0)
        self.assertTrue(torch.equal(layer(x, speakers, valid), normal))

    def test_kl_selects_only_eligible_turns_and_has_grad(self):
        teacher = torch.tensor([[[0.6, 0.4], [0.1, 0.9]]])
        student_logits = torch.tensor([[[0.3, 0.5], [0.9, 0.1]]],
                                     requires_grad=True)
        student = student_logits.softmax(dim=-1)
        mask = torch.tensor([[1, 0]], dtype=torch.bool)
        loss = masked_categorical_kl(teacher, student, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertTrue(torch.equal(
            student_logits.grad[:, 1], torch.zeros_like(student_logits.grad[:, 1])
        ))

    def test_teacher_no_grad_and_no_gold_input_to_model(self):
        torch.manual_seed(24)
        model = TinyV1()
        text, audio, visual = [torch.randn(1, 4, 8) for _ in range(3)]
        mask = torch.ones(1, 4)
        speakers = [0, 1, 0, 1]
        qmask = F.one_hot(torch.tensor([speakers]), num_classes=2).float()
        labels = torch.tensor([[0, 2, 0, 2]])
        inputs = (text, audio, visual, mask, qmask, [4])
        model.train()
        loss, n = v11_training_consistency_loss(
            model, inputs, labels, qmask.argmax(-1), mask.bool()
        )
        self.assertEqual(n, 2)
        self.assertTrue(model.training)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        self.assertFalse(any(
            m.other_diff._forward_hooks
            for m in model.precontext_shift.values()
        ))

    def test_gradient_diagnostic_does_not_modify_parameter_grads(self):
        weight = nn.Parameter(torch.tensor([2.0]))
        baseline = weight.square().sum()  # d/dw = 4
        auxiliary = (weight * 3).sum()    # d/dw = 3
        result = measure_auxiliary_gradient_ratio(
            baseline, auxiliary, 0.5, [weight]
        )
        self.assertAlmostEqual(result["pgm_grad_norm"], 4.0)
        self.assertAlmostEqual(result["raw_kl_grad_norm"], 3.0)
        self.assertAlmostEqual(result["weighted_kl_grad_norm"], 1.5)
        self.assertAlmostEqual(
            result["weighted_to_pgm_grad_ratio"], 0.375
        )
        self.assertIsNone(weight.grad)
        (baseline + 0.5 * auxiliary).backward()
        self.assertAlmostEqual(weight.grad.item(), 5.5)

    def test_no_stable_return_skips_second_forward(self):
        torch.manual_seed(12)
        model = TinyV1()
        text, audio, visual = [torch.randn(1, 3, 8) for _ in range(3)]
        mask = torch.ones(1, 3)
        qmask = F.one_hot(torch.tensor([[0, 1, 0]]), num_classes=2).float()
        labels = torch.tensor([[0, 1, 2]])  # A changed
        inputs = (text, audio, visual, mask, qmask, [3])
        loss, n = v11_training_consistency_loss(
            model, inputs, labels, qmask.argmax(-1), mask.bool()
        )
        self.assertEqual(n, 0)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

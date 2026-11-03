"""V1 pre-context module and CSS integration regression tests (CPU)."""
import unittest

import torch

from model import Transformer_Based_Model
from research.precontext_shift import PreContextSpeakerShift


def small_css(v1=None):
    return Transformer_Based_Model(
        "MELD", 4, 12, 8, 10, 4, 2, 2,
        n_classes=7, hidden_dim=16, n_speakers=9,
        dropout=0.0, precontext_shift_mode=v1
    )


class TestPrecontextShift(unittest.TestCase):
    def test_no_history_identity_and_padding(self):
        torch.manual_seed(13)
        module = PreContextSpeakerShift(8)
        x = torch.randn(2, 4, 8)
        speaker_ids = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])
        valid = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        output = module(x, speaker_ids, valid)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.equal(output[:, 0], x[:, 0]))
        self.assertTrue(torch.equal(output[1, 1:], x[1, 1:]))
        self.assertTrue(torch.isfinite(output).all())

    def test_no_future_input_leakage_in_v1_only(self):
        torch.manual_seed(14)
        module = PreContextSpeakerShift(8)
        x = torch.randn(1, 5, 8)
        speaker_ids = torch.tensor([[0, 1, 0, 1, 0]])
        valid = torch.ones(1, 5, dtype=torch.bool)
        before = module(x, speaker_ids, valid)
        x_changed = x.clone()
        x_changed[:, 4, :] += 100.0
        after = module(x_changed, speaker_ids, valid)
        self.assertTrue(torch.allclose(before[:, :4], after[:, :4], atol=1e-6))
        # CSS layers after V1 are still bidirectional.

    def test_effect_and_gradient_in_each_mode(self):
        for mode in ("both", "self", "other"):
            with self.subTest(mode=mode):
                torch.manual_seed(15)
                module = PreContextSpeakerShift(8, mode=mode)
                x = torch.randn(1, 5, 8, requires_grad=True)
                sp = torch.tensor([[0, 1, 0, 1, 0]])
                valid = torch.ones(1, 5, dtype=torch.bool)
                y = module(x, sp, valid)
                self.assertTrue(torch.isfinite(y).all())
                self.assertFalse(torch.allclose(x[:, 2:], y[:, 2:]))
                y.sum().backward()
                self.assertIsNotNone(module.query.weight.grad)
                self.assertTrue(torch.isfinite(module.query.weight.grad).all())
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_css_shared_weights_match_at_seeded_init(self):
        torch.manual_seed(68)
        css = small_css()
        torch.manual_seed(68)
        v1 = small_css(v1="both")
        css_state = css.state_dict()
        for name, tensor in css_state.items():
            with self.subTest(name=name):
                self.assertTrue(torch.equal(tensor, v1.state_dict()[name]))
        self.assertIsNone(css.precontext_shift)
        self.assertEqual(set(v1.precontext_shift.keys()),
                         {"text", "audio", "visual"})

    def test_v1_css_end_to_end_forward_gradient(self):
        torch.manual_seed(89)
        model = small_css(v1="both")
        t = torch.randn(4, 2, 12)
        v = torch.randn(4, 2, 8)
        a = torch.randn(4, 2, 10)
        valid = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32
        )
        sp = torch.zeros(2, 4, 9)
        sp[0, 0, 0] = 1
        sp[0, 1, 1] = 1
        sp[0, 2, 0] = 1
        sp[0, 3, 2] = 1
        sp[1, :, 1] = 1
        out = model(t, v, a, valid, sp, [4, 2])
        self.assertEqual(out[4].shape, (2, 4, 7))
        out[3].sum().backward()
        self.assertIsNotNone(model.precontext_shift["text"].query.weight.grad)
        self.assertIsNotNone(model.precontext_shift["audio"].query.weight.grad)
        self.assertIsNotNone(model.precontext_shift["visual"].query.weight.grad)


if __name__ == "__main__":
    unittest.main()

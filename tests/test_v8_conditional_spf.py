"""No-data CPU tests for V8 conditional SPF pilot."""
import unittest

import torch

from model import SPFusion, Transformer_Based_Model
from research.conditional_spf_gain import (
    CrossSpeakerSPFGain, previous_different_speaker_indices,
    seven_context_features
)


class TestV8SpeakerContext(unittest.TestCase):
    def test_preceding_other_speaker_no_future_or_padding(self):
        speakers = torch.tensor([
            [0, 0, 1, 2, 1, 0, 9],
            [1, 1, 1, 9, 9, 9, 9],
        ])
        valid = torch.tensor([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ]).bool()
        idx = previous_different_speaker_indices(speakers, valid)
        self.assertEqual(idx[0].tolist(), [-1, -1, 1, 2, 3, 4, -1])
        self.assertEqual(idx[1].tolist(), [-1, -1, -1, -1, -1, -1, -1])

    def test_no_future_speaker_turns_change_earlier_gate_features(self):
        torch.manual_seed(113)
        t, a, v = [torch.randn(1, 5, 6) for _ in range(3)]
        speakers = torch.tensor([[0, 1, 0, 1, 0]])
        valid = torch.ones(1, 5, dtype=torch.bool)
        original = seven_context_features(t, a, v, speakers, valid)
        t[:, 3:] += 100
        a[:, 3:] += 100
        v[:, 3:] += 100
        after = seven_context_features(t, a, v, speakers, valid)
        torch.testing.assert_close(original[:, :3], after[:, :3])

    def test_exact_gain4_at_init_all_modes(self):
        torch.manual_seed(114)
        t, a, v = [torch.randn(2, 4, 8) for _ in range(3)]
        speakers = torch.tensor([[0, 1, 0, 2], [0, 0, 9, 9]])
        valid = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]).bool()
        for mode in ("current", "other"):
            gate = CrossSpeakerSPFGain(mode)
            gain = gate(t, a, v, speakers, valid)
            self.assertEqual(gain.shape, (2, 4, 1))
            torch.testing.assert_close(gain[valid], torch.full(
                (int(valid.sum()), 1), 4.0
            ))
            torch.testing.assert_close(gain[~valid], torch.ones(
                (int((~valid).sum()), 1)
            ))

    def test_current_gate_strictly_ignores_other_speaker_features(self):
        torch.manual_seed(115)
        t, a, v = [torch.randn(1, 4, 6) for _ in range(3)]
        valid = torch.ones(1, 4, dtype=torch.bool)
        gate = CrossSpeakerSPFGain("current")
        with torch.no_grad():
            gate.proj.weight.fill_(1.0)
        ids1 = torch.tensor([[0, 1, 0, 1]])
        ids2 = torch.tensor([[0, 0, 1, 1]])
        out1 = gate(t, a, v, ids1, valid)
        out2 = gate(t, a, v, ids2, valid)
        torch.testing.assert_close(out1, out2)

    def test_other_gate_uses_history_and_gradient_reaches_gate(self):
        torch.manual_seed(116)
        t, a, v = [torch.randn(1, 4, 8) for _ in range(3)]
        ids = torch.tensor([[0, 1, 0, 1]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        gate = CrossSpeakerSPFGain("other")
        spf = SPFusion(input_dim=8, rank=4, poly_order=3, output_dim=8)
        s = gate(t, a, v, ids, valid)
        fused = spf(t, a, v, polynomial_scale=s)
        loss = fused.square().mean()
        loss.backward()
        self.assertIsNotNone(gate.proj.weight.grad)
        self.assertGreater(float(gate.proj.weight.grad.abs().sum()), 0)
        self.assertTrue(bool((s >= 2).all().item()))
        self.assertTrue(bool((s <= 6).all().item()))

    def test_original_css_initialization_unchanged_with_gate_appended(self):
        kw = dict(
            dataset="MELD", temp=4, D_text=8, D_visual=6, D_audio=7,
            n_head=2, rank=4, poly_order=3, n_classes=7,
            hidden_dim=16, n_speakers=9, dropout=0.1
        )
        torch.manual_seed(117)
        original = Transformer_Based_Model(**kw, v8_spf_mode="off")
        torch.manual_seed(117)
        improved = Transformer_Based_Model(**kw, v8_spf_mode="other")
        for key, old_tensor in original.state_dict().items():
            self.assertIn(key, improved.state_dict())
            torch.testing.assert_close(old_tensor, improved.state_dict()[key])
        self.assertTrue(any(
            "v8_spf_gain" in key for key in improved.state_dict()
        ))

    def test_spf_gain4_exact_polynomial_addition(self):
        torch.manual_seed(118)
        spf = SPFusion(input_dim=8, rank=4, poly_order=3, output_dim=8)
        t, a, v = [torch.randn(1, 3, 8) for _ in range(3)]
        from research.spf_diagnostics import decompose_spfusion
        original = decompose_spfusion(spf, t, a, v)
        output = spf(t, a, v, polynomial_scale=torch.full((1, 3, 1), 4.0))
        torch.testing.assert_close(
            output, 4.0*original["polynomial"] + original["residual"],
            rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            spf(t,a,v), original["reconstructed"],
            rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()

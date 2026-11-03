"""CPU-only tests of V2 causal ordering, other-speaker separation and data shapes."""
import unittest

import torch
from torch.nn import functional as F

from research.v2_state_model import MODES, V2SpeakerStateProbe


def make_batch(speakers=(0, 1, 0, 1), text_dim=8, audio_dim=5, visual_dim=6):
    torch.manual_seed(19)
    length = len(speakers)
    text = torch.randn(length, 1, text_dim)
    visual = torch.randn(length, 1, visual_dim)
    audio = torch.randn(length, 1, audio_dim)
    speaker = torch.tensor(speakers).view(length, 1)
    qmask = F.one_hot(speaker, num_classes=3).float()
    umask = torch.ones(1, length)
    return (text, visual, audio, qmask, umask)


def new_model(mode):
    return V2SpeakerStateProbe(
        mode=mode, hidden_dim=12, text_dim=8, audio_dim=5,
        visual_dim=6, classes=7, speakers=3, dropout=0.0,
    ).eval()


class TestV2SpeakerState(unittest.TestCase):
    def test_all_modes_shape_and_backprop(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = new_model(mode).train()
                data = make_batch()
                logits = model(*data)
                self.assertEqual(tuple(logits.shape), (1, 4, 7))
                self.assertTrue(torch.isfinite(logits).all())
                label = torch.tensor([[1, 2, 3, 4]])
                loss = F.cross_entropy(logits.reshape(-1, 7), label.reshape(-1))
                loss.backward()
                self.assertTrue(any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ))

    def test_future_utterances_cannot_change_past_predictions(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(9)
                model = new_model(mode)
                batch = make_batch()
                full = model(*batch)
                prefix = tuple(x[:2] for x in batch[:4]) + (batch[4][:, :2],)
                short = model(*prefix)
                torch.testing.assert_close(full[:, :2], short, rtol=1e-5, atol=1e-6)

    def test_other_speaker_features_do_not_affect_own_only_arms(self):
        # Timeline: A(0), B(1), A(2). Perturb ONLY B's text.
        # Own-only history modes must have identical A(2) prediction.
        for mode in ("current", "last", "ema", "adaptive"):
            with self.subTest(mode=mode):
                model = new_model(mode)
                data = make_batch((0, 1, 0))
                changed = list(x.clone() for x in data)
                changed[0][1] = changed[0][1] * -5.0
                before = model(*data)[0, 2]
                after = model(*changed)[0, 2]
                torch.testing.assert_close(before, after, rtol=0, atol=1e-7)

    def test_intervening_other_information_reaches_adaptive_other(self):
        model = new_model("adaptive_other")
        with torch.no_grad():
            model.raw_other_gain.fill_(3.0)
            model.other_projection.weight.copy_(
                0.5 * torch.eye(model.hidden_dim)
            )
        data = make_batch((0, 1, 0))
        changed = list(x.clone() for x in data)
        changed[0][1] = changed[0][1] * -7.0
        before = model(*data)[0, 2]
        after = model(*changed)[0, 2]
        self.assertGreater((before - after).abs().max().item(), 1e-7)

    def test_cross_history_resets_after_speaker_turn(self):
        model = new_model("adaptive_other")
        data = make_batch((0, 1, 0, 0))
        _, info = model(*data, return_diagnostics=True)
        self.assertEqual(info["has_own_history"].tolist(),
                         [[False, False, True, True]])
        self.assertEqual(info["has_other_since_last"].tolist(),
                         [[False, True, True, False]])

    def test_padding_does_not_influence_prefix_or_history(self):
        model = new_model("adaptive_other")
        text, visual, audio, qmask, umask = make_batch((0, 1, 0, 1))
        full_umask = umask.clone()
        full_umask[:, 2:] = 0
        padded = model(text, visual, audio, qmask, full_umask)
        trimmed = model(text[:2], visual[:2], audio[:2],
                        qmask[:2], umask[:, :2])
        torch.testing.assert_close(padded[:, :2], trimmed, rtol=1e-5, atol=1e-6)

    def test_validates_mode_and_shapes(self):
        with self.assertRaises(ValueError):
            new_model("unknown")
        model = new_model("adaptive")
        t, v, a, q, u = make_batch()
        with self.assertRaises(ValueError):
            model(t, v, a, q[:, :, :2], u)


if __name__ == "__main__":
    unittest.main()

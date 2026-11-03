"""Synthetic tests for V2 read-only matched-validation path diagnosis."""
import unittest

import torch

from research.v2_state_model import V2SpeakerStateProbe
from scripts.compare_validation import group_labels
from scripts.diagnose_v2_state_paths import (
    compare_predictions, same_protocol,
    temporarily_remove_external_vector,
)


def fake_checkpoint(mode, inds=(2, 5)):
    return {
        "config": {
            "mode": mode, "seed": 10068, "split_seed": 10068,
            "batch_size": 32, "hidden_dim": 64, "dropout": 0.15,
            "epochs": 12, "lr": 0.001, "weight_decay": 0.0001,
            "valid_ratio": 0.1, "dataset": "MELD",
            "validation_dialogues": 115, "max_train_batches": 0,
        },
        "valid_dialogue_indices": list(inds),
        "selection": "validation_weighted_f1",
    }


class TestV2PairedPaths(unittest.TestCase):
    def test_aligned_protocol(self):
        modes = ("current", "ema", "adaptive", "adaptive_other")
        ckpts = {m: fake_checkpoint(m) for m in modes}
        self.assertEqual(same_protocol(ckpts), [2, 5])
        ckpts["ema"]["valid_dialogue_indices"] = [2, 7]
        with self.assertRaisesRegex(ValueError, "different validation"):
            same_protocol(ckpts)

    def test_rejects_smoke_selection(self):
        modes = ("current", "ema", "adaptive", "adaptive_other")
        ckpts = {m: fake_checkpoint(m) for m in modes}
        ckpts["current"]["config"]["max_train_batches"] = 2
        with self.assertRaisesRegex(ValueError, "complete training"):
            same_protocol(ckpts)

    def test_projection_hook_zeroes_output_and_restores(self):
        torch.manual_seed(33)
        model = V2SpeakerStateProbe(
            mode="adaptive_other", hidden_dim=8,
            text_dim=4, visual_dim=3, audio_dim=2,
            speakers=3, dropout=0.0
        )
        x = torch.randn(2, 8)
        normal = model.other_projection(x)
        self.assertGreater(normal.abs().sum().item(), 0)
        with temporarily_remove_external_vector(model):
            torch.testing.assert_close(
                model.other_projection(x), torch.zeros_like(normal)
            )
        torch.testing.assert_close(model.other_projection(x), normal)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with temporarily_remove_external_vector(model):
                raise RuntimeError("deliberate")
        self.assertEqual(len(model.other_projection._forward_hooks), 0)

    def test_comparison_is_paired(self):
        y = [0, 1, 2, 0, 1]
        speakers = [0, 1, 0, 1, 0]
        tags = group_labels(y, speakers)
        preds = {
            "current": [0, 0, 2, 1, 1],
            "ema": [0, 1, 2, 1, 1],
            "adaptive": [0, 0, 0, 1, 1],
            "adaptive_other": [0, 1, 2, 0, 1],
            "adaptive_other_zero_external_vector": [0, 0, 2, 0, 1],
        }
        result = compare_predictions(y, tags, preds)
        x = result["paired"]["current -> ema"]["all"]
        self.assertEqual(x["fixed"], 1)
        self.assertEqual(x["broken"], 0)
        self.assertEqual(x["net_correct_change"], 1)
        ablated = result["paired"][
            "adaptive_other -> adaptive_other_zero_external_vector"
        ]["all"]
        self.assertEqual(ablated["fixed"], 0)
        self.assertEqual(ablated["broken"], 1)

    def test_rejects_wrong_mode(self):
        model = V2SpeakerStateProbe(mode="ema", hidden_dim=8,
                                    text_dim=4, visual_dim=3,
                                    audio_dim=2, speakers=3)
        with self.assertRaises(ValueError):
            with temporarily_remove_external_vector(model):
                pass


if __name__ == "__main__":
    unittest.main()

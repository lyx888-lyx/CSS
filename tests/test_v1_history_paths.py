"""Synthetic CPU unit tests for inference-only self/other projection ablation.

No dataset, checkpoint or official test utterances needed.
"""
import unittest

import torch
from torch import nn

from research.precontext_shift import PreContextSpeakerShift
from scripts.diagnose_v1_history_paths import (
    temporarily_ablate_history_paths, validate_checkpoints
)


class ToyV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.precontext_shift = nn.ModuleDict({
            "text": PreContextSpeakerShift(8),
            "audio": PreContextSpeakerShift(8),
            "visual": PreContextSpeakerShift(8),
        })


class TestHistoryPathIntervention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = ToyV1().eval()
        self.x = torch.randn(2, 5, 8)
        self.speakers = torch.tensor([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ])
        self.valid = torch.ones(2, 5, dtype=torch.bool)

    @torch.no_grad()
    def test_neither_branch_means_identity(self):
        for layer in self.model.precontext_shift.values():
            with temporarily_ablate_history_paths(self.model, "neither"):
                actual = layer(self.x, self.speakers, self.valid)
                self.assertTrue(torch.equal(actual, self.x))
            # Hooks are removed, and output reverts to original V1.
            output = layer(self.x, self.speakers, self.valid)
            self.assertFalse(torch.equal(output, self.x))

    @torch.no_grad()
    def test_each_branch_can_be_removed_independently(self):
        layer = self.model.precontext_shift["text"]
        baseline = layer(self.x, self.speakers, self.valid)
        with temporarily_ablate_history_paths(self.model, "self"):
            self_only = layer(self.x, self.speakers, self.valid)
        with temporarily_ablate_history_paths(self.model, "other"):
            other_only = layer(self.x, self.speakers, self.valid)
        with temporarily_ablate_history_paths(self.model, "both"):
            both = layer(self.x, self.speakers, self.valid)
        self.assertTrue(torch.equal(baseline, both))
        self.assertFalse(torch.allclose(baseline, self_only))
        self.assertFalse(torch.allclose(baseline, other_only))
        self.assertFalse(torch.allclose(self_only, other_only))

    def test_hooks_removed_even_on_exception(self):
        layer = self.model.precontext_shift["text"]
        self.assertEqual(len(layer.self_diff._forward_hooks), 0)
        self.assertEqual(len(layer.other_diff._forward_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with temporarily_ablate_history_paths(self.model, "neither"):
                self.assertEqual(len(layer.self_diff._forward_hooks), 1)
                self.assertEqual(len(layer.other_diff._forward_hooks), 1)
                raise RuntimeError("intentional")
        self.assertEqual(len(layer.self_diff._forward_hooks), 0)
        self.assertEqual(len(layer.other_diff._forward_hooks), 0)

    def test_reject_invalid_mode_and_model(self):
        with self.assertRaises(ValueError):
            with temporarily_ablate_history_paths(self.model, "bad"):
                pass
        with self.assertRaises(ValueError):
            with temporarily_ablate_history_paths(nn.Linear(8, 8), "self"):
                pass

    def test_reject_inconsistent_checkpoint_metadata(self):
        cfg = {
            "dataset": "MELD", "seed": 10068, "split_seed": 10068,
            "classes": 7, "batch_size": 32, "hidden_dim": 1024,
            "rank": 16, "heads": 8, "order": 4, "temp": 4,
            "lr": 5e-6, "valid_ratio": 0.1, "dropout": 0.5,
            "speaker_mode": "off", "precontext_mode": "off",
        }
        a = {
            "config": cfg, "valid_dialogue_indices": [1, 2],
            "selection": "validation_weighted_f1",
        }
        b = {
            "config": {**cfg, "precontext_mode": "both"},
            "valid_dialogue_indices": [1, 2],
            "selection": "validation_weighted_f1",
        }
        validate_checkpoints(a, b)
        b["valid_dialogue_indices"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "indices differ"):
            validate_checkpoints(a, b)


if __name__ == "__main__":
    unittest.main()

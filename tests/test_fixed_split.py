"""Unit tests for fixed dialogue splits and independent training seeds."""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_research as research


class ToyDataset:
    def __init__(self, *args, **kwargs):
        self.rows = list(range(40))
        self.collate_fn = lambda items: items

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class TestFixedSplit(unittest.TestCase):
    def test_existing_meld_seed_reproduces_first_split(self):
        train, valid = research.dialogue_split_indices(1152, 0.1, 10068)
        self.assertEqual(len(train), 1037)
        self.assertEqual(len(valid), 115)
        expected = np.random.default_rng(10068).permutation(1152)
        self.assertEqual(valid, sorted(expected[:115].tolist()))
        self.assertEqual(set(train) & set(valid), set())
        self.assertEqual(len(set(train + valid)), 1152)

    def test_split_does_not_change_with_training_seed(self):
        with patch.object(research, "MELDDataset", ToyDataset):
            a = research.get_loaders(
                "MELD", batch_size=4, ratio=0.2,
                split_seed=10068, train_seed=10069
            )
            b = research.get_loaders(
                "MELD", batch_size=4, ratio=0.2,
                split_seed=10068, train_seed=10070
            )
            self.assertEqual(a[3], b[3])
            self.assertEqual(a[4], b[4])
            self.assertEqual(len(a[4]), 8)
            self.assertNotEqual(
                list(iter(a[0].sampler)),
                list(iter(b[0].sampler)),
                "different training seeds should shuffle dialogue order differently"
            )

    def test_split_seed_is_separate_in_configuration(self):
        args = argparse.Namespace(
            dataset="MELD", seed=10069, split_seed=10068, epochs=None,
            lr=None, order=None, temp=None, batch_size=32,
            hidden_dim=1024, heads=8, rank=16, dropout=0.5,
            l2=1e-5, valid_ratio=0.1, speaker_mode="off",
            no_time_decay=False, max_lag=None
        )
        config = research.make_config(args)
        self.assertEqual(config["seed"], 10069)
        self.assertEqual(config["split_seed"], 10068)
        args.split_seed = None
        self.assertEqual(research.make_config(args)["split_seed"], 10069)


if __name__ == "__main__":
    unittest.main()

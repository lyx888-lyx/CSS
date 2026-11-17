"""Synthetic disjoint held-out dialogue split checks for V5.2."""
import unittest

from scripts.train_research import (
    dialogue_split_indices, v52_disjoint_dialogue_split_indices
)


class TestV52DisjointSplit(unittest.TestCase):
    def test_new_val_never_reuses_old_115_dialogues(self):
        train, val, old_val = v52_disjoint_dialogue_split_indices(
            1152, 0.1, split_seed=24117
        )
        self.assertEqual((len(train), len(val), len(old_val)),
                         (933, 104, 115))
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(val) & set(old_val))
        self.assertFalse(set(train) & set(old_val))
        self.assertEqual(len(set(train) | set(val) | set(old_val)), 1152)

    def test_fixed_new_seed_reproducible(self):
        x = v52_disjoint_dialogue_split_indices(1152, 0.1, 24117)
        y = v52_disjoint_dialogue_split_indices(1152, 0.1, 24117)
        self.assertEqual(x, y)

    def test_not_identical_to_legacy_valid(self):
        _, legacy_valid = dialogue_split_indices(1152, 0.1, 10068)
        _, fresh_valid, old_valid = v52_disjoint_dialogue_split_indices(
            1152, 0.1, 24117
        )
        self.assertEqual(old_valid, legacy_valid)
        self.assertNotEqual(fresh_valid, legacy_valid)


if __name__ == "__main__":
    unittest.main()

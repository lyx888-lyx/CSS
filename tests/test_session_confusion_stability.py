"""CPU-only synthetic tests. Never load IEMOCAP features or official TEST."""
import unittest

import numpy as np

from research.session_confusion_stability import (
    SESSIONS, EXPECTED_DIALOGUES, split_by_session,
    fold_indices, summarize_fold, pair_stats, aggregate_folds,
    validate_train_only,
)


class TestSessionConfusionStability(unittest.TestCase):
    def fake_keys(self):
        return [f"{s}M_impro{j:02d}" for s in SESSIONS
                for j in range(EXPECTED_DIALOGUES[s])]

    def test_exact_train_split_and_no_session5(self):
        keys = self.fake_keys()
        groups = split_by_session(keys)
        self.assertEqual([len(groups[s]) for s in SESSIONS], [28, 30, 32, 30])
        for s in SESSIONS:
            a, b = fold_indices(groups, s)
            self.assertEqual(len(a) + len(b), 120)
            self.assertFalse(set(a) & set(b))
        with self.assertRaises(ValueError):
            split_by_session(keys + ["Ses05M_test00"])
        with self.assertRaises(ValueError):
            split_by_session(keys[:-1])
        with self.assertRaises(ValueError):
            split_by_session(keys[:-1] + [keys[0]])

    def test_preflight_train_labels_and_turn_count(self):
        keys = self.fake_keys()
        labels = {k: np.zeros(48, dtype=np.int64) for k in keys}
        # 120*48 + 50 = 5810; the first dialogue gets 50 extra turns.
        labels[keys[0]] = np.zeros(98, dtype=np.int64)
        data = validate_train_only(keys, labels)
        self.assertEqual(data["utterances"], 5810)
        self.assertEqual(data["dialogues_by_session"]["Ses03"], 32)
        labels[keys[0]] = np.zeros(97, dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_train_only(keys, labels)

    def test_direction_rates_and_error_denominators(self):
        y = np.array([3, 3, 3, 5, 5, 4, 0, 0])
        p = np.array([5, 3, 5, 3, 5, 4, 4, 0])
        s = summarize_fold(y, p)
        self.assertEqual(s["errors"], 4)
        self.assertEqual(s["accuracy"], 50.0)
        pair = s["pairs"]["angry<->frustrated"]
        self.assertEqual((pair["a_to_b"], pair["b_to_a"]), (2, 1))
        self.assertAlmostEqual(pair["a_to_b_pct"], 66.6667, places=3)
        self.assertEqual(pair["b_to_a_pct"], 50.0)
        self.assertEqual(pair["pair_errors_per_pair_support_pct"], 60.0)
        self.assertEqual(pair["share_of_all_errors_pct"], 75.0)
        with self.assertRaises(ValueError):
            summarize_fold([0, 1], [0, 6])
        with self.assertRaises(ValueError):
            pair_stats(np.zeros((7, 7)))

    def test_strict_aggregation_and_coverage(self):
        folds = {}
        for s in SESSIONS:
            ids = [f"{s}M_impro{i:02d}" for i in range(EXPECTED_DIALOGUES[s])]
            # 1344 + 20 + 1440 + 1536 + 1470 = 5810 turns.
            lengths = [48 if s != "Ses04" else 49] * len(ids)
            if s == "Ses01":
                lengths[0] += 20
            n = sum(lengths)
            y = np.array([3, 5] * (n // 2) + [3] * (n % 2))
            p = y.copy()
            p[0] = 5
            folds[s] = {
                "held_out_session": s,
                "checkpoint_selection": "fixed_epoch_200_no_heldout_selection",
                "held_out_dialogue_ids": ids,
                "held_out_dialogue_lengths": lengths,
                "labels": y.tolist(), "predictions": p.tolist(),
                "summary": summarize_fold(y, p),
            }
        result = aggregate_folds(folds)
        self.assertEqual(result["pooled_out_of_fold"]["utterances"], 5810)
        self.assertEqual(result["pooled_out_of_fold"]["errors"], 4)
        self.assertEqual(result["pair_stability"]["angry<->frustrated"]["nonzero_a_to_b_folds"], 4)
        folds["Ses02"]["held_out_dialogue_ids"][0] = "Ses05M_fake"
        with self.assertRaises(ValueError):
            aggregate_folds(folds)


if __name__ == "__main__":
    unittest.main()

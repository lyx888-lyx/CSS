"""Synthetic CPU tests for V4 per-class confusion analysis. No MELD data needed."""
import unittest

from scripts.audit_meld_class_confusions import (
    NAMES, summarize_confusion, same_train_validation_split,
)


class DummyMELD:
    def __len__(self):
        return 1152


class TestV4MELDClassConfusions(unittest.TestCase):
    def test_official_label_mapping(self):
        self.assertEqual(NAMES[0], "Neutral")
        self.assertEqual(NAMES[2], "Fear")
        self.assertEqual(NAMES[5], "Disgust")

    def test_class_precision_recall_and_top_confusions(self):
        y = [2, 2, 2, 5, 5, 0]
        p = [2, 0, 0, 6, 5, 0]
        out = summarize_confusion(y, p)
        fear = out["per_class"]["2"]
        disgust = out["per_class"]["5"]
        self.assertEqual(out["n"], 6)
        self.assertEqual(fear["support"], 3)
        self.assertEqual(fear["correct"], 1)
        self.assertEqual(fear["predicted_count"], 1)
        self.assertAlmostEqual(fear["recall"], 33.33, places=2)
        self.assertEqual(fear["top_confusions"][0]["predicted_name"], "Neutral")
        self.assertEqual(fear["top_confusions"][0]["count"], 2)
        self.assertEqual(disgust["support"], 2)
        self.assertEqual(disgust["correct"], 1)
        self.assertEqual(disgust["top_confusions"][0]["predicted_name"], "Anger")

    def test_all_absent_classes_report_zero_f1(self):
        d = summarize_confusion([0, 0], [0, 0])
        self.assertEqual(d["per_class"]["2"]["support"], 0)
        self.assertEqual(d["per_class"]["5"]["predicted_count"], 0)
        self.assertEqual(d["per_class"]["2"]["f1"], 0)

    def test_no_out_of_range_labels(self):
        for y, p in [([7], [0]), ([0], [-1]), ([0, 1], [0])]:
            with self.subTest(y=y, p=p):
                with self.assertRaises(ValueError):
                    summarize_confusion(y, p)

    def test_no_test_selected_checkpoints(self):
        ckpt = {
            "config": {"dataset": "MELD", "split_seed": 10068},
            "valid_dialogue_indices": [0] * 115,
            "selection": "test_weighted_f1",
        }
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            same_train_validation_split(ckpt, DummyMELD())


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests: frozen V11 official TEST protocol, no TEST dataset read."""
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from research.v11_test_protocol import (
    test_metrics as summarize_test_predictions,
    paired_changes,
    validate_saved_checkpoint,
)


class TestV11TestProtocol(unittest.TestCase):
    @staticmethod
    def checkpoint(arm):
        cfg=dict(
            dataset="IEMOCAP",seed=61078,epochs=200,
            v10_feature_mode="original",v10_validation_session=4,
            speaker_mode="off",precontext_mode="off",
            v61_spf_mode="full",v52_pgm_mode="original",
            v8_spf_mode="off",classes=6,speakers=2,
            v11_acoustic_mode="off" if arm=="original" else arm,
        )
        return dict(
            config=cfg,selection="validation_weighted_f1",
            epoch=161,validation_f1=68.96,
            train_dialogue_indices=[0,2,3],
            valid_dialogue_indices=[1],
            model_state_dict={"dummy":0},
        )

    def test_three_arms_accept_exact_matched_splits(self):
        for arm in ("original","self","reaction"):
            x=validate_saved_checkpoint(
                self.checkpoint(arm),arm,[0,2,3],[1]
            )
            self.assertEqual(x["arm"],arm)

    def test_mismatched_training_or_checkpoint_selection_rejected(self):
        obj=self.checkpoint("reaction")
        for key,wrong in (
            ("selection","test_weighted_f1"),
            ("train_dialogue_indices",[0,1,3]),
            ("valid_dialogue_indices",[2]),
        ):
            changed={**obj,key:wrong}
            with self.assertRaises(ValueError):
                validate_saved_checkpoint(changed,"reaction",[0,2,3],[1])
        swapped=self.checkpoint("self")
        with self.assertRaises(ValueError):
            validate_saved_checkpoint(swapped,"reaction",[0,2,3],[1])

    def test_overall_and_classwise_metrics(self):
        y=np.array([0,0,1,2,4,5,5])
        p=np.array([0,1,1,2,3,0,5])
        result=summarize_test_predictions(y,p)
        self.assertAlmostEqual(result["accuracy"],100*accuracy_score(y,p))
        self.assertAlmostEqual(result["weighted_f1"],
             100*f1_score(y,p,average="weighted",
                          labels=list(range(6)),zero_division=0))
        self.assertAlmostEqual(result["macro_f1"],
             100*f1_score(y,p,average="macro",
                          labels=list(range(6)),zero_division=0))
        self.assertEqual(result["class_metrics"]["5"]["support"],2)
        self.assertAlmostEqual(
            result["class_metrics"]["5"]["accuracy_recall"],50.0
        )

    def test_class_flips_distinct_from_fixed_and_broken(self):
        stats=paired_changes([0,1,2],
                             [0,5,2],
                             [4,3,2])
        self.assertEqual(stats["fixed"],1)
        self.assertEqual(stats["broken"],0)
        self.assertEqual(stats["net_more_correct"],1)
        self.assertEqual(stats["top1_flips"],2)

    def test_invalid_class_index_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            summarize_test_predictions([0,6],[0,1])
        with self.assertRaises(ValueError):
            summarize_test_predictions([0,1],[0])


if __name__=="__main__":
    unittest.main()

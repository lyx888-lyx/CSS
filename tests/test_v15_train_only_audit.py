"""V15 posthoc error-audit synthetic tests. No dataset loading."""
import unittest
import numpy as np

from scripts.audit_v15_train_only import paired_counts, audit


class V15PosthocAuditTests(unittest.TestCase):
    def test_fixed_broken_and_wrong_to_wrong(self):
        y=[0,1,2,3,4,5]
        reference=[1,1,0,3,4,1]
        challenger=[0,2,3,0,4,5]
        got=paired_counts(y,reference,challenger)
        self.assertEqual(got["n"],6)
        self.assertEqual(got["fixed"],2)
        self.assertEqual(got["broken"],2)
        self.assertEqual(got["net_more_correct"],0)
        self.assertEqual(got["raw_prediction_flips"],5)
        self.assertEqual(got["wrong_to_different_wrong"],1)

    def test_subgroup_mask_alignment(self):
        y=[0,1,2,3]
        ref=[1,1,2,3]
        other=[0,0,2,3]
        sub=paired_counts(y,ref,other,[True,True,False,False])
        self.assertEqual(sub["n"],2)
        self.assertEqual(sub["fixed"],1)
        self.assertEqual(sub["broken"],1)
        with self.assertRaises(ValueError):
            paired_counts(y,ref,other,[True])

    def test_reject_nontrain_report(self):
        with self.assertRaises(ValueError):
            audit({"protocol":{"partition":"OFFICIAL TEST",
                               "official_test_scored":True}})
        with self.assertRaises(ValueError):
            audit({"protocol":{"partition":"ORIGINAL TRAIN ONLY (Ses01-Ses04)",
                               "official_test_scored":False,
                               "official_test_loader_created":False,
                               "smoke":False},
                   "seed_results":[]})


if __name__=="__main__":
    unittest.main()

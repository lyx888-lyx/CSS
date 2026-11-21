"""Synthetic regression tests for no-training V11 paired evaluation."""
import unittest

import numpy as np
from sklearn.metrics import f1_score

from research.v11_paired_analysis import (
    GROUPS, bootstrap_dialogues, fixed_broken, summarize_groups, weighted_f1
)


def dialogue(labels, original, self, reaction, after):
    y=np.array(labels,dtype=np.int64)
    after=np.asarray(after,dtype=bool)
    n=len(y)
    masks={
        "all":np.ones(n,dtype=bool),
        "first":np.array([True]+[False]*(n-1)),
        "stable":np.array([False]+[True]*(n-1)),
        "changed":after,
        "stable_after_other":np.zeros(n,dtype=bool),
        "changed_after_other":after,
    }
    return {
        "labels":y,
        "predictions":{
            "original":np.array(original,dtype=np.int64),
            "self":np.array(self,dtype=np.int64),
            "reaction":np.array(reaction,dtype=np.int64)
        },
        "masks":masks,
    }


class TestV11PairedAudit(unittest.TestCase):
    def test_f1_matches_sklearn_weighted(self):
        rng=np.random.default_rng(13)
        for n in (2,20,150):
            y=rng.integers(0,6,size=n)
            pred=rng.integers(0,6,size=n)
            self.assertAlmostEqual(
                weighted_f1(y,pred),
                float(100*f1_score(y,pred,labels=list(range(6)),
                                   average="weighted",zero_division=0)),
                places=9
            )

    def test_f1_perfect_and_empty(self):
        self.assertEqual(weighted_f1(np.array([0,2]),
                                     np.array([0,2])),100.0)
        self.assertIsNone(weighted_f1(np.array([],dtype=np.int64),
                                      np.array([],dtype=np.int64)))
        with self.assertRaises(ValueError):
            weighted_f1(np.array([7]),np.array([0]))

    def test_pairs_count_raw_flips_separately_from_correctness(self):
        # Wrong class -> another wrong class changes top1 without fixed.
        stats=fixed_broken([0,1,2],[0,4,2],[3,5,2])
        self.assertEqual(stats["fixed"],1)
        self.assertEqual(stats["broken"],0)
        self.assertEqual(stats["top1_flips"],2)

    def test_overlapping_true_label_groups_and_support(self):
        ds=[
            dialogue([0,1,2],[0,0,2],[0,1,0],[0,1,2],
                     [False,True,True]),
            dialogue([1,2,0],[1,2,1],[1,1,0],[1,2,0],
                     [False,False,True])
        ]
        res=summarize_groups(ds,("original","self","reaction"))
        self.assertEqual(res["all"]["n"],6)
        self.assertEqual(res["changed_after_other"]["n"],3)
        pair=res["all"]["paired"]["reaction_minus_original"]
        self.assertEqual(pair["fixed"],2)
        self.assertEqual(pair["broken"],0)
        self.assertEqual(pair["top1_flips"],2)
        self.assertEqual(res["changed_after_other"]["paired"]
                         ["reaction_minus_self"]["fixed"],1)

    def test_reproducible_dialogue_bootstrap(self):
        ds=[
            dialogue([0,1,2],[0,0,2],[0,1,0],[0,1,2],
                     [False,True,True]),
            dialogue([1,2,0],[1,2,1],[1,1,0],[1,2,0],
                     [False,False,True]),
            dialogue([0,2,0],[0,0,0],[0,2,1],[0,2,0],
                     [False,True,True]),
        ]
        a=bootstrap_dialogues(
            ds,repetitions=100,seed=31,
            groups=("all","changed_after_other")
        )
        b=bootstrap_dialogues(
            ds,repetitions=100,seed=31,
            groups=("all","changed_after_other")
        )
        self.assertEqual(a,b)
        self.assertEqual(
            a["all"]["reaction_minus_original"]["replicates_with_support"],100
        )


if __name__=="__main__":
    unittest.main()

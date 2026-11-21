"""V15 synthetic CPU tests. Never load IEMOCAP or official TEST."""
import unittest

import numpy as np
import torch

from research.v14_b_incremental_probe import select_partition
from research.v15_conditional_corrector import (
    ConditionalCorrection, ARMS, prepare_fold, seed_all,
    train_a_only, train_correction, infer, run_one_seed, MAX_CORRECTION,
)
from scripts.run_v15_conditional_probe import decide


def synthetic_aba():
    rng = np.random.default_rng(999)
    rows = []
    for session in ("Ses01", "Ses02", "Ses03", "Ses04"):
        for dialogue in range(3):
            for i in range(8):
                rows.append({
                    "current": rng.normal(size=8),
                    "previous": rng.normal(size=8),
                    "other": rng.normal(size=8),
                    "label": (i+dialogue+int(session[-1]))%6,
                    "changed": i%3==0,
                    "session": session,
                    "dialogue": f"{session}_dia{dialogue}",
                    "b_role": "F" if i%2==0 else "M",
                    "gap_bin": 1,
                })
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


class TestV15ConditionalProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = synthetic_aba()
        cls.tr = select_partition(cls.ex, ("Ses01", "Ses02", "Ses03"))
        cls.va = select_partition(cls.ex, ("Ses04",))
        cls.fold = prepare_fold(cls.ex, cls.tr, cls.va, 31)

    def test_exact_shape_and_same_session_donor_partition(self):
        fold = self.fold
        for prefix in ("a","b","sb"):
            self.assertEqual(fold[f"{prefix}_train"].shape[0],len(self.tr))
            self.assertEqual(fold[f"{prefix}_held"].shape[0],len(self.va))
            self.assertEqual(fold[f"{prefix}_train"].shape[1],24)
        self.assertTrue(np.isfinite(fold["sb_held"]).all())

    def test_gold_previous_emotion_not_a_probe_input(self):
        copied = {k: v.copy() for k,v in self.ex.items()}
        copied["label"][:] = (copied["label"]+2)%6
        copied["changed"][:] = ~copied["changed"]
        new = prepare_fold(copied,self.tr,self.va,31)
        for key in ("a_train","a_held","b_train","b_held",
                    "sb_train","sb_held"):
            np.testing.assert_array_equal(self.fold[key],new[key])

    def test_holdout_and_dialogue_leakage_guards(self):
        with self.assertRaises(ValueError):
            prepare_fold(self.ex,np.r_[self.tr,self.va[:1]],self.va,31)
        with self.assertRaises(ValueError):
            prepare_fold(self.ex,self.tr,self.tr[:3],31)

    def test_matched_capacity_and_zero_output_initialization(self):
        a=torch.randn(5,24)
        b=torch.randn(5,24)
        base=torch.randn(5,6)
        params=[]
        for arm in ARMS:
            seed_all(61078)
            mod=ConditionalCorrection(24,24)
            params.append(sum(p.numel() for p in mod.parameters()))
            bi=torch.zeros_like(b) if arm=="self" else b
            logits,delta=mod(a,bi,base)
            torch.testing.assert_close(logits,base)
            torch.testing.assert_close(delta,torch.zeros_like(delta))
        self.assertEqual(len(set(params)),1)

    def test_nonlinear_a_times_b_interaction_and_bounded_change(self):
        seed_all(91)
        mod=ConditionalCorrection(24,24)
        with torch.no_grad():
            mod.adjust.weight.normal_(0,.05)
            mod.adjust.bias.normal_(0,.05)
        a=torch.randn(8,24)
        b=torch.randn(8,24)
        old=torch.randn(8,6)
        out1,delta=mod(a,b,old)
        out2,_=mod(a,torch.zeros_like(b),old)
        self.assertGreater(float((out1-out2).detach().abs().sum()),0)
        uncertainty=1-old.softmax(-1).amax(-1)
        self.assertTrue(bool(torch.all(
            delta.abs().amax(-1) <= MAX_CORRECTION*uncertainty + 1e-6
        )))

    def test_frozen_baseline_while_training_correction(self):
        device=torch.device("cpu")
        net=train_a_only(self.fold,23,device,epochs=2)
        before={k:v.detach().clone() for k,v in net.state_dict().items()}
        corr=train_correction(net,self.fold,"real_b",23,device,epochs=2)
        for k,v in before.items():
            torch.testing.assert_close(net.state_dict()[k],v)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        pred,mag,flip=infer(
            net,corr,self.fold["a_held"],self.fold["b_held"],device
        )
        self.assertEqual(pred.shape,(len(self.va),))
        self.assertLessEqual(mag,MAX_CORRECTION+1e-6)
        self.assertTrue(0<=flip<=1)

    def test_full_synthetic_smoke_all_matched_arms(self):
        preds,diag=run_one_seed(
            self.fold,61078,torch.device("cpu"),smoke=True
        )
        self.assertEqual(set(preds),{
            "a_only","self","real_b","shuffled_b","real_b_replaced"
        })
        for arm,p in preds.items():
            self.assertEqual(p.shape,(len(self.va),),arm)
            self.assertTrue(np.all((p>=0)&(p<6)))
        self.assertEqual(len(diag),4)

    def test_screen_rejects_smoke_or_inadequate_session_gains(self):
        example={
            "pooled_deltas_pp":{
                "real_minus_frozen_a_pp":-1.,
                "real_minus_self_pp":.5,
                "real_minus_shuffled_pp":-.3,
                "real_minus_replaced_pp":1.5,
            },
            "positive_sessions":0,
        }
        self.assertFalse(decide([example],smoke=True)["passed"])
        self.assertFalse(decide([example]*3,smoke=False)["passed"])
        good={
            "pooled_deltas_pp":{k:.6 for k in example["pooled_deltas_pp"]},
            "positive_sessions":3,
        }
        self.assertTrue(decide([good]*3,smoke=False)["passed"])


if __name__=="__main__":
    unittest.main()

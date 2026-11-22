"""CPU unit checks for fixed original CSS EMA/SWA averaging.

Synthetic tiny model only: no IEMOCAP datasets, CUDA, PGM, or TEST.
"""
import unittest

import torch

from research.css_weight_averaging import (
    EpochWeightAverager,frozen_cpu_state,verify_same_state_keys,
    EMA_START,EMA_DECAY,SWA_START,FINAL_EPOCH
)
from scripts.run_css_weight_averaging import validate_full_manifest


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l=torch.nn.Linear(1,1,bias=False)
        self.register_buffer("step",torch.tensor(0,dtype=torch.long))
        self.register_buffer("fixed",torch.tensor([2.0]))


class TestCSSWeightAverage(unittest.TestCase):
    def setUp(self):
        self.model=Tiny()

    def set_weight(self,value,step):
        with torch.no_grad():
            self.model.l.weight.fill_(value)
            self.model.step.fill_(step)

    def test_known_ema_swa_equations_and_integer_buffer(self):
        ema=EpochWeightAverager("ema",2,decay=0.5)
        swa=EpochWeightAverager("swa",2)
        for e,w in [(1,100.),(2,2.),(3,4.),(4,8.)]:
            self.set_weight(w,e)
            ema.update(self.model,e)
            swa.update(self.model,e)
        self.assertEqual(ema.count,3)
        self.assertEqual(swa.count,3)
        self.assertAlmostEqual(float(ema.get_state()["l.weight"][0,0]),5.5)
        # SWA state is float32: 14/3 rounds to 4.6666665, so use an\n        # absolute tolerance appropriate for float32, not 7 decimal places.\n        self.assertAlmostEqual(\n            float(swa.get_state()["l.weight"][0,0]),14/3,delta=1e-6\n        )
        self.assertEqual(int(ema.get_state()["step"]),4)
        self.assertEqual(int(swa.get_state()["step"]),4)
        self.assertAlmostEqual(float(self.model.l.weight),8.0)

    def test_initial_shadow_is_detached_copy_and_not_live_reference(self):
        avg=EpochWeightAverager("ema",1,decay=.9)
        self.set_weight(10.,1)
        avg.update(self.model,1)
        shadow=avg.get_state()
        self.set_weight(20.,2)
        self.assertAlmostEqual(float(shadow["l.weight"][0,0]),10.)
        avg.update(self.model,2)
        self.assertAlmostEqual(float(avg.get_state()["l.weight"][0,0]),11.)
        frozen=frozen_cpu_state(self.model)
        self.set_weight(30.,3)
        self.assertAlmostEqual(float(frozen["l.weight"][0,0]),20.)

    def test_reject_epoch_double_update_and_wrong_decay(self):
        with self.assertRaises(ValueError):
            EpochWeightAverager("ema",1,decay=1.)
        avg=EpochWeightAverager("swa",3)
        with self.assertRaises(RuntimeError):
            avg.get_state()
        avg.update(self.model,1)
        with self.assertRaises(RuntimeError):
            avg.get_state()
        with self.assertRaises(ValueError):
            avg.update(self.model,1)

    def test_all_methods_have_identical_state_specs(self):
        raw=frozen_cpu_state(self.model)
        self.assertTrue(verify_same_state_keys(raw,raw))
        with self.assertRaises(ValueError):
            verify_same_state_keys(raw,{"different":torch.tensor(1.)})
        with self.assertRaises(ValueError):
            verify_same_state_keys()

    def test_fixed_windows_cover_fifty_and_hundred_one_epochs(self):
        self.assertEqual((FINAL_EPOCH-SWA_START+1),50)
        self.assertEqual((FINAL_EPOCH-EMA_START+1),101)
        self.assertEqual(EMA_DECAY,.9)

    def test_locked_manifest_accepts_only_full_no_test_checkpoint_sets(self):
        manifest={
            "protocol":{
                "train_sessions":[1,2,3,4],
                "train_dialogues":120,
                "train_utterances":5810,
                "train_seed":61078,
                "epochs":FINAL_EPOCH,
                "checkpoint_selection":"fixed final epoch, NOT TEST",
                "same_training_trajectory_for_all_arms":True,
                "official_test_scored_during_training":False,
                "smoke":False,
                "ema_start":EMA_START,
                "ema_decay":EMA_DECAY,
                "swa_start":SWA_START,
            },
            "ema":{"updates":101},
            "swa":{"updates":50},
            "checkpoints":{
                "raw_final":{},"ema_final":{},"swa_final":{}
            },
        }
        self.assertTrue(validate_full_manifest(manifest))
        manifest["protocol"]["checkpoint_selection"]="best TEST F1"
        with self.assertRaises(ValueError):
            validate_full_manifest(manifest)


if __name__=="__main__":
    unittest.main()

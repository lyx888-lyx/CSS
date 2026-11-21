"""V12 protocol checks using mock datasets/models, without touching TEST."""
import unittest
from unittest.mock import patch
import torch
from torch import nn

from scripts import run_v12_author_protocol as v12


class ToyCSS(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.encoder=nn.Linear(4,4)
        self.v11_acoustic_mode="off"
        self.v11_acoustic_refiner=None


class ToyDataset:
    def __init__(self,train=True):
        if not train:
            raise AssertionError("TEST partition should not be opened in preflight")
        self.keys=[f"Ses0{s}{'F' if i%2 else 'M'}_impro{i:02d}"
                   for s in range(1,5) for i in range(30)]
        # Make exactly 5810 total train utterances:
        # 119*48 + 98 = 5810.
        self.videoLabels={key:[0]*(48 if i<119 else 98)
                          for i,key in enumerate(self.keys)}


class TestV12Protocol(unittest.TestCase):
    def test_training_partition_includes_all_four_sessions(self):
        with patch.object(v12,"IEMOCAPDataset",ToyDataset):
            info=v12.inspect_train()
        self.assertEqual(info["train_dialogues"],120)
        self.assertEqual(info["train_turns"],5810)
        self.assertEqual(info["train_sessions"],
                         ["Ses01","Ses02","Ses03","Ses04"])

    def test_wrong_session_fails_fast(self):
        class Invalid(ToyDataset):
            def __init__(self,train=True):
                super().__init__(train)
                self.keys[119]="Ses05F_impro30"
                self.videoLabels[self.keys[119]]=[0]*98
        with patch.object(v12,"IEMOCAPDataset",Invalid):
            with self.assertRaises(ValueError):
                v12.inspect_train()

    def test_common_weights_and_rng_equal_among_three_arms(self):
        with patch.object(v12,"Transformer_Based_Model",ToyCSS):
            output={}
            for arm in v12.MODELS:
                torch.manual_seed(61078)
                model=v12.create_model(arm)
                output[arm]=(model.encoder.weight.detach().clone(),
                             torch.rand(5))
                if arm!="original":
                    self.assertIsNotNone(model.v11_acoustic_refiner)
                    self.assertEqual(model.v11_acoustic_mode,arm)
            for arm in ("self","reaction"):
                torch.testing.assert_close(
                    output["original"][0],output[arm][0])
                torch.testing.assert_close(
                    output["original"][1],output[arm][1])

    def test_three_arm_labels_exist(self):
        self.assertEqual(v12.MODELS,("original","self","reaction"))
        self.assertEqual(v12.EPOCHS,200)
        self.assertEqual(v12.SEED,61078)
        self.assertEqual(len(v12.CLASS_WEIGHTS),6)


if __name__=="__main__":
    unittest.main()

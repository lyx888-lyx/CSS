"""CPU-only unit tests for CSS V16 four-head complementarity audit.

No IEMOCAP files, checkpoints, or TEST datasets loaded.
"""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from research.css_head_complementarity import HEADS,summarize
from scripts.audit_css_four_heads import (
    validate_v10_session_split,predict_heads,
)


class FakeIEMOCAPTrain:
    def __init__(self):
        self.keys=[]
        for session,count in (("Ses01",28),("Ses02",30),
                              ("Ses03",32),("Ses04",30)):
            self.keys.extend(f"{session}M_impro{k:02d}" for k in range(count))
        self.videoLabels={}
        # 90 dialogue Ses01-03 with 4298 utterances;
        # 30 dialogue Ses04 with 1512 utterances.
        for i,key in enumerate(self.keys):
            size=(48 if i<68 else 47) if i<90 else (51 if i<102 else 50)
            self.videoLabels[key]=[0]*size

    def __len__(self):
        return len(self.keys)


def saved_v10_object(ds):
    return {
        "config":{
            "dataset":"IEMOCAP","v10_feature_mode":"original",
            "v10_validation_session":4,"speaker_mode":"off",
            "precontext_mode":"off","v8_spf_mode":"off",
            "v61_spf_mode":"full","seed":61078,"classes":6,"speakers":2,
        },
        "selection":"validation_weighted_f1",
        "train_dialogue_indices":list(range(90)),
        "valid_dialogue_indices":list(range(90,120)),
    }


class TestFourHeadAudit(unittest.TestCase):
    def test_preexisting_session4_split_unequal_dialogue_counts(self):
        ds=FakeIEMOCAPTrain()
        cp=saved_v10_object(ds)
        train,valid,nt,nv=validate_v10_session_split(ds,cp)
        self.assertEqual((len(train),len(valid),nt,nv),(90,30,4298,1512))

    def test_reject_same_session_train_and_valid_leakage(self):
        ds=FakeIEMOCAPTrain()
        cp=saved_v10_object(ds)
        cp["valid_dialogue_indices"]=[89]+list(range(91,120))
        cp["train_dialogue_indices"]=list(range(89))+[90]
        with self.assertRaises(ValueError):
            validate_v10_session_split(ds,cp)

    def test_oracle_rescue_vs_wrong_head_risk(self):
        y=np.array([0,1,2,3,4,5])
        pred={
            "text":np.array([0,1,0,3,5,0]),
            "audio":np.array([1,1,2,4,4,0]),
            "visual":np.array([0,0,0,3,5,5]),
            "fused":np.array([1,1,2,3,4,0]),
        }
        m=summarize(y,pred)
        self.assertEqual(m["utterances"],6)
        self.assertEqual(m["fused_error_count"],2)
        self.assertEqual(m["any_single_head_correct_when_fused_wrong"],2)
        self.assertEqual(m["oracle_any_head_correct"],6)
        self.assertEqual(m["fused_correct_and_all_three_single_heads_wrong"],0)
        self.assertEqual(
            m["per_other_head"]["text"]["fixable_fused_errors_by_oracle"],1
        )
        self.assertEqual(
            m["per_other_head"]["text"]["fused_correct_but_head_wrong"],2
        )

    def test_reject_head_shape_and_invalid_probabilities(self):
        y=[0,1,2]
        pred={name:np.array(y) for name in HEADS}
        with self.assertRaises(ValueError):
            summarize(y,{**pred,"text":[0,1]})
        prob={name:np.eye(6)[y].copy() for name in HEADS}
        prob["audio"][0]*=.5
        with self.assertRaises(ValueError):
            summarize(y,pred,prob)

    def test_masked_predictions_from_all_four_heads(self):
        class FakeCSS(torch.nn.Module):
            def forward(self,text,visual,audio,umask,qmask,lengths):
                B,L=text.shape[:2]
                base=torch.zeros(B,L,6,device=text.device)
                logits=[
                    base+torch.arange(6,device=text.device)[None,None,:]*v
                    for v in (0.0,.2,.4,.6)
                ]
                logp=[F.log_softmax(z,-1) for z in logits]
                return tuple(logp+[logp[3].exp()])
        text=torch.zeros(2,3,4)
        visual=torch.zeros(2,3,2)
        audio=torch.zeros(2,3,5)
        qmask=torch.zeros(3,2,2)
        qmask[:,:,0]=1.0
        umask=torch.tensor([[1.,1.,0.],[1.,1.,1.]])
        label=torch.tensor([[0,1,0],[2,3,4]])
        loader=[(text,visual,audio,qmask,umask,label,["a","b"])]
        y,p,prob,nd=predict_heads(FakeCSS(),loader,torch.device("cpu"))
        self.assertEqual(nd,2)
        np.testing.assert_array_equal(y,[0,1,2,3,4])
        for name in HEADS:
            self.assertEqual(p[name].shape,(5,))
            self.assertEqual(prob[name].shape,(5,6))
            np.testing.assert_allclose(prob[name].sum(-1),1,atol=1e-6)


if __name__=="__main__":
    unittest.main()

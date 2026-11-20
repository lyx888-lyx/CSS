"""No-data V11 checks: A-B-A intervals, speaker masks, zero initial change.

Run: python -m unittest discover -s tests -p 'test_v11_acoustic_reaction.py' -v
"""
import unittest

import torch

from model import Transformer_Based_Model
from research.v11_acoustic_reaction import (
    SpeakerRelativeAcousticReaction, acoustic_reaction_features,
)


class TestV11AcousticReaction(unittest.TestCase):
    def test_aba_and_consecutive_self_are_disjoint(self):
        audio=torch.arange(12,dtype=torch.float32).reshape(1,6,2)
        speakers=torch.tensor([[0,0,1,0,1,0]])
        mask=torch.ones((1,6),dtype=torch.bool)
        feat,eligible,prior,counts=acoustic_reaction_features(
            audio,speakers,mask
        )
        self.assertEqual(prior.tolist(),[[-1,0,-1,1,2,3]])
        self.assertEqual(counts.tolist(),[[0,0,0,1,1,1]])
        self.assertEqual(eligible.tolist(),[[False,False,False,True,True,True]])
        self.assertEqual(feat.shape,(1,6,6))
        self.assertTrue(torch.equal(feat[~eligible],torch.zeros_like(feat[~eligible])))

    def test_padding_never_contributes_as_other(self):
        torch.manual_seed(111)
        audio=torch.randn(1,6,4)
        speakers=torch.tensor([[0,1,0,2,2,2]])
        valid=torch.tensor([[1,1,1,0,0,0]],dtype=torch.bool)
        a=acoustic_reaction_features(audio,speakers,valid)
        self.assertEqual(a[1].tolist(),[[False,False,True,False,False,False]])
        audio[:,3:]+=100
        b=acoustic_reaction_features(audio,speakers,valid)
        torch.testing.assert_close(a[0][:,:3],b[0][:,:3])

    def test_intervening_other_affects_reaction_but_not_self_inputs(self):
        torch.manual_seed(112)
        audio=torch.randn(1,5,8)
        speaker=torch.tensor([[0,1,0,1,0]])
        valid=torch.ones(1,5,dtype=torch.bool)
        a=acoustic_reaction_features(audio,speaker,valid)[0]
        changed=audio.clone()
        changed[:,3]+=8
        b=acoustic_reaction_features(changed,speaker,valid)[0]
        torch.testing.assert_close(a[:,4,:8],b[:,4,:8])
        self.assertGreater(float((a[:,4,8:]-b[:,4,8:]).abs().sum()),0)

    def test_zero_init_protects_all_initial_outputs_and_gradients_flow(self):
        torch.manual_seed(113)
        x=torch.randn(2,5,12,requires_grad=True)
        speakers=torch.tensor([[0,1,0,1,0],[0,0,1,0,2]])
        valid=torch.tensor([[1,1,1,1,1],[1,1,1,1,0]],dtype=torch.bool)
        m=SpeakerRelativeAcousticReaction(12,"reaction",bottleneck=16)
        out=m(x,speakers,valid)
        torch.testing.assert_close(out,x)
        out.square().mean().backward()
        self.assertGreater(float(m.project[-1].weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_no_updates_for_first_and_consecutive_same_even_after_training(self):
        torch.manual_seed(114)
        x=torch.randn(1,4,8)
        speakers=torch.tensor([[0,0,1,0]])
        valid=torch.ones((1,4),dtype=torch.bool)
        m=SpeakerRelativeAcousticReaction(8,"reaction",bottleneck=8)
        with torch.no_grad():
            m.project[-1].weight.normal_()
            m.project[-1].bias.fill_(10)
        out=m(x,speakers,valid)
        torch.testing.assert_close(out[:,:3],x[:,:3])
        self.assertGreater(float((out[:,3:]-x[:,3:]).abs().sum()),0)

    def test_same_base_model_seed_and_identical_v11_arm_parameters(self):
        kwargs=dict(dataset="IEMOCAP",temp=1,D_text=1024,
                    D_visual=342,D_audio=1582,n_head=2,rank=4,
                    poly_order=3,n_classes=6,hidden_dim=32,
                    n_speakers=2,dropout=0.1)
        torch.manual_seed(115)
        base=Transformer_Based_Model(**kwargs)
        torch.manual_seed(115)
        self_mode=Transformer_Based_Model(**kwargs,v11_acoustic_mode="self")
        torch.manual_seed(115)
        reaction_mode=Transformer_Based_Model(**kwargs,v11_acoustic_mode="reaction")
        for key,tensor in base.state_dict().items():
            torch.testing.assert_close(tensor,self_mode.state_dict()[key])
            torch.testing.assert_close(tensor,reaction_mode.state_dict()[key])
        for key,tensor in self_mode.v11_acoustic_refiner.state_dict().items():
            torch.testing.assert_close(tensor,
                reaction_mode.v11_acoustic_refiner.state_dict()[key])
        n_self=sum(p.numel() for p in self_mode.v11_acoustic_refiner.parameters())
        n_other=sum(p.numel() for p in reaction_mode.v11_acoustic_refiner.parameters())
        self.assertEqual(n_self,n_other)


if __name__=="__main__":
    unittest.main()

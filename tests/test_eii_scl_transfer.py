"""CPU synthetic tests of EII-SCL equations and the CSS auxiliary hook.

No real IEMOCAP data, checkpoint, or official TEST is loaded.
"""
import math
import unittest

import torch
from torch import nn

from research.eii_scl_transfer import (
    ARMS, ALPHA, pair_masks, dynamic_windows, contrastive_loss,
    SupervisedContrastHook,
)


def speakers(ids):
    return torch.nn.functional.one_hot(torch.tensor(ids),num_classes=2).float()


class TestEIISCLTransfer(unittest.TestCase):
    def test_pair_masks_exclude_padding_self_and_cross_dialogue(self):
        y = torch.tensor([[0,0,1,0],[1,0,1,1]])
        s = torch.tensor([[0,0,1,0],[1,0,1,0]])
        valid = torch.tensor([[1,1,1,0],[1,1,0,0]]).bool()
        e,ss,sy = pair_masks(y,s,valid)
        self.assertEqual(e.shape,(2,4,4))
        self.assertEqual(int(e.sum()),8)
        self.assertFalse(bool(e.diagonal(dim1=1,dim2=2).any()))
        self.assertFalse(bool(e[:,:,3].any()))
        self.assertTrue(bool(ss[0,0,1] and sy[0,0,1]))

    def test_dynamic_windows_identity_uniform_attention(self):
        f = torch.ones(1,6,2)
        y = torch.zeros(1,6,dtype=torch.long)
        s = torch.tensor([[0,1,0,1,0,0]])
        mask = torch.tensor([[1,1,1,1,1,0]]).bool()
        e,ss,_ = pair_masks(y,s,mask)
        windows = dynamic_windows(f,e&ss)
        torch.testing.assert_close(windows,torch.tensor([[3,2,2,2,3,0]]))
        self.assertFalse(windows.requires_grad)

    def test_literal_eii_formula_fixed_window(self):
        f = torch.tensor([[[1.,0.],[1.,0.],[0.,1.]]],requires_grad=True)
        y = torch.tensor([[0,0,1]])
        q = speakers([[0,0,0]])
        m = torch.ones(1,3)
        loss,stats = contrastive_loss(f,y,q,m,"eii_fixed",tau=1.,
                                      fixed_window=1,return_stats=True)
        expected = (math.log1p(math.exp(-1)) +
                    math.log1p(.5*math.exp(-1)))/2
        self.assertAlmostEqual(float(loss.detach()),expected,places=6)
        self.assertEqual(stats["eligible_anchors"],2)
        self.assertEqual(stats["hard_negative_pairs"],1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(f.grad).all()))

    def test_cross_speaker_same_label_is_eii_easy_negative(self):
        f = torch.ones(1,3,2,requires_grad=True)
        y = torch.tensor([[0,0,0]])
        q = speakers([[0,0,1]])
        m = torch.ones(1,3)
        loss,stats = contrastive_loss(f,y,q,m,"eii_fixed",return_stats=True)
        self.assertAlmostEqual(float(loss.detach()),math.log(2),places=6)
        self.assertEqual(stats["eligible_anchors"],2)
        self.assertEqual(stats["hard_negative_pairs"],0)

    def test_no_positives_returns_graph_connected_zero(self):
        for arm in ARMS:
            f = torch.randn(1,3,4,requires_grad=True)
            loss,stats = contrastive_loss(f,torch.tensor([[0,1,2]]),
                speakers([[0,0,1]]),torch.ones(1,3),arm,return_stats=True)
            self.assertEqual(float(loss.detach()),0.)
            self.assertEqual(stats["eligible_anchors"],0)
            loss.backward()
            torch.testing.assert_close(f.grad,torch.zeros_like(f))

    def test_padding_does_not_affect_loss_or_gradient(self):
        torch.manual_seed(1)
        for arm in ARMS:
            f = torch.randn(1,4,5,requires_grad=True)
            y = torch.tensor([[0,0,1,1]])
            q = speakers([[0,0,0,1]])
            m = torch.ones(1,4)
            small = contrastive_loss(f,y,q,m,arm)
            padded = torch.cat((f.detach(),torch.randn(1,2,5)*100),dim=1).requires_grad_()
            big = contrastive_loss(padded,torch.tensor([[0,0,1,1,5,5]]),
                speakers([[0,0,0,1,0,1]]),torch.tensor([[1,1,1,1,0,0]]),arm)
            torch.testing.assert_close(small,big,atol=1e-6,rtol=1e-5)
            big.backward()
            self.assertTrue(bool(torch.isfinite(padded.grad).all()))
            torch.testing.assert_close(padded.grad[:,4:],torch.zeros_like(padded.grad[:,4:]))

    def test_hook_consumes_feature_and_restores_unmodified_inference(self):
        class FakeCSS(nn.Module):
            def __init__(self):
                super().__init__()
                self.all_output_layer = nn.Linear(4,6)
            def forward(self,x):
                return self.all_output_layer(x)
        torch.manual_seed(9)
        model = FakeCSS()
        f = torch.randn(1,4,4,requires_grad=True)
        before = model(f).detach().clone()
        hook = SupervisedContrastHook(model,"eii_fixed")
        try:
            pred = model(f)
            y = torch.tensor([[0,0,1,0]])
            q = speakers([[0,0,0,1]])
            m = torch.ones(1,4)
            raw = contrastive_loss(f,y,q,m,"eii_fixed")
            loss = hook(model,y,q,m)
            torch.testing.assert_close(loss,ALPHA*raw)
            self.assertEqual(hook.stats["batches"],1)
            self.assertIsNone(hook.feature)
            with self.assertRaises(RuntimeError):
                hook(model,y,q,m)
            (loss+pred.mean()).backward()
            self.assertTrue(bool(torch.isfinite(f.grad).all()))
        finally:
            hook.close()
        torch.testing.assert_close(model(f).detach(),before)


if __name__ == "__main__":
    unittest.main()

"""V13 sanity checks, no IEMOCAP/TEST access, CPU-only."""
import inspect
import unittest

import torch

import train_IEMOCAP as trainer
from research.v13_cross_speaker_pretext import (
    CrossSpeakerPredictivePretext, V13LossHook, gather_aba_audio,
)
from scripts.run_v13_cross_speaker_pretext import build_head


class TestV13CrossSpeakerPretext(unittest.TestCase):
    def test_exact_aba_turn_positions_and_no_padding(self):
        x=torch.arange(24,dtype=torch.float32).reshape(1,6,4)
        speakers=torch.tensor([[0,0,1,0,1,0]])
        valid=torch.ones(1,6,dtype=torch.bool)
        prev,other,current,eligible=gather_aba_audio(x,speakers,valid)
        self.assertEqual(
            eligible.tolist(),[[False,False,False,True,True,True]]
        )
        torch.testing.assert_close(prev[:,3],x[:,1])
        torch.testing.assert_close(other[:,3],x[:,2])
        torch.testing.assert_close(prev[:,4],x[:,2])
        torch.testing.assert_close(other[:,4],x[:,3])
        torch.testing.assert_close(prev[:,5],x[:,3])
        torch.testing.assert_close(other[:,5],x[:,4])
        torch.testing.assert_close(current,x)

        pad=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool)
        _,_,_,masked=gather_aba_audio(x,speakers,pad)
        self.assertEqual(masked.tolist(),
                         [[False,False,False,True,False,False]])

    def test_future_changes_cannot_affect_past_sources(self):
        torch.manual_seed(13)
        x=torch.randn(1,5,8)
        speaker=torch.tensor([[0,1,0,1,0]])
        valid=torch.ones((1,5),dtype=torch.bool)
        prev1,between1,_,mask1=gather_aba_audio(x,speaker,valid)
        changed=x.clone()
        changed[:,4,:]+=torch.arange(8,dtype=torch.float32)*10
        prev2,between2,_,mask2=gather_aba_audio(changed,speaker,valid)
        torch.testing.assert_close(prev1[:,:4],prev2[:,:4])
        torch.testing.assert_close(between1[:,:4],between2[:,:4])
        torch.testing.assert_close(mask1,mask2)

    def test_audio_targets_are_detached_and_source_gradients_flow(self):
        torch.manual_seed(14)
        audio=torch.randn(2,5,8,requires_grad=True)
        speaker=torch.tensor([[0,1,0,1,0],
                              [1,0,1,0,1]])
        valid=torch.ones(2,5,dtype=torch.bool)
        head=CrossSpeakerPredictivePretext(
            dim=8,bottleneck=4,mode="cross",temperature=.2
        )
        loss=head(audio,speaker,valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(audio.grad[:,:4].abs().sum()),0)
        # Targets are stop-gradient; last turn cannot be a source.
        torch.testing.assert_close(
            audio.grad[:,4],
            torch.zeros_like(audio.grad[:,4]),
        )
        self.assertGreater(
            sum(float(p.grad.abs().sum()) for p in head.parameters()),0
        )

    def test_same_capacity_self_and_cross_and_matched_initial_weights(self):
        torch.manual_seed(15)
        a=build_head("self")
        r=build_head("cross")
        self.assertEqual(a.mode,"self")
        self.assertEqual(r.mode,"cross")
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in r.parameters())
        )
        for x,y in zip(a.parameters(),r.parameters()):
            torch.testing.assert_close(x,y)

    def test_head_init_does_not_perturb_original_training_rng(self):
        torch.manual_seed(16)
        first=torch.randn(2)
        expected=torch.randn(5)
        torch.manual_seed(16)
        again=torch.randn(2)
        build_head("cross")
        actual=torch.randn(5)
        torch.testing.assert_close(first,again)
        torch.testing.assert_close(expected,actual)

    def test_single_or_zero_triplets_skips_loss_cleanly(self):
        x=torch.randn(1,3,8,requires_grad=True)
        speaker=torch.tensor([[0,1,0]])
        valid=torch.ones(1,3,dtype=torch.bool)
        head=CrossSpeakerPredictivePretext(dim=8,bottleneck=4,mode="cross")
        loss=head(x,speaker,valid)
        self.assertEqual(loss.item(),0.0)
        loss.backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x.grad))

    def test_trainer_optional_hook_keeps_legacy_default_off(self):
        spec=inspect.signature(trainer.train_or_eval_model)
        self.assertIn("auxiliary_loss_fn",spec.parameters)
        self.assertIsNone(spec.parameters["auxiliary_loss_fn"].default)

    def test_auxiliary_hook_uses_speaker_metadata_without_labels(self):
        class MockModel:
            def __init__(self):
                self.acouf_input=torch.nn.Conv1d(8,8,1,bias=False)
        torch.manual_seed(17)
        model=MockModel()
        head=CrossSpeakerPredictivePretext(dim=8,bottleneck=4)
        hook=V13LossHook(head,coefficient=.05)
        x=torch.randn(5,2,8,requires_grad=True)
        qmask=torch.nn.functional.one_hot(
            torch.tensor([[0,1,0,1,0],[1,0,1,0,1]]),2
        ).float()
        umask=torch.ones(2,5)
        loss=hook(model,x,qmask,umask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(hook.last_triplets,1)
        loss.backward()
        self.assertGreater(float(model.acouf_input.weight.grad.abs().sum()),0)
        self.assertGreater(float(x.grad.abs().sum()),0)


if __name__=="__main__":
    unittest.main()

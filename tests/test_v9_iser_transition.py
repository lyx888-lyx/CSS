"""No-data tests for V9 predictive transition refinement."""
import unittest

import torch

from research.iser_transition import (
    InteractionTransition, SelfTransition, history_features,
    normalize_modalities, predictive_distribution, refine_logits,
    shuffled_train_donors
)


class TestISER(unittest.TestCase):
    def test_exact_intervening_other_turns_and_first_turns(self):
        # A A B C A B B A
        speakers = torch.tensor([0,0,1,2,0,1,1,0])
        p = torch.rand(8,7).softmax(-1)
        other = torch.arange(16).float().reshape(8,2)
        x = history_features(p, speakers, other)
        self.assertEqual(x["repeat"].tolist(),
                         [False,True,False,False,True,True,True,True])
        self.assertEqual(x["eligible"].tolist(),
                         [False,False,False,False,True,True,False,True])
        torch.testing.assert_close(x["previous"][4], p[1])
        torch.testing.assert_close(x["other"][4], other[2:4].mean(0))
        torch.testing.assert_close(x["other"][7], other[5:7].mean(0))
        torch.testing.assert_close(x["other"][5], other[3:5].mean(0))
        self.assertTrue((x["other"][~x["eligible"]] == 0).all())

    def test_future_raw_turns_cannot_change_earlier_history(self):
        p = torch.rand(5,7).softmax(-1)
        speakers = torch.tensor([0,1,0,1,0])
        x = torch.randn(5,6)
        one = history_features(p, speakers,x)
        x[3:] += 100
        p[3:] = torch.rand(2,7).softmax(-1)
        two = history_features(p,speakers,x)
        torch.testing.assert_close(one["other"][:3], two["other"][:3])
        torch.testing.assert_close(one["previous"][:3], two["previous"][:3])

    def test_predicted_distribution_not_hard_class(self):
        prev = torch.tensor([[.7,.3]])
        scores = torch.tensor([[2.,0.],[0.,2.]])
        q = predictive_distribution(prev,scores)
        torch.testing.assert_close(q, prev @ scores.softmax(-1))
        torch.testing.assert_close(q.sum(-1), torch.ones(1))

    def test_conditional_init_is_self_and_increment_starts_zero(self):
        self_model = SelfTransition(7)
        with torch.no_grad():
            self_model.scores.normal_()
        p = torch.randn(4,7).softmax(-1)
        x = torch.randn(4,10)
        base = torch.randn(4,7).log_softmax(-1)
        mask = torch.tensor([True,False,True,False])
        prior = torch.full((7,), 1/7)
        head = InteractionTransition(self_model.scores, 10)
        logits,q = refine_logits(base,p,mask,prior,self_model,head,x,"incremental")
        torch.testing.assert_close(logits, base, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(q,self_model(p))
        a,_ = refine_logits(base,p,mask,prior,self_model,mode="self")
        b,_ = refine_logits(base,p,mask,prior,self_model,head,x,"conditional")
        torch.testing.assert_close(a,b)

    def test_gradient_and_unchanged_ineligible_predictions(self):
        self_model = SelfTransition(7).requires_grad_(False)
        head = InteractionTransition(self_model.scores, 10)
        mask = torch.tensor([True,False,True,False])
        p = torch.rand(4,7).softmax(-1)
        x = torch.randn(4,10)
        base = torch.rand(4,7).log_softmax(-1)
        prior = torch.full((7,),1/7)
        with torch.no_grad():
            head.net[-1].weight.normal_()
        z,q = refine_logits(base,p,mask,prior,self_model,head,x,"incremental")
        self.assertTrue(torch.equal(z[~mask],base[~mask]))
        loss = torch.nn.functional.cross_entropy(z,torch.tensor([1,2,3,4]))
        loss.backward()
        self.assertGreater(float(head.net[-1].weight.grad.abs().sum()),0)
        self.assertIsNone(self_model.scores.grad)

    def test_shuffled_donors_train_only_and_different_dialogue(self):
        train = dict(eligible=torch.tensor([True,True,True,False]),
                     dialogue=torch.tensor([10,20,30,20]),
                     other=torch.tensor([[10.],[20.],[30.],[99.]]))
        a = shuffled_train_donors(train,train,11)
        b = shuffled_train_donors(train,train,11)
        torch.testing.assert_close(a,b)
        self.assertTrue((a[:3,0] != train["dialogue"][:3]).all())
        self.assertEqual(a[3,0].item(),0)
        val = dict(eligible=torch.tensor([True,False]),
                   dialogue=torch.tensor([40,41]),
                   other=torch.tensor([[999.],[999.]]))
        c = shuffled_train_donors(val,train,12)
        self.assertIn(c[0,0].item(),[10,20,30])
        self.assertEqual(c[1,0].item(),0)

    def test_equal_modal_scales(self):
        a,b,c = torch.randn(3,10),torch.randn(3,4),torch.randn(3,2)
        x = normalize_modalities(a,b,c)
        self.assertEqual(x.shape,(3,16))
        torch.testing.assert_close(x[:,:10].norm(dim=-1),x[:,10:14].norm(dim=-1))
        torch.testing.assert_close(x[:,:10].norm(dim=-1),x[:,14:].norm(dim=-1))


if __name__ == "__main__":
    unittest.main()

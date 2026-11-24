"""Synthetic tests; no real dataset or checkpoint access."""
import unittest
import numpy as np
import torch
from research.history_masking_audit import matched_positions, summarize
from scripts.run_history_masking_audit import infer_masks


class TestHistoryMasking(unittest.TestCase):
    def test_controls_are_past_distinct_and_label_free(self):
        positions = matched_positions(["M","F","M","M","F","M"])
        self.assertEqual(positions, [(2,0,1,None),(3,2,1,0),
                                    (4,1,2,None),(5,3,4,2)])
        self.assertEqual(matched_positions(["M","M"]), [])
        for i,j,k,h in positions:
            self.assertLess(j,i)
            self.assertLess(k,i)
            self.assertNotEqual(j,k)
            if h is not None:
                self.assertLess(h,j)

    def test_mask_only_changes_one_key_not_length_or_input(self):
        class FakeModel(torch.nn.Module):
            def forward(self,t,v,a,u,q,lengths):
                self.assertions(t,v,a,u,q,lengths)
                b,l = u.shape
                logits = torch.zeros(b,l,6)
                logits[:,:,0] = u[:,0].unsqueeze(-1)
                logits[:,:,1] = u[:,1].unsqueeze(-1)
                p = torch.softmax(logits,dim=-1)
                return (None,None,None,None,p)
            def assertions(self,t,v,a,u,q,lengths):
                assert t.shape[0] == 3 and q.shape[1] == 3
                assert all(x == 3 for x in lengths)
                assert torch.all(t == 7)
                assert torch.all((u == 0).sum(1) <= 1)
        item = (torch.full((3,4),7.),torch.zeros(3,2),torch.zeros(3,5),
                torch.tensor([[1.,0.],[0.,1.],[1.,0.]]),torch.ones(3),
                torch.tensor([0,1,2]),"Ses01_fake")
        result = infer_masks(FakeModel(),item,[None,0,1],torch.device("cpu"),2)
        self.assertGreater(result[None][2,0],result[0][2,0])
        self.assertGreater(result[None][2,1],result[1][2,1])
        self.assertEqual(result[0].shape,(3,6))

    def test_fixed_broken_history_errors_and_subgroups(self):
        def row(index,y,prior,b,o,c):
            r = dict(dialogue="Ses01_fake",session="Ses01",index=index,
                     previous_own_index=1,near_other_index=0,
                     earlier_own_index=None,true=y,previous_own_true=prior)
            for arm,pred in (("baseline",b),("previous_own",o),("near_other",c)):
                r[arm+"_pred"] = pred
                r[arm+"_prob"] = np.eye(6)[pred].tolist()
            return r
        r = [row(2,3,5,5,3,5),row(3,5,5,5,3,5)]
        s = summarize(r)["primary_groups"]
        self.assertEqual(s["all_matched"]["previous_own"]["fixed"],1)
        self.assertEqual(s["all_matched"]["previous_own"]["broken"],1)
        self.assertEqual(s["changed"]["previous_own"]["net_reduction_wrong_previous_emotion"],1)
        self.assertEqual(s["changed"]["near_other"]["net_reduction_wrong_previous_emotion"],0)
        self.assertEqual(s["stable"]["previous_own"]["accuracy_delta_pp"],-100.)
        self.assertEqual(s["three_pair_transitions"]["previous_own"]["n"],1)
        with self.assertRaises(ValueError):
            summarize(r+r[:1])


if __name__ == "__main__":
    unittest.main()

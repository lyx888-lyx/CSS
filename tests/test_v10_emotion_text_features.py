import unittest
import torch
from torch import nn
from research.v10_emotion_text_features import (
    EXTRA_DIM, session_split, text_inputs, pack_tokens,
    expand_text_projection, FeatureDataset
)


class ToyTokenizer:
    def num_special_tokens_to_add(self,pair=False):
        return 4 if pair else 2

    def encode(self,text,add_special_tokens=False,truncation=True,max_length=128):
        return [i+4 for i in range(len(text.split()))][:max_length]

    def build_inputs_with_special_tokens(self,ids,past=None):
        return [0]+ids+[2] if past is None else [0]+ids+[2,2]+past+[2]


class TestV10(unittest.TestCase):
    def test_fixed_session_split(self):
        keys=["Ses01F_impro01","Ses02M_impro02","Ses04F_script01","Ses03F_impro01"]
        self.assertEqual(session_split(keys),([0,1,3],[2]))
        with self.assertRaises(ValueError):
            session_split(keys+["Ses05F_impro01"])

    def test_roles_and_no_future(self):
        speakers=["M","F","M","F","M","M"]
        sentences=["a","b","c","d","e","f"]
        out=text_inputs(sentences,speakers)
        self.assertEqual(out[0],("a",""))
        self.assertEqual(out[2],("c","OTHER: b | SELF: a"))
        self.assertEqual(out[5],("f","SELF: e | OTHER: d | SELF: c | OTHER: b"))
        changed=text_inputs(sentences[:3]+["NEW","NEW","NEW"],speakers)
        self.assertEqual(out[:3],changed[:3])

    def test_token_budget_preserves_current(self):
        tok=ToyTokenizer()
        target=" ".join(["target"]*300)
        history=" ".join(["history"]*400)
        a=pack_tokens(tok,target,"")["input_ids"]
        b=pack_tokens(tok,target,history)["input_ids"]
        self.assertEqual(len(a),130)
        self.assertEqual(len(b),256)
        self.assertEqual(a[1:129],b[1:129])

    def test_expanded_projection_initially_same_and_trainable(self):
        torch.manual_seed(61)
        model=nn.Module()
        model.textf_input=nn.Conv1d(1024,16,1,bias=False)
        x=torch.randn(2,1024,4)
        extra=torch.randn(2,EXTRA_DIM,4)
        reference=model.textf_input(x).detach()
        expand_text_projection(model)
        output=model.textf_input(torch.cat([x,extra],dim=1))
        torch.testing.assert_close(output,reference,atol=1e-5,rtol=1e-5)
        self.assertTrue((model.textf_input.weight[:,1024:]==0).all())
        output.square().mean().backward()
        self.assertGreater(float(model.textf_input.weight.grad[:,1024:].abs().sum()),0)

    def test_wrapped_dataset_preserves_nontext(self):
        class Toy:
            def __len__(self): return 1
            def __getitem__(self,i):
                return (torch.ones(3,1024),torch.zeros(3,4),torch.zeros(3,5),
                        torch.zeros(3,2),torch.ones(3),torch.tensor([1,0,2]),"d")
            def collate_fn(self,rows): return rows
        ds=Toy()
        cache={"current":{"d":torch.ones(3,EXTRA_DIM)}}
        base=FeatureDataset(ds,"original",cache)[0]
        ext=FeatureDataset(ds,"current",cache)[0]
        self.assertEqual(ext[0].shape,(3,1792))
        torch.testing.assert_close(ext[0][:,:1024],base[0])
        torch.testing.assert_close(ext[0][:,1024:].norm(dim=-1),base[0].norm(dim=-1))
        for a,b in zip(base[1:-1],ext[1:-1]):
            torch.testing.assert_close(a,b)
        self.assertEqual(base[-1],ext[-1])


if __name__=="__main__":
    unittest.main()

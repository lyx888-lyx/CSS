#!/usr/bin/env python3
"""Audit CSS's four existing heads on a SAVED TRAIN-only Session-4 checkpoint.

Reuses the original V10 CSS Session1-3 TRAIN / Session4 VALID model.
Does NOT train, create a TEST dataset, or evaluate Session5. Reports
which unimodal heads could have helped in hindsight, and which would
have harmed already-correct fused predictions. Not a learned selector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from dataloader import IEMOCAPDataset
from scripts.train_research import build_model
from research.css_head_complementarity import HEADS,summarize

DEFAULT_CKPT=Path("results/v10_iemocap/paired/original_valid_best.pt")
DEFAULT_OUTPUT=Path("results/css_head_audit/session4_original_heads.json")


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda:handle.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def validate_v10_session_split(ds,ckpt):
    conf=ckpt.get("config",{})
    expected={
        "dataset":"IEMOCAP","v10_feature_mode":"original",
        "v10_validation_session":4,"speaker_mode":"off",
        "precontext_mode":"off","v8_spf_mode":"off",
        "v61_spf_mode":"full","seed":61078,
        "classes":6,"speakers":2,
    }
    for key,want in expected.items():
        if conf.get(key)!=want:
            raise ValueError(f"checkpoint config {key}: expected {want}; "
                             f"got {conf.get(key)}")
    if ckpt.get("selection")!="validation_weighted_f1":
        raise ValueError("not an original CSS validation-selected checkpoint")
    train=list(ckpt.get("train_dialogue_indices",[]))
    valid=list(ckpt.get("valid_dialogue_indices",[]))
    if (len(ds)!=120 or len(train)+len(valid)!=120 or
            len(set(train+valid))!=120 or
            sorted(train+valid)!=list(range(120))):
        raise ValueError("invalid original-TRAIN dialogue split")
    if (not valid or
            any(not str(ds.keys[i]).startswith("Ses04") for i in valid) or
            any(str(ds.keys[i]).startswith("Ses04") for i in train) or
            {str(ds.keys[i])[:5] for i in train}!={"Ses01","Ses02","Ses03"}):
        raise ValueError("V10 original checkpoint is NOT Session4-held-out")
    n_train=sum(len(ds.videoLabels[ds.keys[i]]) for i in train)
    n_valid=sum(len(ds.videoLabels[ds.keys[i]]) for i in valid)
    if (n_train,n_valid)!=(4298,1512):
        raise ValueError(f"unexpected utterance split {(n_train,n_valid)}")
    return train,valid,n_train,n_valid


@torch.no_grad()
def predict_heads(model,loader,device):
    model.eval()
    y=[]
    pred={head:[] for head in HEADS}
    probs={head:[] for head in HEADS}
    seen_dialogues=0
    for batch in loader:
        text,visual,audio,qmask,umask,labels,vids=batch
        mask=umask.bool()
        lens=umask.long().sum(dim=1).tolist()
        if any(x<=0 for x in lens):
            raise ValueError("empty valid dialogue")
        outputs=model(
            text.to(device),visual.to(device),audio.to(device),
            umask.to(device),qmask.permute(1,0,2).to(device),lens
        )
        if len(outputs)<5:
            raise ValueError("CSS model must output 4 head log-probabilities")
        mask_device=mask.to(device)
        y.extend(labels[mask].long().cpu().tolist())
        for name,idx in zip(HEADS,range(4)):
            lp=outputs[idx]
            if lp.shape[-1]!=6 or tuple(lp.shape[:2])!=tuple(mask.shape):
                raise ValueError(f"CSS head {name} shape mismatch")
            prob=lp.exp()[mask_device]
            probs[name].append(prob.cpu().numpy())
            pred[name].extend(prob.argmax(-1).cpu().tolist())
        # outputs[4] is fused all_prob; exact match is an integrity check.
        fused_from_model=outputs[4][mask_device]
        if not torch.allclose(
            fused_from_model,outputs[3].exp()[mask_device],
            atol=1e-5,rtol=1e-5
        ):
            raise RuntimeError("fused log-prob and softmax disagree")
        seen_dialogues+=len(vids)
    return (np.asarray(y,dtype=np.int64),
            {head:np.asarray(pred[head],dtype=np.int64) for head in HEADS},
            {head:np.concatenate(probs[head],axis=0) for head in HEADS},
            seen_dialogues)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=DEFAULT_CKPT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cuda",action="store_true")
    args=parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Missing V10 original CSS Session4 checkpoint: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite previous audit: {args.output}")
    # Only load a checkpoint saved by the user's own previous CSS training.
    cp=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    ds=IEMOCAPDataset(train=True)
    _,valid,n_train,n_valid=validate_v10_session_split(ds,cp)
    device=torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model=build_model(cp["config"]).to(device)
    model.load_state_dict(cp["model_state_dict"],strict=True)
    loader=DataLoader(
        Subset(ds,valid),batch_size=32,shuffle=False,
        collate_fn=ds.collate_fn,num_workers=0
    )
    y,preds,probs,n_dialogues=predict_heads(model,loader,device)
    if n_dialogues!=len(valid) or len(y)!=n_valid:
        raise RuntimeError(f"VALID coverage wrong: {n_dialogues} / {len(y)}")
    metrics=summarize(y,preds,probs)
    baseline=float(cp["validation_f1"])
    if abs(metrics["per_head"]["fused"]["weighted_f1"]-baseline)>0.03:
        raise RuntimeError(
            f"checkpoint re-eval differs from stored W-F1 {baseline:.2f}: "
            f"{metrics['per_head']['fused']['weighted_f1']:.2f}"
        )
    report={
        "protocol":{
            "partition":"IEMOCAP OFFICIAL TRAIN ONLY",
            "train_sessions":[1,2,3],"held_out_train_session":4,
            "train_utterances":n_train,"held_out_utterances":n_valid,
            "held_out_dialogues":len(valid),
            "checkpoint_selection":"preexisting V10 validation selected",
            "checkpoint_epoch":int(cp["epoch"]),
            "checkpoint_sha256":file_hash(args.checkpoint),
            "test_loader_instantiated":False,"test_scored":False,
            "original_css_architecture_unchanged":True,
            "training_performed":False,
            "oracle_head_selector_leaks_gold":True,
        },
        "audit":metrics,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print("CSS FOUR-HEAD COMPLEMENTARITY AUDIT | EXISTING Session4 VALID")
    print("TRAIN Ses01-03 / held-out Ses04; official TEST never loaded")
    for h in HEADS:
        m=metrics["per_head"][h]
        print(f"{h:<7} ACC={m['accuracy']:.2f}% "
              f"W-F1={m['weighted_f1']:.2f}% Macro-F1={m['macro_f1']:.2f}%")
    print(f"Fused wrong: {metrics['fused_error_count']} / {len(y)}")
    print("Fused wrong, >=1 single head correct (ORACLE ONLY):",
          metrics["any_single_head_correct_when_fused_wrong"])
    print("Fused correct, >=1 single head DISAGREES:",
          metrics["fused_correct_but_some_single_head_disagrees"])
    print("Potential rescue per single head:")
    for h,x in metrics["per_other_head"].items():
        print(f"  {h}: rescue={x['fixable_fused_errors_by_oracle']} "
              f"risk={x['fused_correct_but_head_wrong']} "
              f"disagreement={x['head_disagrees_with_fused']}")
    print("ORACLE is not a realizable selection method; no performance claim.")
    print("Saved:",args.output)


if __name__=="__main__":
    main()

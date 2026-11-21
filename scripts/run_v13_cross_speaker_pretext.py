#!/usr/bin/env python3
"""V13 CSS-only: label-free predictive A->B->A auxiliary learning.

Uses the SAME original-author full 120-dialogue/5810-utterance TRAIN,
original PGM, batch32, 200 epochs, source Test-selected best checkpoint
benchmark as V12. This is only an author-style comparison, NOT an
independent TEST: previous V11/V12 and the author used TEST.

No extra network is used at inference. The only change from V12 CSS
is a training auxiliary InfoNCE objective, with matched-capacity arms:
  self  : infer A's next audio from A's previous audio
  cross : infer A's next audio from prior A + interposed B
All using strictly earlier turns; target audio is stop-gradient.
No gold emotion labels or TEST features in the auxiliary task.

Original CSS comparison = existing V12 full-train baseline, 75.28% TEST
W-F1. No retuning coefficients or architecture based on TEST outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedKLDivLoss, MaskedNLLLoss
from research.v13_cross_speaker_pretext import CrossSpeakerPredictivePretext, V13LossHook
from research.v11_test_protocol import test_metrics
from scripts.run_v12_author_protocol import (
    CLASS_WEIGHTS, create_model, inspect_train
)

SEED=61078
EPOCHS=200
ARMS=("self","cross")
BASELINE=Path("results/v12_original_css_author_protocol/author_test_selected/original/report.json")
ROOT=Path("results/v13_cross_speaker_pretext")
AUX_COEFFICIENT=0.05
TEMPERATURE=0.2
BOTTLENECK=256


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def verify_baseline(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing full author-style CSS baseline {path}; "
            "V12 original must be verified first."
        )
    base=json.loads(path.read_text(encoding="utf-8"))
    proto=base.get("protocol",{})
    if (base.get("arm")!="original"
        or proto.get("train_sessions")!=[1,2,3,4]
        or proto.get("train_dialogues")!=120
        or proto.get("train_utterances")!=5810
        or proto.get("official_test_session")!=5
        or proto.get("test_utterances")!=1623
        or proto.get("test_selection")!="best_test_weighted_f1_every_epoch"
        or proto.get("test_used_to_select_checkpoint") is not True
        or proto.get("seed")!=SEED
        or proto.get("training_epochs")!=EPOCHS
        or abs(base.get("metrics",{}).get("weighted_f1",0)-75.28)>0.025):
        raise ValueError("V12 baseline is not the known 75.28% original CSS run")
    return {"path":str(path),"sha256":sha256(path),
            "weighted_f1":float(base["metrics"]["weighted_f1"])}


def build_head(mode):
    if mode not in ARMS:
        raise ValueError("V13 mode must be self or cross")
    # CPU initialization of auxiliary head must not change global RNG:
    # this preserves the exact original CSS shared base initialization,
    # DataLoader sampler and initial dropout sequence at seed 61078.
    state=torch.random.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    head=CrossSpeakerPredictivePretext(
        dim=1024,bottleneck=BOTTLENECK,mode=mode,temperature=TEMPERATURE
    )
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return head


def execute(args):
    info=inspect_train()
    baseline=verify_baseline(args.original_report)
    print("V13 CSS original-source comparable PRETEXT:")
    print(json.dumps(info,ensure_ascii=False))
    print("V12 frozen original benchmark:",baseline,flush=True)
    print("Fixed arms: self / cross | lambda=0.05, temp=0.2, bottleneck=256")
    print("TRAIN all official sessions 1-4, NO session-4 exclusion.")
    if args.preflight:
        print("Preflight passed; no model training or TEST scoring.")
        return
    if not args.smoke and not args.acknowledge_test_based_selection:
        raise ValueError(
            "Full historical author protocol selects BEST epoch on TEST. "
            "Pass --acknowledge-test-based-selection; this is not an "
            "untouched, independent final TEST."
        )
    target_dir=args.output_dir/("smoke" if args.smoke else "author_test_selected")/args.arm
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {target_dir}")
    trainer.set_seed(SEED)
    trainer.cuda=torch.cuda.is_available() and not args.no_cuda
    trainer.args=argparse.Namespace(tensorboard=False)
    device=torch.device("cuda" if trainer.cuda else "cpu")

    model=create_model("original")
    head=build_head(args.arm)
    model.to(device)
    head.to(device)
    optimizer=torch.optim.Adam(
        list(model.parameters())+list(head.parameters()),
        lr=1e-4,weight_decay=1e-5
    )
    loss_fn=MaskedNLLLoss(
        torch.tensor([1/x for x in CLASS_WEIGHTS],device=device,dtype=torch.float32)
    )
    kl_fn=MaskedKLDivLoss()
    aux_fn=V13LossHook(head,AUX_COEFFICIENT)
    if args.smoke:
        # The TRAIN pickle includes the original TEST arrays but this does
        # not construct a TEST dataset/loader or call its evaluator.
        ds=IEMOCAPDataset(train=True)
        train_loader=DataLoader(
            ds,batch_size=32,
            sampler=SubsetRandomSampler(list(range(len(ds)))),
            collate_fn=ds.collate_fn,num_workers=0,
        )
        valid_loader=None
        test_loader=None
    else:
        # Source-faithful loader, including the original EMPTY valid loader.
        train_loader,valid_loader,test_loader=trainer.get_IEMOCAP_loaders(
            batch_size=32,valid=0.0,num_workers=0
        )
    target_dir.mkdir(parents=True,exist_ok=False)
    best_f1=-float("inf")
    best_acc=-float("inf")
    best_epoch=None
    history=[]
    ckpt_file=target_dir/"best.pt"

    for epoch in range(1 if args.smoke else EPOCHS):
        tr_loss,tr_acc,_,_,_,tr_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_fn,train_loader,epoch,
            optimizer=optimizer,train=True,auxiliary_loss_fn=aux_fn
        )
        if args.smoke:
            if aux_fn.last_aux_loss is None or aux_fn.last_triplets is None:
                raise RuntimeError("V13 auxiliary hook did not run")
            if not np.isfinite(aux_fn.last_aux_loss):
                raise RuntimeError("nonfinite auxiliary loss")
            print(f"V13 {args.arm} TRAIN smoke loss={tr_loss:.4f} "
                  f"aux_infoNCE_last_batch={aux_fn.last_aux_loss:.4f} "
                  f"last_batch_triples={aux_fn.last_triplets}",flush=True)
            (target_dir/"smoke.json").write_text(json.dumps({
                "mode":args.arm,
                "train_utterances":5810,
                "test_loader_created":False,
                "test_scored":False,
                "last_batch_auxiliary_loss":aux_fn.last_aux_loss,
                "last_batch_aba_triplets":aux_fn.last_triplets,
            },indent=2)+"\n",encoding="utf-8")
            return

        # The historical author loop calls an empty validation iterator.
        trainer.train_or_eval_model(
            model,loss_fn,kl_fn,valid_loader,epoch,train=False
        )
        te_loss,te_acc,_,_,_,te_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_fn,test_loader,epoch,train=False
        )
        tr_loss,tr_acc,tr_f1,te_acc,te_f1=[
            float(x) for x in (tr_loss,tr_acc,tr_f1,te_acc,te_f1)
        ]
        if (te_f1>best_f1 or (te_f1==best_f1 and te_acc>best_acc)):
            best_f1,best_acc,best_epoch=te_f1,te_acc,epoch+1
            torch.save({
                "model_state_dict":model.state_dict(),
                "pretext_head_state_dict":head.state_dict(),
                "epoch":best_epoch,
                "selection":"author_original_test_weighted_f1",
                "test_selected_wf1":te_f1,
                "test_selected_acc":te_acc,
                "model":"original CSS forward plus training-only pretext loss",
                "mode":args.arm,
                "seed":SEED,
                "train_sessions":[1,2,3,4],
                "official_test_session":5,
                "test_used_for_selection":True,
                "train_utterances":5810,
                "config":{
                    "aux_coefficient":AUX_COEFFICIENT,
                    "temperature":TEMPERATURE,
                    "bottleneck":BOTTLENECK,
                    "trained_epochs":EPOCHS,
                    "batch_size":32,
                    "lr":1e-4,"weight_decay":1e-5,
                },
            },ckpt_file)
        history.append({
            "epoch":epoch+1,"train_f1":tr_f1,
            "train_loss":tr_loss,"test_f1":te_f1,"test_accuracy":te_acc,
            "best_test_f1":best_f1,"best_epoch":best_epoch,
            "last_batch_triplets":aux_fn.last_triplets,
            "last_batch_aux_loss":aux_fn.last_aux_loss
        })
        print(f"V13 {args.arm} epoch={epoch+1:03d} "
              f"train_F1={tr_f1:.2f} TEST_F1={te_f1:.2f} "
              f"best_TEST_F1={best_f1:.2f} best_epoch={best_epoch} "
              f"aux_last={aux_fn.last_aux_loss:.4f}",flush=True)
        (target_dir/"history.json").write_text(
            json.dumps(history,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )

    saved=torch.load(ckpt_file,map_location="cpu",weights_only=True)
    if saved["epoch"]!=best_epoch:
        raise RuntimeError("saved best epoch mismatch")
    model.load_state_dict(saved["model_state_dict"],strict=True)
    te_acc,y,p,mask,te_f1=trainer.evaluate_loaded_model(model,test_loader)
    if abs(te_f1-best_f1)>.025:
        raise RuntimeError("best checkpoint TEST score mismatch")
    m=np.asarray(mask).astype(bool)
    y=np.asarray(y)[m]
    p=np.asarray(p)[m]
    if len(y)!=1623:
        raise RuntimeError("expected 1623 official TEST utterances")
    metrics=test_metrics(y,p)
    report={
        "protocol":{
            "scope":"author-style TEST-selected, not independent evaluation",
            "train_sessions":[1,2,3,4],
            "train_utterances":5810,
            "test_session":5,
            "test_utterances":1623,
            "test_selection":"best_test_weighted_f1_every_epoch",
            "test_used_to_select_checkpoint":True,
            "clean_independent_test":False,
            "seed":SEED,"epochs":EPOCHS,
            "aux_only_during_training":True,
            "identical_original_css_inference_architecture":True,
            "unmodified_pgm_gradient_balance":True,
            "baseline_manifest":baseline,
            "prior_test_look":"V11/V12 official TEST already viewed",
        },
        "arm":args.arm,
        "epoch":int(best_epoch),
        "checkpoint":str(ckpt_file),
        "checkpoint_sha256":sha256(ckpt_file),
        "metrics":metrics,
        "labels":list(map(int,y)),
        "predictions":list(map(int,p)),
    }
    (target_dir/"report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(f"\nV13 AUTHOR-STYLE FULL-TRAIN {args.arm} "
          f"BEST epoch={best_epoch} "
          f"TEST ACC={metrics['accuracy']:.2f}% "
          f"W-F1={metrics['weighted_f1']:.2f}% "
          f"Macro-F1={metrics['macro_f1']:.2f}%",flush=True)
    print("Original CSS frozen benchmark W-F1=75.28%; CSS paper=75.66%.")
    print("WARNING: checkpoint was selected on official TEST by source.")
    print("Saved:",target_dir/"report.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=ARMS,default="self")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--acknowledge-test-based-selection",action="store_true")
    p.add_argument("--original-report",type=Path,default=BASELINE)
    p.add_argument("--output-dir",type=Path,default=ROOT)
    p.add_argument("--no-cuda",action="store_true")
    args=p.parse_args()
    if args.preflight and (args.smoke or args.acknowledge_test_based_selection):
        p.error("--preflight cannot combine with smoke or TEST selection")
    if args.smoke and args.acknowledge_test_based_selection:
        p.error("smoke does not evaluate TEST")
    execute(args)


if __name__=="__main__":
    main()

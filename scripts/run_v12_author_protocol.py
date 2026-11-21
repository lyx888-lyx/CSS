#!/usr/bin/env python3
"""V12 faithful original CSS IEMOCAP benchmark, inside /code/CSS ONLY.

AUTHOR-COMPARABLE PROTOCOL (not independently held-out performance):
  - ALL original 120 TRAIN dialogues, Ses01-Ses04; no validation subtraction.
  - Ses05 original TEST; 200 epochs; seed61078; lr1e-4, wd1e-5,
    batch32, hidden1024, heads8, SPF rank16/order3, dropout0.5,
    temp1; original PGM and class weights.
  - Evaluate TEST at EVERY epoch and save best by TEST weighted F1
    (ACC breaks tied F1): faithful to source train_IEMOCAP.py.
  - No hyperparameter or architecture selection based on these
    benchmark TEST scores. THIS IS TEST-SELECTED, NOT A CLEAN TEST.

Run ORIGINAL FIRST and verify near the prior user 75.28% W-F1 result.
Only then run self/reaction unchanged, same protocol.

--smoke runs one training epoch without loading/evaluating TEST.
--preflight prints original TRAIN/session checks without loading TEST.
Full benchmark requires explicit --acknowledge-test-based-selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedKLDivLoss, MaskedNLLLoss, Transformer_Based_Model
from research.v11_acoustic_reaction import SpeakerRelativeAcousticReaction
from research.v11_test_protocol import test_metrics, paired_changes

SEED=61078
EPOCHS=200
MODELS=("original","self","reaction")
OUTPUT=Path("results/v12_original_css_author_protocol")
CLASS_WEIGHTS=(.086747,.144406,.227883,.160585,.127711,.252668)


def session_label(vid):
    name=str(vid)
    if not name.startswith("Ses0") or len(name)<5:
        raise ValueError(f"unexpected IEMOCAP recording: {name}")
    return name[:5]


def inspect_train():
    """Only original TRAIN partition, and only check session IDs."""
    ds=IEMOCAPDataset(train=True)
    keys=list(ds.keys)
    names={session_label(k) for k in keys}
    if names!={"Ses01","Ses02","Ses03","Ses04"}:
        raise ValueError(f"TRAIN does not cover only sessions1-4: {names}")
    if len(keys)!=120 or len(set(keys))!=120:
        raise ValueError(f"expected original TRAIN 120 dialogues; got {len(keys)}")
    n=sum(len(ds.videoLabels[k]) for k in keys)
    if n!=5810:
        raise ValueError(f"expected 5810 TRAIN turns; got {n}")
    return dict(train_dialogues=len(keys),train_turns=n,
                train_sessions=sorted(names),
                test_not_loaded=True)


def checksum(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def create_model(arm):
    """Keep every ORIGINAL CSS weight AND random training stream identical.

    Original model is instantiated EXACTLY as original source.
    V11 arm is appended after original init with a separate private
    consumed RNG subsequence, then restore global RNG state:
    original weight initialization/shuffle/dropout remain aligned,
    and the two V11 arms get identical adapter initial parameters.
    """
    if arm not in MODELS:
        raise ValueError(f"unknown arm: {arm}")
    model=Transformer_Based_Model(
        "IEMOCAP",1,1024,342,1582,8,16,3,
        n_classes=6,hidden_dim=1024,n_speakers=2,dropout=.5,
    )
    if arm!="original":
        before=torch.random.get_rng_state()
        cuda_before=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model.v11_acoustic_mode=arm
        model.v11_acoustic_refiner=SpeakerRelativeAcousticReaction(
            dim=1024,mode=arm
        )
        torch.random.set_rng_state(before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
    if (model.v11_acoustic_mode != ("off" if arm=="original" else arm)):
        raise RuntimeError("V12 model architecture mismatch")
    return model


def run(args):
    info=inspect_train()
    print("V12 ORIGINAL CSS AUTHOR BENCHMARK PROTOCOL:")
    print(json.dumps(info,indent=2),flush=True)
    print("Train: ALL 120 dialogues / 5810 utterances (Ses01-04)")
    print("Original source: test-selected best epoch, no real validation split.")
    if args.preflight:
        print("PREFLIGHT ONLY: no TEST loaded or scored.")
        return
    if not args.smoke and not args.acknowledge_test_based_selection:
        raise ValueError(
            "Full original-paper comparable protocol evaluates and selects "
            "checkpoints on TEST every epoch. To acknowledge this, pass "
            "--acknowledge-test-based-selection. Never claim an untouched TEST."
        )
    arm=args.arm
    output_dir=args.output_dir / ("smoke" if args.smoke else "author_test_selected") / arm
    if output_dir.exists():
        raise FileExistsError(
            f"Will not overwrite {output_dir}. Keep every benchmark run."
        )
    # Check first, before creating files or touching official TEST.
    if not args.smoke:
        if arm!="original":
            orig=(args.output_dir/"author_test_selected"/"original"/"report.json")
            if not orig.is_file():
                raise FileNotFoundError(
                    "Run the original CSS baseline with exact original "
                    "TRAIN/TEST protocol first, inspect its F1 before methods. "
                    f"Missing: {orig}"
                )
            baseline=json.loads(orig.read_text(encoding="utf-8"))
            if baseline["protocol"]["test_selection"]!="best_test_weighted_f1_every_epoch":
                raise ValueError("original benchmark has wrong protocol")
            base_f1=float(baseline["metrics"]["weighted_f1"])
            if base_f1<74.0:
                raise RuntimeError(
                    f"Baseline currently only {base_f1:.2f}% TEST W-F1 "
                    "(prior run 75.28%, paper 75.66%). Diagnose protocol "
                    "or dependencies before further method training."
                )

    trainer.set_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainer.cuda=(device.type=="cuda")
    trainer.args=argparse.Namespace(tensorboard=False)
    model=create_model(arm).to(device)

    # Match original source setup: optimizer and class weights, THEN loaders.
    optimizer=torch.optim.Adam(
        model.parameters(),lr=1e-4,weight_decay=1e-5
    )
    weights=torch.tensor([1.0/x for x in CLASS_WEIGHTS],
                         dtype=torch.float32,device=device)
    loss_fn=MaskedNLLLoss(weights)
    kl_fn=MaskedKLDivLoss()
    if args.smoke:
        # One epoch TRAIN only: test partition is never instantiated.
        train_ds=IEMOCAPDataset(train=True)
        sampler=torch.utils.data.SubsetRandomSampler(list(range(len(train_ds))))
        train_loader=torch.utils.data.DataLoader(
            train_ds,batch_size=32,sampler=sampler,
            collate_fn=train_ds.collate_fn,num_workers=0,
        )
        test_loader=None
    else:
        # Exact loader and sampler method used in original source,
        # with valid=0.0: no TRAIN dialogue is held back.
        train_loader, _valid_loader, test_loader=trainer.get_IEMOCAP_loaders(
            valid=0.0,batch_size=32,num_workers=0
        )

    output_dir.mkdir(parents=True,exist_ok=False)
    checkpoint_path=output_dir/"best.pt"
    history=[]
    best_f1=-float("inf")
    best_acc=-float("inf")
    best_epoch=None
    latest=None
    for epoch in range(1 if args.smoke else EPOCHS):
        tr_loss,tr_acc,_,_,_,tr_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_fn,train_loader,epoch,
            optimizer=optimizer,train=True
        )
        if args.smoke:
            print(f"V12 {arm} TRAIN-ONLY smoke epoch1 "
                  f"loss={tr_loss:.4f} acc={tr_acc:.2f} f1={tr_f1:.2f}",
                  flush=True)
            break
        # AUTHOR PROTOCOL reproduces the historical TEST-at-each-epoch
        # selection; intentionally NOT equivalent to a clean final TEST.
        test_loss,test_acc,labels,preds,masks,test_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_fn,test_loader,epoch,train=False
        )
        tr_acc,tr_f1,test_acc,test_f1=[
            float(x) for x in (tr_acc,tr_f1,test_acc,test_f1)
        ]
        is_best=(test_f1>best_f1 or
                 (test_f1==best_f1 and test_acc>best_acc))
        if is_best:
            best_f1,best_acc,best_epoch=test_f1,test_acc,epoch+1
            # Keep checkpoint and provenance, but no NumPy scalar objects.
            torch.save({
                "model_state_dict":model.state_dict(),
                "config":dict(
                    dataset="IEMOCAP",seed=SEED,epochs=EPOCHS,
                    temp=1,heads=8,rank=16,order=3,dropout=.5,
                    hidden_dim=1024,lr=1e-4,l2=1e-5,
                    batch_size=32,classes=6,speakers=2,
                    v11_acoustic_mode=arm if arm!="original" else "off",
                    v61_spf_mode="full",v8_spf_mode="off",
                ),
                "epoch":epoch+1,
                "test_selected_wf1":test_f1,
                "test_selected_acc":test_acc,
                "selection":"author_original_test_weighted_f1",
                "train_sessions":[1,2,3,4],
                "official_test_session":5,
                "test_used_for_selection":True,
            },checkpoint_path)
        latest=dict(epoch=epoch+1,train_wf1=tr_f1,
                    train_acc=tr_acc,train_loss=float(tr_loss),
                    test_wf1=test_f1,test_acc=test_acc,
                    best_test_wf1=best_f1,best_epoch=best_epoch)
        history.append(latest)
        print(f"V12 author mode={arm:<8} epoch={epoch+1:03d} "
              f"train_F1={tr_f1:.2f} TEST_F1={test_f1:.2f} "
              f"best_TEST_F1={best_f1:.2f} best_epoch={best_epoch}",
              flush=True)
        # Persist progress even if user interrupts.
        (output_dir/"history.json").write_text(
            json.dumps(history,indent=2,ensure_ascii=False)+"\n",
            encoding="utf-8"
        )

    if args.smoke:
        (output_dir/"smoke.json").write_text(
            json.dumps({
                "arm":arm,"train_dialogues":120,"train_utterances":5810,
                "test_loaded":False,"test_evaluated":False,
                "train_loss":float(tr_loss),"train_wf1":float(tr_f1),
            },indent=2)+"\n",encoding="utf-8")
        print("Smoke passed. Did not open or evaluate TEST.",flush=True)
        return

    if not checkpoint_path.is_file() or best_epoch is None:
        raise RuntimeError("No best TEST-selected checkpoint was saved")
    chosen=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if chosen["epoch"]!=best_epoch:
        raise RuntimeError("saved checkpoint epoch mismatch")
    model.load_state_dict(chosen["model_state_dict"],strict=True)
    # Re-evaluate frozen selected epoch, for classification report only.
    test_acc,y,p,mask,test_f1=trainer.evaluate_loaded_model(model,test_loader)
    if abs(float(test_f1)-best_f1)>.025:
        raise RuntimeError("TEST-selected checkpoint score mismatch")
    sel=np.asarray(mask).astype(bool)
    y=np.asarray(y)[sel]
    p=np.asarray(p)[sel]
    if len(y)!=1623:
        raise RuntimeError(f"unexpected TEST utterances: {len(y)}")
    metrics=test_metrics(y,p)
    report={
        "protocol":{
            "dataset":"IEMOCAP",
            "train_sessions":[1,2,3,4],
            "train_dialogues":120,
            "train_utterances":5810,
            "official_test_session":5,
            "test_utterances":1623,
            "train_validation_ratio":0.0,
            "seed":SEED,
            "training_epochs":EPOCHS,
            "batch_size":32,
            "test_selection":"best_test_weighted_f1_every_epoch",
            "original_author_source":"train_IEMOCAP.py",
            "test_used_to_select_checkpoint":True,
            "clean_independent_test":False,
            "scope":"historical author-style benchmark reproduction, not unbiased generalization",
        },
        "arm":arm,"epoch":best_epoch,
        "checkpoint":str(checkpoint_path),
        "checkpoint_sha256":checksum(checkpoint_path),
        "metrics":metrics,
        "labels":list(map(int,y.tolist())),
        "predictions":list(map(int,p.tolist())),
    }
    (output_dir/"report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    print(f"\nV12 AUTHOR-STYLE FULL-TRAIN IEMOCAP BENCHMARK {arm}:")
    print(f"TRAIN sessions 1-4 (5810 utterances); TEST session 5 (1623)")
    print(f"TEST BEST EPOCH={best_epoch}, ACC={metrics['accuracy']:.2f}% "
          f"weighted_F1={metrics['weighted_f1']:.2f}% "
          f"macro_F1={metrics['macro_f1']:.2f}%",flush=True)
    print("Reference: CSS paper W-F1=75.66%, previous full reproduction 75.28%.")
    print("WARNING: TEST SELECTED checkpoint; NOT an unbiased held-out score.")
    print("Saved:",output_dir/"report.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=MODELS,default="original")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--acknowledge-test-based-selection",action="store_true")
    p.add_argument("--no-cuda",action="store_true")
    p.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=p.parse_args()
    if args.preflight and args.smoke:
        p.error("preflight and smoke are exclusive")
    if args.smoke and args.acknowledge_test_based_selection:
        p.error("smoke never evaluates TEST; do not pass acknowledgment")
    run(args)


if __name__=="__main__":
    main()

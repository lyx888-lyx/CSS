#!/usr/bin/env python3
"""CSS full TRAIN weights-only SWA/EMA controlled experiment.

TRAIN only: official IEMOCAP Ses01–04, 120 dialogues, 5810 turns.
Train original CSS ONCE with untouched original PGM, Adam, batch32,
seed61078, and 200 epochs. Track fixed epoch averages:
  RAW  = trained weights at end of epoch 200
  EMA  = 0.9 exponential moving average, epochs 100..200 inclusive
  SWA  = arithmetic weight average, epochs 151..200 inclusive

Never inspect TEST during training or select a checkpoint based on TEST.
The optional --evaluate-test-once flag explicitly loads Session5 and
reports all 3 FIXED checkpoints in one predetermined comparison.
Previous work repeatedly exposed TEST; no independent holdout claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedKLDivLoss, MaskedNLLLoss
from research.css_weight_averaging import (
    EMA_START,EMA_DECAY,SWA_START,FINAL_EPOCH,
    EpochWeightAverager,frozen_cpu_state,verify_same_state_keys,
)
from research.v11_test_protocol import paired_changes,test_metrics
from scripts.run_v12_author_protocol import (
    CLASS_WEIGHTS,SEED,create_model,inspect_train
)

OUTPUT=Path("results/css_weight_averaging")
FIXED_ARMS=("raw_final","ema_final","swa_final")


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(1048576),b""):
            h.update(piece)
    return h.hexdigest()


def verify_preflight():
    info=inspect_train()
    if (info["train_dialogues"]!=120 or
            info["train_turns"]!=5810 or
            info["train_sessions"]!=["Ses01","Ses02","Ses03","Ses04"]):
        raise RuntimeError("original CSS TRAIN partition mismatch")
    return info


def create_train_loader(ds):
    # Same original source sampler, same full 120 dialogues; no TEST dataset
    # is instantiated here. Different RNG usage from author's TEST-every-
    # epoch loop is intentional: all three arms share one training run.
    return DataLoader(
        ds,batch_size=32,
        sampler=SubsetRandomSampler(list(range(len(ds)))),
        collate_fn=ds.collate_fn,num_workers=0
    )


def train_run(args):
    info=verify_preflight()
    out=args.output_dir/("smoke" if args.smoke else "full")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite previous run: {out}")
    trainer.set_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainer.cuda=(device.type=="cuda")
    trainer.args=argparse.Namespace(tensorboard=False)
    model=create_model("original").to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    class_weight=torch.tensor(
        [1.0/w for w in CLASS_WEIGHTS],dtype=torch.float32,device=device
    )
    loss_fn=MaskedNLLLoss(class_weight)
    kl_loss=MaskedKLDivLoss()
    train_ds=IEMOCAPDataset(train=True)
    loader=create_train_loader(train_ds)
    ema=EpochWeightAverager("ema",start_epoch=1 if args.smoke else EMA_START,
                            decay=EMA_DECAY)
    swa=EpochWeightAverager("swa",start_epoch=1 if args.smoke else SWA_START)
    epochs=1 if args.smoke else FINAL_EPOCH
    out.mkdir(parents=True,exist_ok=False)
    history=[]
    for e in range(epochs):
        train_loss,train_acc,_,_,_,train_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_loss,loader,e,optimizer=optimizer,train=True
        )
        # AFTER optimizer step and after the entire original CSS epoch.
        ema.update(model,e+1)
        swa.update(model,e+1)
        item=dict(epoch=e+1,train_loss=float(train_loss),
                  train_accuracy=float(train_acc),
                  train_weighted_f1=float(train_f1))
        history.append(item)
        print(f"CSS weight-average epoch={e+1:03d} "
              f"train_WF1={train_f1:.2f}% "
              f"train_loss={train_loss:.4f} "
              f"EMA_updates={ema.count} SWA_updates={swa.count}",
              flush=True)
        (out/"history.json").write_text(
            json.dumps(history,indent=2)+"\n",encoding="utf-8"
        )
    raw=frozen_cpu_state(model)
    ema_state=ema.get_state()
    swa_state=swa.get_state()
    verify_same_state_keys(raw,ema_state,swa_state)
    if not args.smoke:
        if ema.count!=101 or swa.count!=50:
            raise RuntimeError("incorrect fixed averaging window")
    manifest={
        "protocol":{
            "dataset":"IEMOCAP",
            "train_sessions":[1,2,3,4],
            "train_dialogues":120,
            "train_utterances":5810,
            "train_seed":SEED,
            "epochs":epochs,
            "batch_size":32,
            "original_PGM_SPF_unchanged":True,
            "original_CSS_forward_unchanged":True,
            "optimizer":"Adam lr=1e-4 wd=1e-5",
            "same_training_trajectory_for_all_arms":True,
            "fixed_raw_epoch":FINAL_EPOCH,
            "ema_start":EMA_START,"ema_decay":EMA_DECAY,
            "swa_start":SWA_START,
            "checkpoint_selection":"fixed final epoch, NOT TEST",
            "historical_comparison_note":(
                "V12 CSS 75.28% was selected by inspecting TEST every epoch; "
                "this raw/EMA/SWA experiment never does that. Different "
                "intra-training RNG consumption is possible because original "
                "author loop traversed VALID/TEST every epoch."
            ),
            "official_test_dataset_instantiated":False,
            "official_test_scored_during_training":False,
            "smoke":bool(args.smoke),
        },
        "train_info":info,
        "ema":ema.summary(),
        "swa":swa.summary(),
        "train_last_epoch":history[-1],
        "checkpoints":{},
    }
    if not args.smoke:
        for name,state in zip(FIXED_ARMS,(raw,ema_state,swa_state)):
            checkpoint=out/(name+".pt")
            torch.save({
                "model_state_dict":state,
                "method":name,
                "selected_epoch":FINAL_EPOCH,
                "source_training":'one original CSS full-TRAIN run',
                "official_test_used_for_selection":False,
                "train_sessions":[1,2,3,4],
                "train_utterances":5810,
                "seed":SEED,
                "epochs":FINAL_EPOCH,
                "ema_start":EMA_START,
                "ema_decay":EMA_DECAY,
                "swa_start":SWA_START,
            },checkpoint)
            manifest["checkpoints"][name]={
                "path":str(checkpoint),
                "sha256":file_hash(checkpoint),
            }
    (out/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    if args.smoke:
        print("TRAIN-only one-epoch smoke passed; NO official TEST accessed.")
    else:
        print("Saved 3 predetermined CSS checkpoint arms; NO TEST evaluated.")
        print("Full-run manifest:",out/"manifest.json")
        print("Evaluate TEST only with explicit --evaluate-test-once "
              "--acknowledge-prior-test-exposure, once the experiment is frozen.")


def validate_full_manifest(manifest):
    p=manifest.get("protocol",{})
    if not (p.get("train_sessions")==[1,2,3,4]
            and p.get("train_dialogues")==120
            and p.get("train_utterances")==5810
            and p.get("train_seed")==SEED
            and p.get("epochs")==FINAL_EPOCH
            and p.get("checkpoint_selection")=="fixed final epoch, NOT TEST"
            and p.get("same_training_trajectory_for_all_arms") is True
            and p.get("official_test_scored_during_training") is False
            and p.get("smoke") is False
            and p.get("ema_start")==EMA_START
            and abs(p.get("ema_decay",-1)-EMA_DECAY)<1e-9
            and p.get("swa_start")==SWA_START):
        raise ValueError("not fixed final-epoch original CSS full TRAIN manifest")
    if (manifest.get("ema",{}).get("updates")!=101
            or manifest.get("swa",{}).get("updates")!=50):
        raise ValueError("weight averaging update count mismatch")
    if set(manifest.get("checkpoints",{}))!=set(FIXED_ARMS):
        raise ValueError("must evaluate all 3 predetermined arms")
    return True


def evaluate_once(args):
    out=args.output_dir/"full"
    report_path=out/"official_test_onepass_report.json"
    if report_path.exists():
        raise FileExistsError(f"Already has official TEST result: {report_path}")
    manifest_path=out/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Complete full-TRAIN run first: {manifest_path}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_full_manifest(manifest)
    if not args.acknowledge_prior_test_exposure:
        raise ValueError(
            "Official TEST Session5 was used repeatedly in V11-V13. "
            "Use --acknowledge-prior-test-exposure only for exploratory "
            "one-pass fixed-epoch comparison; it is NOT independent."
        )
    for name in FIXED_ARMS:
        row=manifest["checkpoints"][name]
        path=out/(name+".pt")
        if str(path)!=row["path"] or not path.is_file() or file_hash(path)!=row["sha256"]:
            raise ValueError(f"missing or changed frozen checkpoint {name}")
    # TEST is constructed here, and ONLY here: never in preflight/train.
    test_ds=IEMOCAPDataset(train=False)
    if len(test_ds)!=40:
        raise ValueError("expected 40 official Session5 TEST dialogues")
    if any(not str(k).startswith("Ses05") for k in test_ds.keys):
        raise ValueError("official TEST contains another session")
    loader=DataLoader(test_ds,batch_size=32,shuffle=False,
                      collate_fn=test_ds.collate_fn,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainer.cuda=(device.type=="cuda")
    trainer.args=argparse.Namespace(tensorboard=False)
    results={}
    base_labels=None
    for name in FIXED_ARMS:
        path=out/(name+".pt")
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if (checkpoint["method"]!=name or checkpoint["epochs"]!=FINAL_EPOCH
                or checkpoint["official_test_used_for_selection"] is not False):
            raise ValueError("checkpoint content/provenance mismatch")
        model=create_model("original").to(device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        acc,labels,preds,masks,_=trainer.evaluate_loaded_model(model,loader)
        valid=np.asarray(masks,dtype=bool)
        y=np.asarray(labels,dtype=np.int64)[valid]
        p=np.asarray(preds,dtype=np.int64)[valid]
        if y.shape!=(1623,):
            raise RuntimeError("official TEST expected 1623 valid utterances")
        if base_labels is None:
            base_labels=y.copy()
        elif not np.array_equal(base_labels,y):
            raise RuntimeError("different TEST sample ordering across methods")
        results[name]={
            "metrics":test_metrics(y,p),
            "predictions":p.astype(int).tolist(),
            "checkpoint_sha256":manifest["checkpoints"][name]["sha256"],
        }
        print(f"FIXED-epoch 200 {name:<10} TEST "
              f"ACC={results[name]['metrics']['accuracy']:.2f}% "
              f"W-F1={results[name]['metrics']['weighted_f1']:.2f}% "
              f"Macro-F1={results[name]['metrics']['macro_f1']:.2f}%",
              flush=True)
        del model,checkpoint
        if device.type=="cuda":
            torch.cuda.empty_cache()
    paired={}
    for name in FIXED_ARMS[1:]:
        paired[name]=paired_changes(
            base_labels,results[name]["predictions"],
            results["raw_final"]["predictions"]
        )
    report={
        "protocol":{
            "dataset":"IEMOCAP",
            "train_sessions":[1,2,3,4],
            "official_test_session":5,
            "official_test_utterances":1623,
            "selection":"ALL 3 fixed final-epoch states, one-pass TEST",
            "test_used_to_select_epoch":False,
            "historical_test_previously_exposed":True,
            "independent_confirmatory_result":False,
            "historical_test_selected_css_reference_wf1":75.28,
            "paper_css_reference_wf1":75.66,
            "comparison_is_matched_raw_epoch200_not_best_test_epoch106":True,
            "all_methods_share_exact_training_trajectory":True,
        },
        "labels":base_labels.tolist(),
        "results":results,
        "paired_vs_raw":paired,
        "training_manifest_sha256":file_hash(manifest_path),
    }
    report_path.write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    print("Paired correctness vs SAME-RUN RAW epoch 200:",paired)
    print("75.28% historical CSS used TEST to choose epoch; not equivalent.")
    print("Saved:",report_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight",action="store_true")
    modes.add_argument("--smoke",action="store_true")
    modes.add_argument("--train",action="store_true")
    modes.add_argument("--evaluate-test-once",action="store_true")
    parser.add_argument("--acknowledge-prior-test-exposure",action="store_true")
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.preflight:
        print("CSS weight averaging original full TRAIN preflight:")
        print(json.dumps(verify_preflight(),indent=2))
        print("NO TEST inference, model fitting, or dataset construction.")
    elif args.smoke or args.train:
        if args.acknowledge_prior_test_exposure:
            parser.error("TEST acknowledgement valid only in explicit onepass evaluation")
        train_run(args)
    else:
        evaluate_once(args)


if __name__=="__main__":
    main()

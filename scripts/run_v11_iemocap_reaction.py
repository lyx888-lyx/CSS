#!/usr/bin/env python3
"""V11 CSS-only dyadic acoustic response pilot (IEMOCAP).

Reuses V10's original CSS checkpoint, session 1-3 TRAIN/session 4 VALID,
original audio/text/visual SDT features and optimizer. No official TEST.
Two new SAME-CAPACITY from-scratch training arms:
   self     : prior-A -> current-A acoustic shift on A...B...A
   reaction : prior-A / interposed-B / current-A acoustic interactions

Session 4 was previously inspected in V10. This is hypothesis development,
not an independent confirmation or a claim of a causal effect.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedKLDivLoss, MaskedNLLLoss
from scripts.compare_validation import compute_report, predict_valid
from scripts.run_v10_iemocap_features import config as original_config, preflight
from scripts.train_research import build_model

SEED=61078
EPOCHS=200
MODES=("self","reaction")
GROUPS=("all","first","stable","changed","stable_after_other",
        "changed_after_other")
BASELINE=Path("results/v10_iemocap/paired/original_valid_best.pt")
DEFAULT_OUT=Path("results/v11_iemocap_acoustic_reaction")


def loader(ds,indices,train):
    return DataLoader(
        Subset(ds,indices),batch_size=32,shuffle=train,
        collate_fn=ds.collate_fn,num_workers=0,
        generator=(torch.Generator().manual_seed(SEED+1000) if train else None),
    )


@torch.no_grad()
def read_baseline(ds,tr,va,device,path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved V10 original CSS baseline: {path}. "
            "Do not replace with an unrelated checkpoint."
        )
    ckpt=torch.load(path,map_location="cpu",weights_only=True)
    cfg=ckpt["config"]
    if (cfg.get("dataset")!="IEMOCAP"
        or cfg.get("seed")!=SEED
        or cfg.get("v10_feature_mode")!="original"
        or cfg.get("v10_validation_session")!=4
        or cfg.get("epochs")!=200
        or cfg.get("v11_acoustic_mode","off")!="off"
        or ckpt.get("selection")!="validation_weighted_f1"
        or list(ckpt["train_dialogue_indices"])!=tr
        or list(ckpt["valid_dialogue_indices"])!=va):
        raise ValueError("Baseline is not the matched V10 original Session-4 CSS control")
    model=build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    y,p,tags=predict_valid(model,loader(ds,va,False),device)
    result=compute_report(y,p,tags,6)
    if result["all"]["n"]!=1512:
        raise RuntimeError("Expected V10 session-4 1512 utterances")
    if abs(result["all"]["weighted_f1"]-float(ckpt["validation_f1"]))>.025:
        raise RuntimeError("Baseline checkpoint score mismatch")
    if abs(result["all"]["weighted_f1"]-68.41)>.025:
        raise RuntimeError("Not the user-reported fixed V10 original CSS baseline")
    print("Matched original CSS V10 control re-inference:",
          f"{result['all']['weighted_f1']:.2f}% (session 4, no TEST)",flush=True)
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return dict(mode="original",selected_epoch=int(ckpt["epoch"]),
                scores=result,labels=y,predictions=p,checkpoint=str(path))


def train_arm(ds,tr,va,device,root,mode,epochs):
    if mode not in MODES: raise ValueError("invalid reaction mode")
    target=root/f"{mode}_valid_best.pt"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    cfg=original_config(epochs)
    cfg["v10_feature_mode"]="original"
    cfg["v10_validation_session"]=4
    cfg["v11_acoustic_mode"]=mode
    cfg["v11_pilot"]="same_capacity_response_residual"
    train_loader=loader(ds,tr,True)
    valid_loader=loader(ds,va,False)

    trainer.set_seed(SEED)
    model=build_model(cfg).to(device)
    if model.v11_acoustic_refiner is None:
        raise RuntimeError("V11 component absent")
    trainer.set_seed(SEED)
    trainer.cuda=(device.type=="cuda")
    trainer.args=argparse.Namespace(tensorboard=False)
    weights=torch.tensor([1/.086747,1/.144406,1/.227883,
                          1/.160585,1/.127711,1/.252668],
                         device=device,dtype=torch.float32)
    loss_fn=MaskedNLLLoss(weights)
    kl_fn=MaskedKLDivLoss()
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["l2"])
    best=(-float("inf"),-float("inf"))
    root.mkdir(parents=True,exist_ok=True)

    n_extra=sum(p.numel() for p in model.v11_acoustic_refiner.parameters())
    print(f"\n=== V11 {mode} | {n_extra} reaction params, "
          f"200-epoch fixed budget={epochs==200} ===",flush=True)
    for epoch in range(epochs):
        train_loss,train_acc,_,_,_,train_f1=trainer.train_or_eval_model(
            model,loss_fn,kl_fn,train_loader,epoch,optimizer=opt,train=True
        )
        valid_acc,_,_,_,valid_f1=trainer.evaluate_loaded_model(model,valid_loader)
        valid_acc,valid_f1=float(valid_acc),float(valid_f1)
        if (valid_f1,valid_acc)>best:
            best=(valid_f1,valid_acc)
            torch.save({
                "model_state_dict":model.state_dict(),
                "config":cfg,"epoch":epoch+1,"validation_f1":valid_f1,
                "validation_accuracy":valid_acc,
                "train_dialogue_indices":tr,"valid_dialogue_indices":va,
                "selection":"validation_weighted_f1",
            },target)
        print(f"{mode} epoch={epoch+1:03d} "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.2f} "
              f"valid_f1={valid_f1:.2f} best={best[0]:.2f}",flush=True)

    saved=torch.load(target,map_location="cpu",weights_only=True)
    model.load_state_dict(saved["model_state_dict"],strict=True)
    y,p,tags=predict_valid(model,valid_loader,device)
    scores=compute_report(y,p,tags,6)
    if abs(scores["all"]["weighted_f1"]-best[0])>.025:
        raise RuntimeError("V11 selected checkpoint re-inference mismatch")
    ret=dict(mode=mode,selected_epoch=int(saved["epoch"]),scores=scores,
             labels=y,predictions=p,checkpoint=str(target),n_extra=n_extra)
    del model,opt
    if device.type=="cuda": torch.cuda.empty_cache()
    return ret


def print_report(rows):
    print("\nV11 IEMOCAP SPEAKER-RELATIVE ACOUSTIC REACTION | VIEWED VALID | NO TEST")
    for group in GROUPS:
        baseline=rows[0]["scores"][group]["weighted_f1"]
        n=rows[0]["scores"][group]["n"]
        print(f"\nV11 group: {group} | n={n} utterances")
        for row in rows:
            x=row["scores"][group]["weighted_f1"]
            if x is None or baseline is None:
                print(f"  {row['mode']:<14} W-F1=unavailable")
            else:
                print(f"  {row['mode']:<14} W-F1={x:.2f}% "
                      f"delta-vs-CSS={x-baseline:+.2f}pp")
    y=torch.tensor(rows[0]["labels"])
    b=torch.tensor(rows[0]["predictions"])
    print("\nPaired correctness vs original CSS:")
    for row in rows:
        p=torch.tensor(row["predictions"])
        fixed=int(((p==y)&(b!=y)).sum())
        broken=int(((p!=y)&(b==y)).sum())
        flip=int((p!=b).sum())
        print(f"  {row['mode']:<14} fixed={fixed} broken={broken} "
              f"top1_flips={flip}")
    print("Selected epochs:",[(r["mode"],r["selected_epoch"]) for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-checkpoint",type=Path,default=BASELINE)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--no-cuda",action="store_true")
    args=p.parse_args()
    ds=IEMOCAPDataset(train=True)
    tr,va=preflight(ds)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    root=args.output_dir/("smoke" if args.smoke else "paired")
    modes=("reaction",) if args.smoke else MODES
    if (root/"report.json").exists() or any(
            (root/f"{m}_valid_best.pt").exists() for m in modes):
        raise FileExistsError(f"Existing output in {root}; refuse to overwrite")
    rows=[read_baseline(ds,tr,va,device,args.original_checkpoint)]
    for mode in modes:
        rows.append(train_arm(ds,tr,va,device,root,mode,
                              1 if args.smoke else EPOCHS))
    if any(r["labels"]!=rows[0]["labels"] for r in rows):
        raise RuntimeError("validation label/order mismatch")
    print_report(rows)
    report=dict(protocol={
        "dataset":"IEMOCAP","train_sessions":[1,2,3],
        "validation_session":4,"seed":SEED,
        "epochs":1 if args.smoke else EPOCHS,
        "already_inspected_validation":True,
        "official_test_scored":False,
        "same_speaker_previous_gold_labels_used":False,
        "other_speaker_features":(
            "projected acoustic features strictly in the open interval "
            "between the latest prior SAME speaker turn and current"),
        "checkpoint_selection":"validation_weighted_f1",
        "baseline_path":str(args.original_checkpoint),
        "base_audio_text_visual_features":"unchanged CSS SDT",
        "notes":(
            "V11 is exploratory. A positive change is not novelty or "
            "generalization proof. Reaction and self have matched parameter count."
        ),
    },models=rows)
    root.mkdir(parents=True,exist_ok=True)
    (root/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("Saved:",root/"report.json")
    print("NO TEST. Do not sweep coefficients or splits based on this VALID.")


if __name__=="__main__":
    main()

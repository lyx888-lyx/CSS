#!/usr/bin/env python3
"""V11: one-shot EXPLORATORY official TEST evaluation of 3 matched checkpoints.

Important: checkpoints trained only on sessions 1-3 and chosen by
session-4 validation; scores are NOT the paper's full sessions-1-4
training protocol and do not establish publication-level improvement.
This official TEST inspection makes TEST no longer independent for
any later V11 changes made after viewing these numbers.

Usage from /code/CSS:
  python -u -m scripts.evaluate_v11_test_once --preflight
  python -u -m scripts.evaluate_v11_test_once --evaluate-test
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset
from research.v10_emotion_text_features import session_split
from research.v11_test_protocol import (
    ARMS, test_metrics, validate_saved_checkpoint, paired_changes
)
from scripts.compare_validation import compute_report, predict_valid
from scripts.train_research import build_model

ROOT = Path("results/v11_iemocap_acoustic_reaction")
CHECKPOINTS = {
    "original": Path("results/v10_iemocap/paired/original_valid_best.pt"),
    "self": ROOT / "paired/self_valid_best.pt",
    "reaction": ROOT / "paired/reaction_valid_best.pt",
}
DEV_REPORT = ROOT / "paired/report.json"
OUTPUT = ROOT / "test_exploratory_once/report.json"
GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def preflight():
    """Verify stored TRAIN/VALID provenance BEFORE opening any official TEST."""
    ds=IEMOCAPDataset(train=True)
    train,valid=session_split(ds.keys)
    if (len(train), len(valid)) != (90,30):
        raise ValueError("unexpected training session split")
    if not DEV_REPORT.is_file():
        raise FileNotFoundError(f"missing original V11 VALID report: {DEV_REPORT}")
    dev=json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    expected=dev.get("models",[])
    if [m.get("mode") for m in expected] != list(ARMS):
        raise ValueError("V11 stored VALID report must have original/self/reaction")
    manifest=[]
    for arm,row in zip(ARMS,expected):
        path=CHECKPOINTS[arm]
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        ckpt=torch.load(path,map_location="cpu",weights_only=True)
        record=validate_saved_checkpoint(ckpt,arm,train,valid)
        if record["selected_epoch"] != row["selected_epoch"]:
            raise ValueError(f"{arm}: checkpoint epoch differs from VALID report")
        if abs(record["validation_f1"]-row["scores"]["all"]["weighted_f1"])>.025:
            raise ValueError(f"{arm}: checkpoint F1 differs from VALID report")
        # Explicitly check strict state-dict compatibility BEFORE seeing TEST.
        model=build_model(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"],strict=True)
        del model,ckpt
        record["checkpoint"]=str(path)
        record["checkpoint_sha256"]=sha256(path)
        manifest.append(record)
    print("V11 TEST PREFLIGHT PASSED. No official TEST loaded.")
    for r in manifest:
        print(f"  {r['arm']:<9} VALID={r['validation_f1']:.2f}% "
              f"selected_epoch={r['selected_epoch']} "
              f"checkpoint={r['checkpoint']}",flush=True)
    del ds
    return manifest


def evaluate(manifest, device, output):
    if output.exists() or (output.parent/"test_access_started.json").exists():
        raise FileExistsError(
            f"TEST already attempted in {output.parent}; do not repeat "
            "a test-based search or silently overwrite evidence."
        )
    output.parent.mkdir(parents=True,exist_ok=True)
    marker=output.parent/"test_access_started.json"
    # Durable record before loading the TEST partition.
    marker.write_text(json.dumps({
        "purpose":"exploratory one-time comparison, NOT final paper table",
        "train_sessions":[1,2,3],
        "dev_selection_session":4,
        "official_test_session":5,
        "official_test_access_started":True,
        "no_training_or_checkpoint_selection":True,
        "manifest":manifest,
        "warning":"Future V11 tuning after viewing TEST invalidates independent TEST claims"
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    ds=IEMOCAPDataset(train=False)
    keys=[str(k) for k in ds.keys]
    if (not keys or len(set(keys))!=len(keys) or
            any(not k.startswith("Ses05") for k in keys)):
        raise ValueError("Expected official IEMOCAP TEST to be session 5 only")
    loader=DataLoader(ds,batch_size=32,shuffle=False,
                      collate_fn=ds.collate_fn,num_workers=0)
    results=[]
    ref_y=None
    ref_groups=None
    for item in manifest:
        arm=item["arm"]
        ckpt=torch.load(item["checkpoint"],map_location="cpu",weights_only=True)
        if sha256(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed since TEST preflight")
        model=build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"],strict=True)
        y,p,groups=predict_valid(model,loader,device)
        if ref_y is None:
            ref_y,ref_groups=y,groups
        elif y!=ref_y or groups!=ref_groups:
            raise RuntimeError("Non-aligned TEST predictions/groups across arms")
        measured=test_metrics(y,p)
        subgroup=compute_report(y,p,groups,6)
        if measured["utterances"] != subgroup["all"]["n"]:
            raise RuntimeError("inconsistent TEST utterance count")
        if abs(measured["weighted_f1"]-subgroup["all"]["weighted_f1"])>.025:
            raise RuntimeError("two weighted F1 implementations disagree")
        print(f"TEST {arm:>9}: ACC={measured['accuracy']:.2f}% "
              f"W-F1={measured['weighted_f1']:.2f}% "
              f"Macro-F1={measured['macro_f1']:.2f}%",flush=True)
        results.append({
            "arm":arm,
            "checkpoint":item["checkpoint"],
            "checkpoint_sha256":item["checkpoint_sha256"],
            "selected_epoch":item["selected_epoch"],
            "dev_weighted_f1":item["validation_f1"],
            "test":measured,
            "test_groups":subgroup,
            "predictions":list(map(int,p)),
        })
        del model,ckpt
        if device.type=="cuda":
            torch.cuda.empty_cache()
    changes={}
    names={r["arm"]:r["predictions"] for r in results}
    for left,right in (("reaction","original"),
                       ("reaction","self"),("self","original")):
        changes[f"{left}_vs_{right}"]=paired_changes(
            ref_y,names[left],names[right]
        )
    report={
        "protocol":{
            "purpose":"EXPLORATORY_TEST_LOOK: consumes independent TEST status for subsequent V11 redesign",
            "trained_sessions":[1,2,3],
            "checkpoint_selected_on_session":4,
            "official_test_session":5,
            "official_test_dialogues":len(keys),
            "official_test_utterances":len(ref_y),
            "backbone":"CSS; original SDT text/audio/visual features",
            "arms_compared":["original","self","reaction"],
            "all_arms_frozen_before_test":True,
            "no_training":True,
            "no_test_selection":True,
            "not_comparable_directly_to_paper_full_train_protocol":True,
        },
        "models":results,
        "paired_changes":changes,
        "labels":list(map(int,ref_y)),
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print("\nV11 EXPLORATORY OFFICIAL TEST — THREE MATCHED ARMS")
    for g in GROUPS:
        print(f"{g:<22} n={results[0]['test_groups'][g]['n']}")
        for r in results:
            x=r["test_groups"][g]["weighted_f1"]
            base=results[0]["test_groups"][g]["weighted_f1"]
            print(f"  {r['arm']:<12} W-F1={x:.2f}% delta-CSS={x-base:+.2f}pp")
    print("\nPer-class TEST results (ACC means per-class recall):")
    for cls in range(6):
        label=results[0]["test"]["class_metrics"][str(cls)]["name"]
        print(f"  class={cls} {label}")
        for r in results:
            c=r["test"]["class_metrics"][str(cls)]
            print(f"    {r['arm']:<10} N={c['support']:>4} "
                  f"ACC={c['accuracy_recall']:.2f}% F1={c['f1']:.2f}%")
    print("\nPaired correctness changes:",changes)
    print("\nSaved:",output)
    print("Do NOT choose a new method or hyperparameters based on this TEST.")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight",action="store_true",
                      help="validate all artifacts; absolutely no TEST access")
    mode.add_argument("--evaluate-test",action="store_true",
                      help="EXPLICITLY consume TEST for one exploratory 3-model comparison")
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    # This always runs first, even if TEST has been requested.
    manifest=preflight()
    if args.preflight:
        print("Preflight finished. No TEST scored.")
        return
    device=torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                        else "cuda")
    print("EXPLORATORY TEST requested; methods/checkpoints are locked.",flush=True)
    evaluate(manifest,device,args.output)


if __name__=="__main__":
    main()

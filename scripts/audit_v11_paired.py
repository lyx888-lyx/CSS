#!/usr/bin/env python3
"""V11: post-hoc paired analysis on already-saved session-4 VALID predictions.

NO TRAINING, NO CHECKPOINT OR TEST LOADING. Reconstruct per-dialogue
boundaries using only the original TRAIN partition of IEMOCAP.
This diagnoses the previously observed V11 +0.55pp overall vs CSS and
+0.29pp vs self, but is not a new holdout or test of generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import IEMOCAPDataset
from research.v10_emotion_text_features import session_split
from research.v11_paired_analysis import (
    GROUPS, PAIR_NAMES, bootstrap_dialogues, fixed_broken, summarize_groups
)
from scripts.compare_validation import group_labels


REPORT=Path("results/v11_iemocap_acoustic_reaction/paired/report.json")
OUTPUT=Path("results/v11_iemocap_acoustic_reaction/paired/v11_paired_uncertainty.json")


def parse_saved_report(path):
    data=json.loads(path.read_text(encoding="utf-8"))
    proto=data.get("protocol",{})
    if (proto.get("dataset") != "IEMOCAP"
        or proto.get("train_sessions") != [1,2,3]
        or proto.get("validation_session") != 4
        or proto.get("seed") != 61078
        or proto.get("epochs") != 200
        or proto.get("official_test_scored") is not False):
        raise ValueError("not the fixed original V11 200-epoch validation report")
    rows=data.get("models",[])
    if [m.get("mode") for m in rows] != ["original","self","reaction"]:
        raise ValueError("need exact original/self/reaction paired report")
    if any(m.get("scores",{}).get("all",{}).get("n")!=1512 for m in rows):
        raise ValueError("unexpected paired validation length")
    y=rows[0]["labels"]
    if (len(y)!=1512 or any(m["labels"]!=y or
           len(m["predictions"])!=len(y) for m in rows)):
        raise ValueError("unaligned or mismatched target utterances")
    return data,rows


def reconstruct_dialogues(rows):
    ds=IEMOCAPDataset(train=True)  # ORIGINAL TRAIN partition, NOT TEST
    tr,va=session_split(ds.keys)
    if len(tr)!=90 or len(va)!=30:
        raise ValueError("unexpected session split, abort")
    n=sum(len(ds.videoLabels[ds.keys[i]]) for i in va)
    if n!=1512:
        raise ValueError("unexpected session-4 turn count")

    pred={m["mode"]:np.asarray(m["predictions"],dtype=np.int64)
          for m in rows}
    y=np.asarray(rows[0]["labels"],dtype=np.int64)
    names=list(pred)
    all_dialogues=[]
    offset=0
    for i in va:
        vid=ds.keys[i]
        label=np.asarray(ds.videoLabels[vid],dtype=np.int64)
        spk=ds.videoSpeakers[vid]
        if len(label)!=len(spk) or not set(spk).issubset({"M","F"}):
            raise ValueError(f"invalid speakers/labels {vid}")
        size=len(label)
        if size==0 or not np.array_equal(y[offset:offset+size],label):
            raise ValueError(f"label or dialogue order mismatch: {vid}")
        tags=group_labels(label.tolist(),
                          [0 if x=="M" else 1 for x in spk])
        masks={
            group:np.asarray([
                group=="all" or group in tag for tag in tags
            ],dtype=bool)
            for group in GROUPS
        }
        all_dialogues.append({
            "dialogue_id":str(vid),
            "labels":label,
            "predictions":{
                name:arr[offset:offset+size].copy()
                for name,arr in pred.items()
            },
            "masks":masks,
        })
        offset+=size
    if offset!=len(y):
        raise ValueError("unconsumed validation predictions")
    return all_dialogues


def by_class(dialogues,group):
    out={}
    for cls in range(6):
        labels=[]
        p={name:[] for name in ("original","self","reaction")}
        for d in dialogues:
            selected=d["masks"][group]&(d["labels"]==cls)
            labels.extend(d["labels"][selected].tolist())
            for name in p:
                p[name].extend(d["predictions"][name][selected].tolist())
        y=np.asarray(labels,dtype=np.int64)
        pred={name:np.asarray(values,dtype=np.int64)
              for name,values in p.items()}
        out[str(cls)]={
            "n":int(len(y)),
            "correct":{
                name:int((values==y).sum())
                for name,values in pred.items()
            },
            "reaction_minus_original":fixed_broken(
                y,pred["reaction"],pred["original"]
            ),
            "reaction_minus_self":fixed_broken(
                y,pred["reaction"],pred["self"]
            ),
        }
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if not args.report.is_file():
        parser.error(f"missing report: {args.report}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    saved,models=parse_saved_report(args.report)
    dialogues=reconstruct_dialogues(models)
    point=summarize_groups(dialogues,["original","self","reaction"])
    for name in GROUPS:
        old=models[0]["scores"][name]
        if (old["n"]!=point[name]["n"]
            or abs(old["weighted_f1"]-point[name]["f1"]["original"])>0.025):
            raise RuntimeError(f"saved original CSS vs reconstructed group mismatch: {name}")
        for row in models[1:]:
            if abs(row["scores"][name]["weighted_f1"]-
                   point[name]["f1"][row["mode"]])>0.025:
                raise RuntimeError(f"saved result mismatch: {name} {row['mode']}")

    print("V11 PAIRED ERROR AUDIT | 30 SESSION-4 DEV DIALOGUES | NO TEST")
    for name in GROUPS:
        print(f"\nGroup={name}, n={point[name]['n']}")
        for pair,stats in point[name]["paired"].items():
            print(f"  {pair:<25} "
                  f"delta={stats['f1_delta_pp']:+.2f}pp "
                  f"fixed={stats['fixed']} broken={stats['broken']} "
                  f"top1_flips={stats['top1_flips']}")

    ci=bootstrap_dialogues(
        dialogues,groups=("all","changed","changed_after_other"),
        repetitions=1500,seed=20261121,
    )
    print("\nDescriptive percentile ranges: resample WHOLE DIALOGUES")
    for group in ci:
        for pair,stats in ci[group].items():
            print(f"  {group:<23} {pair:<25} "
                  f"[{stats['percentile_95_low_pp']:+.2f}, "
                  f"{stats['percentile_95_high_pp']:+.2f}]pp")
    print("NOTE: checkpoints were selected on THIS VALID cohort; these")
    print("      bootstrap ranges do not establish independent significance.")

    class_info={
        name:by_class(dialogues,name)
        for name in ("all","changed","changed_after_other")
    }
    print("\nTrue-class correctness changes: REACTION minus ORIGINAL")
    for group,rows in class_info.items():
        print(f"  {group}:")
        for cls,record in rows.items():
            changes=record["reaction_minus_original"]
            print(f"    class={cls} n={record['n']:>4} "
                  f"fixed={changes['fixed']:>3} "
                  f"broken={changes['broken']:>3}")
    output={
        "protocol":{
            "dataset":"IEMOCAP",
            "validation_session":4,
            "dialogues":len(dialogues),
            "utterances":1512,
            "previously_inspected_validation":True,
            "checkpoints_chosen_by_this_validation":True,
            "model_retraining":False,
            "official_test_loaded_or_scored":False,
            "dialogue_bootstrap_replicates":1500,
            "bootstrap_seed":20261121,
            "limitations":"Resampling already viewed VALID cannot confirm gains; class/subgroup findings exploratory."
        },
        "point":point,
        "dialogue_bootstrap":ci,
        "class_correctness":class_info,
        "source_report":str(args.report),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("\nSaved:",args.output)
    print("No new training and no official TEST.")


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""V14 diagnostic ONLY: is actual intervening B useful beyond A's history?

NO CSS 200-epoch training. NO official TEST access, prediction or scoring.
Runs fixed 4-fold leave-one-TRAIN-session-out classification probes:
  (A) A(current) + A(previous) + self change.
  (B) Same A plus actual intervening B.
  (C) Same A plus unrelated B from other dialogue / same session / same
      B speaker role (three prespecified random draws).
  (D) Real-B-trained probe evaluated with B replaced by the draw.

This compares the incremental information in B on matched A-B-A
examples, NOT final CSS or paper TEST W-F1. Session-4 is just one
of the four original TRAIN-based development folds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import IEMOCAPDataset
from research.v14_b_incremental_probe import (
    SESSIONS, build_aba_examples, evaluate_fold, select_partition,
    subgroup_metrics
)

ROOT=Path("results/v14_b_incremental_diagnostic")
SEEDS=(31,47,73)
ARMS=("a_only","real_b",
      "shuffled_b_0","shuffled_b_1","shuffled_b_2",
      "real_b_shuffled_eval_0","real_b_shuffled_eval_1",
      "real_b_shuffled_eval_2")


def preflight():
    ds=IEMOCAPDataset(train=True)
    ex=build_aba_examples(ds,strict=True)
    total=len(ex["label"])
    info={
        "dataset":"IEMOCAP",
        "original_train_sessions":list(SESSIONS),
        "original_train_dialogues":int(ex["train_dialogues"]),
        "original_train_utterances":int(ex["train_utterances"]),
        "aba_eligible":int(total),
        "aba_changed":int(ex["changed"].sum()),
        "aba_stable":int((~ex["changed"]).sum()),
        "by_session":{
            session:{
                "dialogues":int(ex["session_dialogue_counts"][session]),
                "eligible":int((ex["session"]==session).sum()),
                "changed":int(np.sum(
                    (ex["session"]==session)&ex["changed"]
                ))
            }
            for session in SESSIONS
        },
        "official_test_dataset_instantiated":False,
        "official_test_scored":False,
        "notes":"Underlying original TRAIN pickle includes TEST arrays; "
                "the diagnostic accesses only ds.keys from TRAIN",
    }
    print("V14 B-INCREMENTAL DIAGNOSTIC — TRAIN SESSION PREFLIGHT")
    print(json.dumps(info,indent=2,ensure_ascii=False),flush=True)
    if any(item["eligible"]<20 for item in info["by_session"].values()):
        raise RuntimeError("too few A-B-A examples for session-wise diagnostic")
    return ex,info


def select_smoke(ex,indices,max_n=64):
    ids=np.asarray(indices,dtype=np.int64)
    if len(ids)<=max_n:
        return ids
    positions=np.linspace(0,len(ids)-1,num=max_n,dtype=np.int64)
    return ids[positions]


def run_folds(ex,smoke=False):
    folds=[]
    pooled_y=[]
    pooled_changed=[]
    pooled_pred={arm:[] for arm in ARMS}
    for held_session in (("Ses04",) if smoke else SESSIONS):
        train_sessions=[x for x in SESSIONS if x!=held_session]
        train_ids=select_partition(ex,train_sessions)
        valid_ids=select_partition(ex,(held_session,))
        if smoke:
            train_ids=np.concatenate([
                select_smoke(ex,select_partition(ex,(s,)))
                for s in train_sessions
            ])
            valid_ids=select_smoke(ex,valid_ids)
        if not len(train_ids) or not len(valid_ids):
            raise RuntimeError("empty fold")
        if not all(ex["session"][train_ids]!=held_session):
            raise RuntimeError("held-out session leakage")
        labels=ex["label"][valid_ids]
        changed=ex["changed"][valid_ids]
        print(f"\nV14 fold held={held_session} train_ABA={len(train_ids)} "
              f"valid_ABA={len(valid_ids)} "
              f"changed={int(changed.sum())}",flush=True)

        predictions,donor_checks=evaluate_fold(
            ex,train_ids,valid_ids,shuffle_seeds=SEEDS
        )
        if set(predictions)!=set(ARMS):
            raise RuntimeError("missing fixed arm predictions")
        measured={
            arm:subgroup_metrics(labels,predictions[arm],changed)
            for arm in ARMS
        }
        for arm in ARMS:
            pooled_pred[arm].extend(predictions[arm].astype(int).tolist())
        pooled_y.extend(labels.astype(int).tolist())
        pooled_changed.extend(changed.astype(bool).tolist())
        def number(arm):
            return measured[arm]["all"]["weighted_f1"]
        shuffled=[number(f"shuffled_b_{i}") for i in range(len(SEEDS))]
        probed=[number(f"real_b_shuffled_eval_{i}") for i in range(len(SEEDS))]
        print(f"  A-only W-F1={number('a_only'):.2f}% "
              f"Real B={number('real_b'):.2f}% "
              f"shuffle-trained mean={np.mean(shuffled):.2f}% "
              f"Real-B model / shuffled inputs mean={np.mean(probed):.2f}%",
              flush=True)
        folds.append({
            "held_out_original_train_session":held_session,
            "train_aba_examples":int(len(train_ids)),
            "valid_aba_examples":int(len(valid_ids)),
            "valid_dialogues":int(len(set(ex["dialogue"][valid_ids]))),
            "all_predictions_aligned":True,
            "modes":measured,
            "paired_delta_real_minus_a_pp":float(
                number("real_b")-number("a_only")
            ),
            "paired_delta_real_minus_mean_shuffled_pp":float(
                number("real_b")-np.mean(shuffled)
            ),
            "donor_checks":donor_checks,
        })
    pooled_y=np.asarray(pooled_y,dtype=np.int64)
    pooled_changed=np.asarray(pooled_changed,dtype=bool)
    pooled={
        arm:subgroup_metrics(
            pooled_y,np.asarray(pooled_pred[arm],dtype=np.int64),
            pooled_changed
        )
        for arm in ARMS
    }
    return folds,pooled,pooled_y,pooled_changed,pooled_pred


def print_decision(folds,pooled):
    score=lambda arm:pooled[arm]["all"]["weighted_f1"]
    a,real=score("a_only"),score("real_b")
    shuf=np.mean([score(f"shuffled_b_{i}") for i in range(len(SEEDS))])
    replaced=np.mean([
        score(f"real_b_shuffled_eval_{i}") for i in range(len(SEEDS))
    ])
    print("\nV14 TRAIN-ONLY OUT-OF-SESSION DIAGNOSTIC")
    print(f"  A-only             W-F1={a:.2f}%")
    print(f"  Real B             W-F1={real:.2f}% (vs A {real-a:+.2f} pp)")
    print(f"  Shuffled B mean    W-F1={shuf:.2f}% "
          f"(real minus shuffled {real-shuf:+.2f} pp)")
    print(f"  Real B, B replaced W-F1={replaced:.2f}% "
          f"(real minus replaced {real-replaced:+.2f} pp)")
    print("\nPer-held-out-session real B advantage:")
    for row in folds:
        print(f"  {row['held_out_original_train_session']}: "
              f"real-A={row['paired_delta_real_minus_a_pp']:+.2f}pp; "
              f"real-shuffled={row['paired_delta_real_minus_mean_shuffled_pp']:+.2f}pp")
    print("\nAnalysis-only changed-turn subset W-F1:")
    for arm in ("a_only","real_b","shuffled_b_0",
                "shuffled_b_1","shuffled_b_2"):
        m=pooled[arm]["changed"]
        print(f"  {arm:<15} n={m['n']} W-F1={m['weighted_f1']:.2f}%")
    consistent=sum(
        row["paired_delta_real_minus_a_pp"]>0
        and row["paired_delta_real_minus_mean_shuffled_pp"]>0
        for row in folds
    )
    print(f"\nReal B beat BOTH A-only and mean shuffled B in "
          f"{consistent}/{len(folds)} held-out original TRAIN sessions.")
    screening_pass=(
        len(folds)==4
        and real-a>=0.50
        and real-shuf>=0.50
        and real-replaced>=0.50
        and consistent>=3
    )
    print("V14 prespecified screening:",(
        "POSITIVE SIGNAL (NOT proof of a novel method)"
        if screening_pass else
        "NO SUFFICIENT SIGNAL under this fixed linear probe"
    ))
    print("This is exploratory linear-probe evidence, NOT CSS accuracy,")
    print("NOT official TEST, NOT proof of emotional causality,")
    print("and NOT sufficient to claim top-conference novelty.")
    return bool(screening_pass), int(consistent)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mx=p.add_mutually_exclusive_group()
    mx.add_argument("--preflight",action="store_true")
    mx.add_argument("--smoke",action="store_true")
    p.add_argument("--output-dir",type=Path,default=ROOT)
    args=p.parse_args()
    out=args.output_dir/("smoke.json" if args.smoke else "report.json")
    if not args.preflight and out.exists():
        p.error(f"Refusing to overwrite {out}; preserve the diagnostic")
    ex,info=preflight()
    if args.preflight:
        print("Preflight complete. No classifier trained, no TEST scored.")
        return
    folds,pooled,y,changed,pred=run_folds(ex,smoke=args.smoke)
    screen_passed,positive_folds=print_decision(folds,pooled)
    report={
        "prespecified_screen":{
            "passed":screen_passed,
            "positive_folds":positive_folds,
            "required_positive_folds":3,
            "min_pooled_real_vs_a_pp":0.50,
            "min_pooled_real_vs_shuffled_pp":0.50,
            "min_pooled_real_vs_replaced_pp":0.50,
            "is_final_css_result":False,
        },
        "protocol":{
            "dataset":"IEMOCAP",
            "input_partition":"OFFICIAL_TRAIN_ONLY",
            "original_train_sessions":list(SESSIONS),
            "folds":"Leave one original TRAIN session out",
            "smoke":bool(args.smoke),
            "smoke_only_held_out_session":"Ses04" if args.smoke else None,
            "original_training_data_removed_for_final_CSS":False,
            "official_test_scored":False,
            "official_test_dataset_instantiated":False,
            "no_gold_previous_emotion_input":True,
            "current_gold_emotion_used_only_for_training_target":True,
            "gold_emotion_change_used_only_for_posthoc_analysis":True,
            "feature_projection":"fixed random Rademacher text96/audio96/visual32",
            "feature_scaling":"fit only within each fold TRAIN",
            "classifier":"balanced L2 multinomial logistic C=0.1 lbfgs 350 iters",
            "shuffle_seeds":list(SEEDS),
            "shuffle_donor":"within fold partition, same session, same B role, different dialogue; prefer same intervening-B-count bin",
            "diagnostic_not_final_css":True,
            "limitations":"Linear, lossy-projection diagnostic: even a negative "
                           "result cannot rule out nonlinear B signal. "
                           "Session1-4 folds already part of development. "
                           "No causal or untouched-TEST claim.",
        },
        "preflight":info,
        "folds":folds,
        "pooled_metrics":pooled,
        "per_utterance":{
            "labels":y.astype(int).tolist(),
            "changed_analysis_only":changed.astype(bool).tolist(),
            "predictions":pred,
        },
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("\nSaved:",out,flush=True)


if __name__=="__main__":
    main()

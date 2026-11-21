#!/usr/bin/env python3
"""V15 nonlinear conditional B-information diagnostic, TRAIN ONLY.

Read original IEMOCAP TRAIN Sessions1–4; never load/score official
TEST Session5. Compare frozen A-only against equal-capacity residual
controls Self (no B), Real B, Shuffled B, and Real B with replacement.
Fixed four held-out-TRAIN-session folds, three independently fixed
training seeds, and predetermined no-test decision criteria.

This is a lightweight probe, NOT CSS's official 75.28% benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataloader import IEMOCAPDataset
from research.v14_b_incremental_probe import (
    SESSIONS, build_aba_examples, select_partition, subgroup_metrics
)
from research.v15_conditional_corrector import (
    SEEDS, DONOR_SEEDS, ARMS, BASE_EPOCHS, CORRECTION_EPOCHS,
    BATCH_SIZE, HIDDEN, BRANCH_DIM, LR, WEIGHT_DECAY, ANCHOR,
    MAX_CORRECTION, prepare_fold, run_one_seed
)

OUT = Path("results/v15_conditional_cross_speaker")
ALL = ("a_only", "self", "real_b", "shuffled_b", "real_b_replaced")


def data_preflight():
    ds = IEMOCAPDataset(train=True)
    ex = build_aba_examples(ds, strict=True)
    if ex["eligible_examples"] != 3751 or int(ex["changed"].sum()) != 1166:
        raise ValueError(
            "Training A-B-A fingerprint differs from the previous verified "
            "3751 eligible / 1166 changed. Stop and check feature provenance."
        )
    report = {
        "official_train_dialogues": int(ex["train_dialogues"]),
        "official_train_utterances": int(ex["train_utterances"]),
        "train_sessions": list(SESSIONS),
        "dialogues_by_session": ex["session_dialogue_counts"],
        "aba_eligible": int(len(ex["label"])),
        "aba_changed": int(ex["changed"].sum()),
        "aba_stable": int((~ex["changed"]).sum()),
        "official_test_loader_created": False,
        "official_test_scored": False,
        "feature_source": "original TRAIN SDT multimodal features, fixed V14 projection",
    }
    print("V15 TRAIN-ONLY preflight:", json.dumps(report, ensure_ascii=False), flush=True)
    return ex, report


def sample_smoke(ex, ids, n=64):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= n:
        return ids
    return ids[np.linspace(0, len(ids)-1, n, dtype=np.int64)]


def fold_ids(ex, held, smoke=False):
    train_parts = [select_partition(ex, (s,)) for s in SESSIONS if s != held]
    va = select_partition(ex, (held,))
    if smoke:
        train_parts = [sample_smoke(ex, part) for part in train_parts]
        va = sample_smoke(ex, va)
    tr = np.concatenate(train_parts)
    if (len(tr) == 0 or len(va) == 0 or
            np.any(ex["session"][tr] == held) or
            np.any(ex["session"][va] != held)):
        raise RuntimeError("fold session separation failed")
    return tr, va


def score_bundle(y, changed, predictions):
    return {arm: subgroup_metrics(y, predictions[arm], changed)
            for arm in ALL}


def wf1(metrics, arm):
    return float(metrics[arm]["all"]["weighted_f1"])


def fold_advantage(metrics):
    return {
        "real_minus_frozen_a_pp": wf1(metrics,"real_b")-wf1(metrics,"a_only"),
        "real_minus_self_pp": wf1(metrics,"real_b")-wf1(metrics,"self"),
        "real_minus_shuffled_pp": wf1(metrics,"real_b")-wf1(metrics,"shuffled_b"),
        "real_minus_replaced_pp": wf1(metrics,"real_b")-wf1(metrics,"real_b_replaced"),
    }


def decide(seed_reports, smoke=False):
    """Fixed non-tunable screen: REAL better than ALL controls."""
    differences=[r["pooled_deltas_pp"] for r in seed_reports]
    mean={key:float(np.mean([x[key] for x in differences]))
          for key in differences[0]}
    positive_seeds=sum(
        r["positive_sessions"] >= 3 for r in seed_reports
    )
    all_seed_positive=all(
        d["real_minus_frozen_a_pp"]>0
        and d["real_minus_self_pp"]>0
        and d["real_minus_shuffled_pp"]>0 for d in differences
    )
    passed=(not smoke and len(seed_reports)==3
            and all_seed_positive and positive_seeds>=2
            and all(value >= 0.50 for value in mean.values()))
    return {
        "passed":bool(passed),
        "smoke_ineligible":bool(smoke),
        "mean_deltas_pp":mean,
        "seeds_with_at_least_3_of_4_positive_sessions":int(positive_seeds),
        "require_three_seeds_pooled_positive":True,
        "require_at_least_two_seeds_with_three_positive_sessions":True,
        "min_mean_delta_to_each_control_pp":0.50,
        "warning":"Exploratory TRAIN-session screen only, not CSS nor independent TEST",
    }


def run(ex,device,smoke=False):
    seed_reports=[]
    seed_list=(SEEDS[0],) if smoke else SEEDS
    held_sessions=("Ses04",) if smoke else SESSIONS
    for nseed,seed in enumerate(seed_list):
        donor_seed=DONOR_SEEDS[nseed]
        pooled_y=[]
        pooled_changed=[]
        pooled={arm:[] for arm in ALL}
        folds=[]
        for session in held_sessions:
            tr,va=fold_ids(ex,session,smoke)
            fold=prepare_fold(ex,tr,va,donor_seed)
            predictions,diagnostics=run_one_seed(
                fold,seed,device,smoke=smoke
            )
            if set(predictions)!=set(ALL):
                raise RuntimeError("required matched controls missing")
            if any(len(predictions[arm])!=len(va) for arm in ALL):
                raise RuntimeError("prediction / sample length mismatch")
            metrics=score_bundle(fold["y_held"],fold["changed_held"],predictions)
            delta=fold_advantage(metrics)
            positives=(delta["real_minus_frozen_a_pp"]>0
                       and delta["real_minus_self_pp"]>0
                       and delta["real_minus_shuffled_pp"]>0)
            print(f"V15 seed={seed} held={session} "
                  f"train={len(tr)} held={len(va)} "
                  f"A={wf1(metrics,'a_only'):.2f}% "
                  f"Self={wf1(metrics,'self'):.2f}% "
                  f"RealB={wf1(metrics,'real_b'):.2f}% "
                  f"ShuffleB={wf1(metrics,'shuffled_b'):.2f}% "
                  f"ReplacedB={wf1(metrics,'real_b_replaced'):.2f}%",
                  flush=True)
            folds.append({
                "held_train_session":session,
                "train_n":int(len(tr)),"held_n":int(len(va)),
                "metrics":metrics,
                "deltas_pp":delta,
                "real_b_beats_frozen_self_and_shuffled":bool(positives),
                "prediction_flip_fraction":diagnostics,
                "gold_labels_used_only_for_training_or_scoring":True,
            })
            pooled_y.extend(fold["y_held"].tolist())
            pooled_changed.extend(fold["changed_held"].tolist())
            for arm in ALL:
                pooled[arm].extend(predictions[arm].tolist())
        pooled_metrics=score_bundle(
            np.asarray(pooled_y,dtype=np.int64),
            np.asarray(pooled_changed,dtype=bool),
            {arm:np.asarray(pooled[arm],dtype=np.int64) for arm in ALL},
        )
        pooled_delta=fold_advantage(pooled_metrics)
        print(f"\nV15 seed={seed} POOLED OOF W-F1: "
              f"A={wf1(pooled_metrics,'a_only'):.2f} "
              f"Self={wf1(pooled_metrics,'self'):.2f} "
              f"RealB={wf1(pooled_metrics,'real_b'):.2f} "
              f"ShuffleB={wf1(pooled_metrics,'shuffled_b'):.2f} "
              f"Replaced={wf1(pooled_metrics,'real_b_replaced'):.2f}\n",
              flush=True)
        seed_reports.append({
            "train_seed":int(seed),"negative_control_donor_seed":int(donor_seed),
            "folds":folds,
            "pooled_metrics":pooled_metrics,
            "pooled_deltas_pp":pooled_delta,
            "positive_sessions":int(sum(
                x["real_b_beats_frozen_self_and_shuffled"] for x in folds
            )),
            "pooled_labels":list(map(int,pooled_y)),
            "pooled_changed_for_posthoc_only":list(map(bool,pooled_changed)),
            "pooled_predictions":pooled,
        })
    return seed_reports,decide(seed_reports,smoke)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight",action="store_true")
    modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    if torch.get_num_threads()>4:
        torch.set_num_threads(4)
    path=args.output_dir/("smoke.json" if args.smoke else "report.json")
    if not args.preflight and path.exists():
        parser.error(f"Refusing to overwrite previous experimental report: {path}")
    ex,preflight=data_preflight()
    if args.preflight:
        print("V15 preflight passed; no fitting or TEST scoring.")
        return
    device=torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"V15 device={device}, BASE_EPOCHS={2 if args.smoke else BASE_EPOCHS}, "
          f"CORRECTION_EPOCHS={2 if args.smoke else CORRECTION_EPOCHS}",flush=True)
    reports,screen=run(ex,device,smoke=args.smoke)
    result={
        "protocol":{
            "dataset":"IEMOCAP",
            "partition":"ORIGINAL TRAIN ONLY (Ses01-Ses04)",
            "official_test_scored":False,
            "official_test_loader_created":False,
            "folds":"Leave-one-original-TRAIN-session-out",
            "smoke":bool(args.smoke),
            "feature_projection":"same fixed V14 random 224-D per utterance",
            "same_aba_samples_all_arms":True,
            "frozen_a_only_before_correction":True,
            "matched_correction_parameter_counts":True,
            "conditional_interaction":"hidden A x hidden B learned elementwise product",
            "risk_control":"confidence-scaled bounded delta logits and anchor penalty",
            "fixed_base_epochs":BASE_EPOCHS,
            "fixed_correction_epochs":CORRECTION_EPOCHS,
            "batch_size":BATCH_SIZE,
            "hidden":HIDDEN,
            "branch_dim":BRANCH_DIM,
            "lr":LR,"weight_decay":WEIGHT_DECAY,
            "anchor":ANCHOR,"max_correction":MAX_CORRECTION,
            "seeds":list(SEEDS),
            "donor_seeds":list(DONOR_SEEDS),
            "validation_checkpoint_selection":False,
            "no_gold_previous_emotion_model_inputs":True,
            "not_a_css_or_paper_comparable_score":True,
            "limitations":"Train-only small multimodal projected-feature probe, "
                          "not proof of causality, official TEST, or novelty. "
                          "Original IEMOCAP Session5 already exposed in V11-V13.",
        },
        "preflight":preflight,
        "seed_results":reports,
        "prespecified_screen":screen,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                    encoding="utf-8")
    print("\nV15 fixed TRAIN-only gate:",json.dumps(screen,indent=2),flush=True)
    print("Saved:",path,flush=True)


if __name__=="__main__":
    main()

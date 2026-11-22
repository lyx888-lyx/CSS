"""Pure functions for auditing the four ORIGINAL CSS classification heads.

Heads: text, audio, visual, fused. This is a DESCRIPTIVE audit:
an oracle that picks a correct head using gold labels is NOT deployable.
No learned selection, no test-tuned parameters, no gold labels as inputs.
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

HEADS=("text","audio","visual","fused")
CLASSES=("happy","sad","neutral","angry","excited","frustrated")


def validate_predictions(labels,preds):
    y=np.asarray(labels,dtype=np.int64)
    if y.ndim!=1 or len(y)==0 or np.any((y<0)|(y>=len(CLASSES))):
        raise ValueError("labels must be 1D nonempty IEMOCAP-6 class IDs")
    if set(preds)!=set(HEADS):
        raise ValueError(f"expected exactly {HEADS} head predictions")
    pp={}
    for name in HEADS:
        p=np.asarray(preds[name],dtype=np.int64)
        if p.shape!=y.shape or np.any((p<0)|(p>=len(CLASSES))):
            raise ValueError(f"invalid {name} predictions")
        pp[name]=p
    return y,pp


def metrics(y,p):
    return {
        "accuracy":round(float(accuracy_score(y,p)*100),4),
        "weighted_f1":round(float(f1_score(
            y,p,labels=list(range(6)),average="weighted",
            zero_division=0)*100),4),
        "macro_f1":round(float(f1_score(
            y,p,labels=list(range(6)),average="macro",
            zero_division=0)*100),4),
        "per_class_f1":{
            name:round(float(f1_score(
                y,p,labels=[i],average="macro",
                zero_division=0)*100),4)
            for i,name in enumerate(CLASSES)
        },
    }


def summarize(labels,preds,probabilities=None):
    y,p=validate_predictions(labels,preds)
    n=len(y)
    correct={name:(p[name]==y) for name in HEADS}
    fused_correct=correct["fused"]
    fused_wrong=~fused_correct
    other_any=np.logical_or.reduce([correct[h] for h in HEADS[:3]])
    fused_wrong_rescuable=fused_wrong&other_any
    fused_correct_disagree=fused_correct&np.logical_or.reduce(
        [p[h]!=p["fused"] for h in HEADS[:3]]
    )
    summaries={name:metrics(y,p[name]) for name in HEADS}
    report={
        "utterances":int(n),
        "per_head":summaries,
        "fused_error_count":int(fused_wrong.sum()),
        "any_single_head_correct_when_fused_wrong":int(
            fused_wrong_rescuable.sum()),
        "oracle_any_head_correct":int(
            np.logical_or(fused_correct,other_any).sum()),
        "oracle_any_head_accuracy_ceiling_pct":round(
            float(100*np.logical_or(fused_correct,other_any).mean()),4),
        "fused_correct_but_some_single_head_disagrees":int(
            fused_correct_disagree.sum()),
        "fused_correct_and_all_three_single_heads_wrong":int(
            (fused_correct&~other_any).sum()),
        "per_other_head":{},
        "fused_wrong_with_single_head_agreement":{},
        "confusion_matrix_fused":confusion_matrix(
            y,p["fused"],labels=list(range(6))
        ).astype(int).tolist(),
    }
    for name in HEADS[:3]:
        report["per_other_head"][name]={
            "fixable_fused_errors_by_oracle":int(
                (fused_wrong&correct[name]).sum()),
            "fused_correct_but_head_wrong":int(
                (fused_correct&~correct[name]).sum()),
            "both_wrong":int((fused_wrong&~correct[name]).sum()),
            "head_disagrees_with_fused":int(
                (p[name]!=p["fused"]).sum()),
        }
    for agree_k in range(4):
        same=np.stack([p[h]==p["fused"] for h in HEADS[:3]]).sum(axis=0)
        report["fused_wrong_with_single_head_agreement"][str(agree_k)]=int(
            np.sum(fused_wrong&(same==agree_k))
        )
    report["by_true_class"]={}
    for i,name in enumerate(CLASSES):
        cls=y==i
        report["by_true_class"][name]={
            "n":int(cls.sum()),
            "fused_correct":int((cls&fused_correct).sum()),
            "fused_wrong":int((cls&fused_wrong).sum()),
            "fused_wrong_any_head_correct":int(
                (cls&fused_wrong_rescuable).sum()),
            "fused_correct_but_any_other_disagrees":int(
                (cls&fused_correct_disagree).sum()),
        }
    if probabilities is not None:
        if set(probabilities)!=set(HEADS):
            raise ValueError("probability dict must contain all heads")
        report["probability_diagnostics"]={}
        for name in HEADS:
            prob=np.asarray(probabilities[name],dtype=np.float64)
            if prob.shape!=(n,6) or not np.isfinite(prob).all():
                raise ValueError(f"invalid probability array {name}")
            if np.max(np.abs(prob.sum(-1)-1))>2e-4:
                raise ValueError(f"{name} probability rows not normalized")
            confidence=prob.max(axis=-1)
            report["probability_diagnostics"][name]={
                "mean_confidence":round(float(confidence.mean()),6),
                "mean_confidence_on_fused_wrong":(
                    round(float(confidence[fused_wrong].mean()),6)
                    if fused_wrong.any() else None),
                "mean_confidence_on_fused_correct":(
                    round(float(confidence[fused_correct].mean()),6)
                    if fused_correct.any() else None),
            }
    report["interpretation_limits"]=(
        "All oracle correct-head counts use labels for POSTHOC analysis. "
        "A deployable selector cannot know which head is correct. "
        "The four heads share context and training objectives; "
        "no independent test or improvement claim is implied."
    )
    return report

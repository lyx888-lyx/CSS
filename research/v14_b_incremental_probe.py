"""V14 TRAIN-only A-B-A incremental evidence diagnostic for CSS.

No neural-network training and NO official TEST inference/scoring.
Use only original IEMOCAP TRAIN dialogues (Sessions 1–4).
A-only, real-B and shuffled-B arms use identical A-B-A target turns.
All shuffling is restricted to the same partition AND same Session,
B speaker role, and a different dialogue. Emotion labels are NEVER
features (only classification targets or post-prediction diagnostics).

This is an inexpensive linear probe with FIXED random feature maps.
A negative result does NOT prove absence of cross-speaker information
in a nonlinear neural model. A positive result is exploratory, not
proof of physical causation or publication-level novelty.
"""
from collections import Counter
import numpy as np

SESSIONS=("Ses01","Ses02","Ses03","Ses04")
FEATURE_DIMS={"text":(1024,96),"audio":(1582,96),"visual":(342,32)}
PROJECTION_SEED=20261121


def session_of(vid):
    name=str(vid)
    if len(name)<5 or name[:5] not in SESSIONS:
        raise ValueError(f"not an official IEMOCAP TRAIN session key: {name}")
    return name[:5]


def validate_train_session_counts(counts, n_dialogues):
    """Check official TRAIN coverage, not an invented equal-session split.

    The original CSS/IEMOCAP TRAIN has 120 dialogues across Ses01–04,
    but different sessions need not contain exactly 30 dialogues.
    Source-derived per-session counts must be retained and reported.
    """
    actual=set(counts)
    expected=set(SESSIONS)
    if (n_dialogues!=120 or actual!=expected or
            any(int(counts[s])<=0 for s in SESSIONS) or
            sum(int(counts[s]) for s in SESSIONS)!=n_dialogues):
        raise ValueError(
            "Expected 120 unique official TRAIN dialogues across "
            f"Ses01–Ses04 (unequal per-session counts allowed); got {dict(counts)}"
        )


def make_projection(seed=PROJECTION_SEED):
    """Fixed, label-free Rademacher maps shared across A and B."""
    rng=np.random.default_rng(seed)
    proj={}
    for modality,(raw,out) in FEATURE_DIMS.items():
        values=rng.integers(0,2,size=(raw,out),dtype=np.int8)
        proj[modality]=(values.astype(np.float32)*2.0-1.0)/np.sqrt(out)
    return proj


def _project_utterances(ds,vid,projection):
    n=len(ds.videoLabels[vid])
    fields=(("text",ds.videoText[vid]),
            ("audio",ds.videoAudio[vid]),
            ("visual",ds.videoVisual[vid]))
    result=[]
    for name,raw in fields:
        mat=np.asarray(raw,dtype=np.float32)
        dim=FEATURE_DIMS[name][0]
        if mat.shape!=(n,dim):
            raise ValueError(f"{vid}: {name} expected {(n,dim)}, got {mat.shape}")
        if not np.isfinite(mat).all():
            raise ValueError(f"{vid}: nonfinite {name}")
        result.append(mat @ projection[name])
    return np.concatenate(result,axis=1)


def build_aba_examples(ds,strict=True,projection=None):
    """Extract eligible A(prev)-B(interposed)-A(current) examples.

    Sample labels and true previous speaker labels are stored in
    metadata for scoring ONLY, never included in feature vectors.
    Within each conversation all current/previous/B features use the
    same non-supervised fixed random projection.
    """
    projection=make_projection() if projection is None else projection
    keys=list(ds.keys)
    if len(keys)!=len(set(keys)):
        raise ValueError("duplicate dialogue IDs")
    counts=Counter(session_of(vid) for vid in keys)
    if strict:
        validate_train_session_counts(counts, len(keys))
    rows=[]
    total=0
    for vid in keys:
        speakers=list(ds.videoSpeakers[vid])
        labels=np.asarray(ds.videoLabels[vid],dtype=np.int64)
        n=len(labels)
        total+=n
        if n!=len(speakers) or not set(speakers).issubset({"M","F"}):
            raise ValueError(f"invalid speaker/label metadata {vid}")
        if n==0 or (labels<0).any() or (labels>=6).any():
            raise ValueError(f"invalid emotion IDs {vid}")
        features=_project_utterances(ds,vid,projection)
        previous={}
        for idx,sp in enumerate(speakers):
            j=previous.get(sp)
            if j is not None:
                intervening=[k for k in range(j+1,idx)
                             if speakers[k]!=sp]
                if intervening:
                    other=features[intervening].mean(axis=0)
                    rows.append((
                        features[idx].copy(),features[j].copy(),other,
                        int(labels[idx]),bool(labels[idx]!=labels[j]),
                        session_of(vid),str(vid),
                        "F" if sp=="M" else "M",
                        min(3,len(intervening)),idx,j,
                    ))
            previous[sp]=idx
    if strict and total!=5810:
        raise ValueError(f"TRAIN utterance count {total} != 5810")
    if not rows:
        raise ValueError("no eligible A-B-A utterances")
    cols=list(zip(*rows))
    out={
        "current":np.stack(cols[0]).astype(np.float32),
        "previous":np.stack(cols[1]).astype(np.float32),
        "other":np.stack(cols[2]).astype(np.float32),
        "label":np.asarray(cols[3],dtype=np.int64),
        "changed":np.asarray(cols[4],dtype=bool),
        "session":np.asarray(cols[5],dtype="U5"),
        "dialogue":np.asarray(cols[6],dtype=str),
        "b_role":np.asarray(cols[7],dtype="U1"),
        "gap_bin":np.asarray(cols[8],dtype=np.int64),
        "turn_index":np.asarray(cols[9],dtype=np.int64),
        "previous_index":np.asarray(cols[10],dtype=np.int64),
        "train_dialogues":len(keys),"train_utterances":total,
        "eligible_examples":len(rows),
        "session_dialogue_counts":dict(counts),
    }
    return out


def select_partition(ex, sessions):
    allowed=set(sessions)
    return np.flatnonzero(np.isin(ex["session"],list(allowed)))


def shuffle_b_donors(ex,partition_indices,seed):
    """Within-partition negative control: same Session, B role, other dialogue.

    Prefer matching number-of-B-turns bin; allow a relaxed gap match
    only if an exact-bin different-dialogue donor does not exist.
    Never allow a donor from the held-out split, same dialogue, or
    different session/speaker role. No labels inform donor selection.
    """
    rng=np.random.default_rng(seed)
    ids=np.asarray(partition_indices,dtype=np.int64)
    if ids.ndim!=1 or len(ids)==0:
        raise ValueError("partition must be nonempty 1D indices")
    if len(set(map(int,ids)))!=len(ids):
        raise ValueError("partition indices cannot repeat")
    donors=np.empty(len(ids),dtype=np.int64)
    for row,idx in enumerate(ids):
        acceptable=ids[
            (ex["session"][ids]==ex["session"][idx])
            &(ex["b_role"][ids]==ex["b_role"][idx])
            &(ex["dialogue"][ids]!=ex["dialogue"][idx])
        ]
        exact=acceptable[
            ex["gap_bin"][acceptable]==ex["gap_bin"][idx]
        ]
        pool=exact if len(exact) else acceptable
        if not len(pool):
            raise ValueError(
                f"No other-dialogue same-session/speaker donor for "
                f"{ex['dialogue'][idx]} at {int(idx)}"
            )
        donors[row]=int(rng.choice(pool))
    return donors


def base_and_b_features(ex,indices,b_donors=None):
    """Return [A current, A previous, A change] and 3 B comparisons.

    The 'current' audio/text/vision is an observed inference input;
    its true label and the 'changed' ground truth are NEVER inputs.
    The B feature's representation dimension is identical for all arms.
    """
    ids=np.asarray(indices,dtype=np.int64)
    a=ex["current"][ids]
    prior=ex["previous"][ids]
    b=ex["other"][ids] if b_donors is None else ex["other"][
        np.asarray(b_donors,dtype=np.int64)
    ]
    if b.shape!=a.shape:
        raise ValueError("donor feature alignment error")
    a_design=np.concatenate([a,prior,a-prior],axis=1)
    b_design=np.concatenate([b,a-b,prior-b],axis=1)
    return a_design.astype(np.float32),b_design.astype(np.float32)


def subgroup_metrics(y,pred,changed,classes=6):
    from sklearn.metrics import accuracy_score, f1_score
    y=np.asarray(y,dtype=np.int64)
    pred=np.asarray(pred,dtype=np.int64)
    changed=np.asarray(changed,dtype=bool)
    if y.shape!=pred.shape or y.shape!=changed.shape or y.ndim!=1:
        raise ValueError("unaligned score inputs")
    out={}
    for name,mask in (
        ("all",np.ones(len(y),dtype=bool)),
        ("stable",~changed),("changed",changed)
    ):
        a,b=y[mask],pred[mask]
        out[name]={
            "n":int(len(a)),
            "accuracy":float(accuracy_score(a,b)*100) if len(a) else None,
            "weighted_f1":float(f1_score(
                a,b,labels=list(range(classes)),
                average="weighted",zero_division=0)*100) if len(a) else None,
            "macro_f1":float(f1_score(
                a,b,labels=list(range(classes)),
                average="macro",zero_division=0)*100) if len(a) else None,
        }
    return out


def evaluate_fold(ex,train_indices,valid_indices,shuffle_seeds=(31,47,73)):
    """One held-out full TRAIN session. No tuning and no TEST."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    train_indices=np.asarray(train_indices,dtype=np.int64)
    valid_indices=np.asarray(valid_indices,dtype=np.int64)
    if not len(train_indices) or not len(valid_indices):
        raise ValueError("empty training or validation subset")
    if set(train_indices.tolist()) & set(valid_indices.tolist()):
        raise ValueError("fold train/valid overlap")
    if set(ex["dialogue"][train_indices]) & set(ex["dialogue"][valid_indices]):
        raise ValueError("dialogue leakage across split")
    if set(ex["session"][train_indices]) & set(ex["session"][valid_indices]):
        raise ValueError("session leakage across split")

    a_tr,b_tr=base_and_b_features(ex,train_indices)
    a_va,b_va=base_and_b_features(ex,valid_indices)
    # Fit two scalers on TRAIN only. Reuse identical transforms for
    # every negative control in this fold.
    a_scale=StandardScaler().fit(a_tr)
    b_scale=StandardScaler().fit(b_tr)
    a_tr,a_va=a_scale.transform(a_tr),a_scale.transform(a_va)
    b_tr,b_va=b_scale.transform(b_tr),b_scale.transform(b_va)
    zeros_tr=np.zeros_like(b_tr)
    zeros_va=np.zeros_like(b_va)
    y_tr=ex["label"][train_indices]
    if len(np.unique(y_tr))<2:
        raise ValueError("fold training labels have fewer than 2 classes")
    results={}

    def fit_predict(name,xtr,xva):
        model=LogisticRegression(
            C=0.1,class_weight="balanced",solver="lbfgs",
            max_iter=350,tol=1e-3
        )
        model.fit(xtr,y_tr)
        return model,model.predict(xva).astype(np.int64)

    design_a_train=np.concatenate([a_tr,zeros_tr],axis=1)
    design_a_valid=np.concatenate([a_va,zeros_va],axis=1)
    _,results["a_only"]=fit_predict("a_only",design_a_train,design_a_valid)

    design_real_train=np.concatenate([a_tr,b_tr],axis=1)
    design_real_valid=np.concatenate([a_va,b_va],axis=1)
    real_model,results["real_b"]=fit_predict(
        "real_b",design_real_train,design_real_valid
    )
    donor_stats=[]
    for repeat,seed in enumerate(shuffle_seeds):
        tr_donors=shuffle_b_donors(ex,train_indices,seed)
        va_donors=shuffle_b_donors(ex,valid_indices,seed+100000)
        _,b_shuf_tr=base_and_b_features(ex,train_indices,tr_donors)
        _,b_shuf_va=base_and_b_features(ex,valid_indices,va_donors)
        bs_tr=b_scale.transform(b_shuf_tr)
        bs_va=b_scale.transform(b_shuf_va)
        tr_shuf=np.concatenate([a_tr,bs_tr],axis=1)
        va_shuf=np.concatenate([a_va,bs_va],axis=1)
        _,results[f"shuffled_b_{repeat}"]=fit_predict(
            f"shuffled_b_{repeat}",tr_shuf,va_shuf
        )
        # Identical REAL model; destroy B association at evaluation
        # to check whether REAL B was actually used by this probe.
        results[f"real_b_shuffled_eval_{repeat}"]=(
            real_model.predict(va_shuf).astype(np.int64)
        )
        donor_stats.append({
            "seed":int(seed),
            "train_examples":int(len(train_indices)),
            "valid_examples":int(len(valid_indices)),
            "valid_different_dialogue":bool(np.all(
                ex["dialogue"][valid_indices]!=ex["dialogue"][va_donors]
            )),
            "valid_same_session":bool(np.all(
                ex["session"][valid_indices]==ex["session"][va_donors]
            )),
            "valid_same_b_role":bool(np.all(
                ex["b_role"][valid_indices]==ex["b_role"][va_donors]
            )),
        })
    return results,donor_stats

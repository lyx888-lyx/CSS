#!/usr/bin/env python3
"""V10: session-disjoint IEMOCAP frozen emotion-text feature comparison.

Official TEST not evaluated. CSS architecture/trainer/PGM unchanged except
for zero-initialized extra input channels in both feature-enhanced arms.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

import train_IEMOCAP as trainer
from dataloader import IEMOCAPDataset
from model import MaskedNLLLoss, MaskedKLDivLoss
from scripts.train_research import build_model
from scripts.compare_validation import predict_valid, compute_report
from research.v10_emotion_text_features import (
    MODEL_ID, REVISION, EXTRA_DIM, MODES, session_split, text_inputs,
    extract_text_features, FeatureDataset, expand_text_projection
)

SEED = 61078
EPOCHS = 200
DATA = Path("data/iemocap_multimodal_features.pkl")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(ds):
    tr, va = session_split(ds.keys)
    for vid in ds.keys:
        n = len(ds.videoLabels[vid])
        if not n or len(ds.videoSentence[vid]) != n or len(ds.videoSpeakers[vid]) != n:
            raise ValueError(f"raw text/label/speaker count mismatch: {vid}")
        text_inputs(ds.videoSentence[vid], ds.videoSpeakers[vid])
    print("V10 fixed IEMOCAP development protocol: sessions 1-3 TRAIN, session 4 VALID")
    print(f"TRAIN dialogues={len(tr)} utterances={sum(len(ds.videoLabels[ds.keys[i]]) for i in tr)}")
    print(f"VALID dialogues={len(va)} utterances={sum(len(ds.videoLabels[ds.keys[i]]) for i in va)}")
    print("Official TEST inference/scoring: NO; session 4 is not official TEST.")
    return tr, va


def get_cache(ds, path, device, local_only, batch_size):
    metadata = dict(model=MODEL_ID, revision=REVISION, source_sha256=file_hash(DATA),
                    dialogue_ids=list(ds.keys), hidden_size=EXTRA_DIM, schema=1)
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["metadata"] != metadata:
            raise ValueError("V10 cache source/revision mismatch; use new cache path")
        cache = data["features"]
        print("Reusing verified local feature cache:", path)
    else:
        cache = extract_text_features(ds, device, batch_size, local_only)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"metadata":metadata, "features":cache}, tmp)
        tmp.replace(path)
        print("Saved:", path)
    for mode in ("current","speaker_context"):
        if set(cache[mode]) != set(ds.keys):
            raise ValueError("cache dialogue keys differ")
        for vid in ds.keys:
            x = cache[mode][vid]
            if x.shape != (len(ds.videoLabels[vid]), EXTRA_DIM) or not torch.isfinite(x).all():
                raise ValueError(f"cache shape/value error: {mode}/{vid}")
    return cache, metadata


def config(epochs):
    return dict(dataset="IEMOCAP", seed=SEED, split_seed=None, epochs=epochs,
                lr=1e-4, order=3, temp=1, audio=1582, classes=6, speakers=2,
                batch_size=32, hidden_dim=1024, heads=8, rank=16, dropout=.5,
                l2=1e-5, speaker_mode="off", time_decay=True, max_lag=None,
                precontext_mode="off", v52_pgm_mode="original",
                v61_spf_mode="full", v8_spf_mode="off")


def train_arm(ds, cache, tr, va, mode, device, root, epochs):
    cfg = config(epochs)
    cfg["v10_feature_mode"] = mode
    cfg["v10_validation_session"] = 4
    path = root / f"{mode}_valid_best.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    wrapped = FeatureDataset(ds, mode, cache)
    train_loader = DataLoader(Subset(wrapped,tr), batch_size=32, shuffle=True,
                              collate_fn=wrapped.collate_fn, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED+1000))
    valid_loader = DataLoader(Subset(wrapped,va), batch_size=32, shuffle=False,
                              collate_fn=wrapped.collate_fn, num_workers=0)
    trainer.set_seed(SEED)
    model = build_model(cfg)
    if mode != "original":
        expand_text_projection(model)
    model.to(device)
    # Reset RNG after extra projection initialization; common dropout
    # and shuffle streams begin identically across all arms.
    trainer.set_seed(SEED)
    trainer.cuda = device.type == "cuda"
    trainer.args = argparse.Namespace(tensorboard=False)
    weights = torch.tensor([1/.086747,1/.144406,1/.227883,
                            1/.160585,1/.127711,1/.252668],
                           device=device,dtype=torch.float32)
    loss_fn = MaskedNLLLoss(weights)
    kl_fn = MaskedKLDivLoss()
    opt = torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["l2"])
    best = (-float("inf"), -float("inf"))
    root.mkdir(parents=True,exist_ok=True)
    print(f"\n=== V10 {mode} input_dim={model.textf_input.in_channels} ===",flush=True)
    for e in range(epochs):
        loss, acc, _, _, _, train_f1 = trainer.train_or_eval_model(
            model,loss_fn,kl_fn,train_loader,e,optimizer=opt,train=True)
        valid_acc, _, _, _, valid_f1 = trainer.evaluate_loaded_model(model,valid_loader)
        # sklearn metrics may return numpy.float64. Save only native Python
        # metadata so torch.load(weights_only=True) can deserialize safely.
        valid_acc, valid_f1 = float(valid_acc), float(valid_f1)
        if (valid_f1,valid_acc) > best:
            best=(valid_f1,valid_acc)
            torch.save(dict(model_state_dict=model.state_dict(),config=cfg,
                            epoch=e+1,validation_f1=valid_f1,
                            validation_accuracy=valid_acc,
                            train_dialogue_indices=tr,valid_dialogue_indices=va,
                            selection="validation_weighted_f1"),path)
        print(f"{mode} epoch={e+1:03d} train_loss={loss:.4f} train_f1={train_f1:.2f} "
              f"valid_f1={valid_f1:.2f} best={best[0]:.2f}",flush=True)
    saved=torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(saved["model_state_dict"],strict=True)
    y,p,tags=predict_valid(model,valid_loader,device)
    report=compute_report(y,p,tags,6)
    if abs(report["all"]["weighted_f1"]-best[0])>.025:
        raise RuntimeError("checkpoint re-inference mismatch")
    result=dict(mode=mode,checkpoint=str(path),selected_epoch=saved["epoch"],
                scores=report,labels=y,predictions=p)
    del model,opt,saved
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("results/v10_iemocap/emotion_text_cache.pt"))
    p.add_argument("--output-dir",type=Path,default=Path("results/v10_iemocap/paired"))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--smoke",action="store_true",help="one epoch in speaker_context arm, separate output")
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--local-files-only",action="store_true")
    p.add_argument("--encoder-batch-size",type=int,default=32)
    args=p.parse_args()
    if args.encoder_batch_size<1:
        p.error("encoder batch size must be positive")
    if args.prepare_only and args.smoke:
        p.error("prepare-only and smoke are mutually exclusive")
    ds=IEMOCAPDataset(train=True)
    tr,va=preflight(ds)
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cache,meta=get_cache(ds,args.cache,device,args.local_files_only,args.encoder_batch_size)
    if args.prepare_only:
        return
    modes=("speaker_context",) if args.smoke else MODES
    epochs=1 if args.smoke else EPOCHS
    root=args.output_dir/"smoke" if args.smoke else args.output_dir
    if any((root/f"{mode}_valid_best.pt").exists() for mode in modes) or (root/"report.json").exists():
        raise FileExistsError(f"Existing V10 outputs in {root}; choose new --output-dir")
    if device.type=="cuda":
        torch.cuda.empty_cache()
    rows=[train_arm(ds,cache,tr,va,mode,device,root,epochs) for mode in modes]
    if any(r["labels"]!=rows[0]["labels"] for r in rows):
        raise RuntimeError("unaligned validation labels across arms")
    result=dict(protocol=dict(dataset="IEMOCAP",train_sessions=[1,2,3],
                              validation_session=4,seed=SEED,epochs=epochs,
                              train_dialogue_indices=tr,valid_dialogue_indices=va,
                              official_test_scored=False,smoke=args.smoke,
                              external_model=MODEL_ID,revision=REVISION,
                              external_features_trainable=False,
                              source_sha256=meta["source_sha256"]),models=rows)
    (root/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("\nV10 IEMOCAP FEATURE PILOT | SESSION 4 VALID | NO TEST")
    for group in ("all","first","stable","changed","stable_after_other","changed_after_other"):
        base=rows[0]["scores"][group]["weighted_f1"]
        print(f"{group} n={rows[0]['scores'][group]['n']}")
        for r in rows:
            value=r["scores"][group]["weighted_f1"]
            if value is None or base is None:
                print(f"  {r['mode']}: unavailable")
            else:
                print(f"  {r['mode']:<17} W-F1={value:.2f}% delta-vs-first={value-base:+.2f}pp")
    print("Selected epochs:",[(r["mode"],r["selected_epoch"]) for r in rows])
    print("Saved:",root/"report.json")
    print("Single development session/seed; external feature gains are NOT novelty proof.")


if __name__=="__main__":
    main()

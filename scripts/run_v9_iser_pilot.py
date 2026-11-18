#!/usr/bin/env python3
"""V9 fixed frozen-CSS ISER pilot. Existing development VALID only.

No TEST inference, no backbone training, no repeated VALID selection of
the adapter epoch. All heads use fixed training budgets. The training
posteriors are in-sample CSS predictions, NOT cross-fitted predictions.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from research.iser_transition import (
    MODES, LAMBDA, SelfTransition, InteractionTransition, history_features,
    normalize_modalities, refine_logits, shuffled_train_donors
)
from scripts.compare_validation import compute_report, group_labels, read_checkpoint
from scripts.train_research import build_model, v61_disjoint_spf_ablation_split_indices

SEED = 10068
SPLIT = 36427
SELF_EPOCHS = 50
HEAD_EPOCHS = 40
HEAD_SEED = 90421
BATCH = 256
AUX = 0.5
ROOT = Path("results/v61_spf_paired_split36427_trainseed10068/checkpoints")
GROUPS = ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other")


def validate_source(ckpt, mode, train, valid):
    cfg = ckpt["config"]
    required = dict(dataset="MELD", seed=SEED, split_seed=SPLIT,
                    v61_clean_split=True, v61_spf_mode=mode,
                    v52_pgm_mode="original", speaker_mode="off",
                    precontext_mode="off", epochs=100, classes=7)
    for name, value in required.items():
        if cfg.get(name) != value:
            raise ValueError(f"checkpoint config {name} != {value}")
    if cfg.get("v8_spf_mode", "off") != "off":
        raise ValueError("V8 checkpoint is not an original CSS baseline")
    if ckpt.get("selection") != "validation_weighted_f1":
        raise ValueError("checkpoint must be selected by VALID only")
    if (list(ckpt["train_dialogue_indices"]) != train
            or list(ckpt["valid_dialogue_indices"]) != valid):
        raise ValueError("checkpoint does not match V6.1 split")


@torch.no_grad()
def cache_features(model, ds, indices, device, batch_size):
    """Store frozen base logp, previous predicted state, raw B-turn features.

    Labels are placed in a separate tensor AFTER constructing all history
    features. No gold label or gold transition tag is a model input.
    """
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size,
                        shuffle=False, collate_fn=ds.collate_fn, num_workers=0)
    fields = {k: [] for k in ("base", "previous", "other", "repeat",
                               "eligible", "labels", "dialogue")}
    tags, ids = [], []
    model.eval()
    for text, visual, audio, speaker, mask, label, vids in loader:
        lengths = mask.sum(-1).long().tolist()
        out = model(text.to(device), visual.to(device), audio.to(device),
                    mask.to(device), speaker.permute(1,0,2).to(device), lengths)
        base = out[3].detach().float().cpu()
        prob = out[4].detach().float().cpu()
        speakers = speaker.argmax(-1).transpose(0,1).cpu()
        for j, n in enumerate(lengths):
            if n < 1:
                raise ValueError("empty dialogue")
            vid = str(vids[j])
            if vid in ids:
                raise ValueError("duplicate dialogue ID")
            raw = normalize_modalities(text[:n,j], audio[:n,j], visual[:n,j])
            history = history_features(prob[j,:n], speakers[j,:n], raw)
            y = label[j,:n].cpu().long()
            fields["base"].append(base[j,:n])
            for k in ("previous", "other", "repeat", "eligible"):
                fields[k].append(history[k])
            fields["labels"].append(y)
            fields["dialogue"].append(torch.full((n,), int(indices[len(ids)]),
                                                   dtype=torch.long))
            tags.extend(group_labels(y.tolist(), speakers[j,:n].tolist()))
            ids.append(vid)
    result = {k: torch.cat(v, dim=0) for k,v in fields.items()}
    result["tags"], result["ids"] = tags, ids
    if not all(torch.isfinite(result[k]).all().item()
               for k in ("base","previous","other")):
        raise ValueError("nonfinite cached features")
    expected = torch.tensor([any(t.endswith("_after_other") for t in tag)
                             for tag in tags])
    if not torch.equal(result["eligible"], expected):
        raise RuntimeError("speaker interval eligibility differs from group definition")
    print(f"Cached {len(ids)} dialogues, {len(tags)} turns; "
          f"repeat={int(result['repeat'].sum())}, "
          f"eligible={int(result['eligible'].sum())}", flush=True)
    return result


def score(data, logits):
    return compute_report(data["labels"].tolist(), logits.argmax(-1).cpu().tolist(),
                          data["tags"], 7)


def fit_self(train, device):
    model = SelfTransition(7).to(device)
    mask = train["repeat"]
    p = train["previous"][mask].to(device)
    y = train["labels"][mask].to(device)
    if len(y) == 0:
        raise ValueError("no repeated-speaker training turns")
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(SELF_EPOCHS):
        opt.zero_grad()
        loss = F.nll_loss(model(p).clamp_min(1e-8).log(), y)
        loss.backward()
        opt.step()
    model.eval().requires_grad_(False)
    print(f"Fitted marginal transition on {len(y)} TRAIN turns; "
          f"final NLL={loss.item():.4f}", flush=True)
    return model


def fit_interaction(train, contexts, prior, self_model, mode, device):
    # Same seed, row order, initialization and budget across conditional arms.
    torch.manual_seed(HEAD_SEED)
    model = InteractionTransition(self_model.scores, contexts.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    indices = torch.nonzero(train["eligible"], as_tuple=True)[0]
    generator = torch.Generator().manual_seed(HEAD_SEED + 1)
    for epoch in range(HEAD_EPOCHS):
        model.train()
        order = indices[torch.randperm(len(indices), generator=generator)]
        total = 0.0
        for idx in order.split(BATCH):
            base = train["base"][idx].to(device)
            previous = train["previous"][idx].to(device)
            other = contexts[idx].to(device)
            y = train["labels"][idx].to(device)
            mask = torch.ones(len(idx), dtype=torch.bool, device=device)
            logits, q = refine_logits(base, previous, mask, prior, self_model,
                                      model, other, mode)
            loss = F.cross_entropy(logits, y) + AUX * F.nll_loss(
                q.clamp_min(1e-8).log(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite adapter loss in {mode}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        if epoch in (0, 9, 19, 29, HEAD_EPOCHS-1):
            print(f"{mode} epoch={epoch+1} train_loss={total/len(indices):.4f}",
                  flush=True)
    return model.eval()


@torch.no_grad()
def predict_adapter(data, contexts, prior, self_model, model, mode, device):
    parts = []
    for idx in torch.arange(len(data["labels"])).split(BATCH):
        out, _ = refine_logits(data["base"][idx].to(device),
                               data["previous"][idx].to(device),
                               data["eligible"][idx].to(device),
                               prior, self_model, model,
                               contexts[idx].to(device), mode)
        parts.append(out.cpu())
    result = torch.cat(parts)
    if not torch.equal(result[~data["eligible"]], data["base"][~data["eligible"]]):
        raise RuntimeError("ineligible turns must preserve original CSS exactly")
    return result


def run(args):
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise FileExistsError("refusing to overwrite V9 outputs; use a new output path")
    ds = MELDDataset("data/meld_multimodal_features.pkl", train=True)
    tr, va, old, v52, seen = v61_disjoint_spf_ablation_split_indices(
        len(ds), 0.1, SPLIT)
    if (len(tr), len(va), len(old), len(v52)) != (846,87,115,104):
        raise RuntimeError("unexpected V6.1 development data split")
    if set(tr) & set(va) or set(va) & (set(old)|set(v52)|set(seen)):
        raise RuntimeError("invalid held-out dialogue IDs")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    # Both baseline checkpoints were generated by the user, with known provenance.
    ckpt = read_checkpoint(args.full_checkpoint)
    validate_source(ckpt, "full", tr, va)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    train = cache_features(model, ds, tr, device, ckpt["config"]["batch_size"])
    valid = cache_features(model, ds, va, device, ckpt["config"]["batch_size"])
    full_epoch = int(ckpt["epoch"])
    del model, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ckpt = read_checkpoint(args.residual_checkpoint)
    validate_source(ckpt, "residual_only", tr, va)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    residual = cache_features(model, ds, va, device, ckpt["config"]["batch_size"])
    residual_epoch = int(ckpt["epoch"])
    del model, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if valid["ids"] != residual["ids"] or not torch.equal(
        valid["labels"], residual["labels"]):
        raise RuntimeError("baseline validation alignment failure")
    scores = {"CSS_full": score(valid, valid["base"]),
              "CSS_residual": score(residual, residual["base"])}
    if (len(valid["labels"]) != 875
        or abs(scores["CSS_full"]["all"]["weighted_f1"]-76.14)>0.025
        or abs(scores["CSS_residual"]["all"]["weighted_f1"]-76.43)>0.025):
        raise RuntimeError("baseline re-inference does not reproduce V6.1 results")
    print("Baseline reproduction passed: CSS 76.14 / residual 76.43", flush=True)

    self_model = fit_self(train, device)
    # Eligible-turn marginal class prior from TRAIN only (Laplace smoothing).
    counts = torch.bincount(train["labels"][train["eligible"]], minlength=7).float()+1
    prior = (counts/counts.sum()).to(device)
    state = {"self": self_model.state_dict(), "train_prior": prior.cpu(), "heads": {}}
    self_out = predict_adapter(valid, valid["other"], prior, self_model,
                               None, "self", device)
    scores["self"] = score(valid, self_out)
    predictions = {"CSS_full": valid["base"].argmax(-1).tolist(),
                   "CSS_residual": residual["base"].argmax(-1).tolist(),
                   "self": self_out.argmax(-1).tolist()}
    for mode in MODES[1:]:
        # Shuffled control sources only from TRAIN, never other VALID dialogues.
        ctx_tr, ctx_va = train["other"], valid["other"]
        if mode == "shuffled":
            ctx_tr = shuffled_train_donors(train, train, HEAD_SEED+2)
            ctx_va = shuffled_train_donors(valid, train, HEAD_SEED+3)
        head = fit_interaction(train, ctx_tr, prior, self_model, mode, device)
        out = predict_adapter(valid, ctx_va, prior, self_model, head, mode, device)
        scores[mode] = score(valid, out)
        predictions[mode] = out.argmax(-1).tolist()
        state["heads"][mode] = {k:v.detach().cpu() for k,v in head.state_dict().items()}
    state["self"] = {k:v.detach().cpu() for k,v in state["self"].items()}

    print("\nV9 ISER SIX-ARM PILOT | FROZEN CSS | EXISTING VALID | NO TEST")
    print("Fields: W-F1 %, delta vs original CSS in pp")
    for group in GROUPS:
        baseline = scores["CSS_full"][group]["weighted_f1"]
        print(f"\n{group} n={scores['CSS_full'][group]['n']}")
        for name, result in scores.items():
            v = result[group]["weighted_f1"]
            print(f"  {name:<16} {v:6.2f}%  {v-baseline:+.2f}pp")
    print("\nPaired correctness changes vs CSS full (all utterances):")
    y = valid["labels"]
    base_ok = torch.tensor(predictions["CSS_full"]) == y
    changes = {}
    for name, pred in predictions.items():
        p = torch.tensor(pred)
        ok = p == y
        changes[name] = dict(fixed=int((ok & ~base_ok).sum()),
                             broken=int((~ok & base_ok).sum()),
                             label_flips=int((p != torch.tensor(predictions["CSS_full"])).sum()))
        print(f"  {name:<16} {changes[name]}")
    report = {
        "protocol": {
            "train_seed": SEED, "split_seed": SPLIT, "head_seed": HEAD_SEED,
            "train_dialogues": len(tr), "validation_dialogues": len(va),
            "existing_inspected_validation": True, "official_test_scored": False,
            "backbone_updates": 0, "lambda": LAMBDA, "aux_weight": AUX,
            "self_epochs": SELF_EPOCHS, "head_epochs": HEAD_EPOCHS,
            "head_selection": "fixed_final_epoch_no_validation_selection",
            "base_training_predictions": "in_sample_not_cross_fitted",
            "full_checkpoint": str(args.full_checkpoint),
            "residual_checkpoint": str(args.residual_checkpoint),
            "full_selected_epoch": full_epoch, "residual_selected_epoch": residual_epoch,
        },
        "scores": scores, "paired_changes": changes, "predictions": predictions,
        "labels": y.tolist(), "dialogue_indices_per_turn": valid["dialogue"].tolist(),
        "eligible_turns": valid["eligible"].tolist()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print("Saved:", args.output, args.output.with_suffix(".pt"))
    print("Exploratory only; no independent generalization or novelty claim.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-checkpoint", type=Path, default=ROOT/"v61_full_valid_best.pt")
    p.add_argument("--residual-checkpoint", type=Path, default=ROOT/"v61_residual_only_valid_best.pt")
    p.add_argument("--output", type=Path, default=Path("results/v9_iser/six_arm_validation.json"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

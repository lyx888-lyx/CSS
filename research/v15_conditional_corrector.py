"""V15 TRAIN-only, capacity-matched condition-dependent B evidence probe.

A-only classifier is trained first and then FROZEN. Three equally sized
correction networks are fitted from scratch on identical A-B-A target turns:
  self: A + zero-filled B channels
  real: A + actual intervening B
  shuffled: A + B from another dialogue (same session/speaker role)

The correction explicitly uses A*B learned interactions and has a
confidence-scaled, bounded residual over A-only logits. It does not
use previous gold emotion or change labels as inputs; no official TEST.
All training epochs, seeds, and hyperparameters are fixed in advance.
"""
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

from research.v14_b_incremental_probe import base_and_b_features, shuffle_b_donors

SEEDS = (61078, 61079, 61080)
DONOR_SEEDS = (31, 47, 73)
ARMS = ("self", "real_b", "shuffled_b")
BASE_EPOCHS = 24
CORRECTION_EPOCHS = 12
BATCH_SIZE = 128
HIDDEN = 96
BRANCH_DIM = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
ANCHOR = 0.02
MAX_CORRECTION = 1.75


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class AOnlyClassifier(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN, classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, classes)
        )

    def forward(self, a):
        return self.net(a)


class ConditionalCorrection(nn.Module):
    """Interaction-based bounded residual; input B=0 is self control."""

    def __init__(self, a_dim, b_dim, dim=BRANCH_DIM, cap=MAX_CORRECTION):
        super().__init__()
        if cap <= 0:
            raise ValueError("positive max correction required")
        self.cap = float(cap)
        self.a = nn.Sequential(nn.Linear(a_dim, dim), nn.GELU())
        self.b = nn.Sequential(nn.Linear(b_dim, dim), nn.GELU())
        self.gate = nn.Linear(dim * 3, 1)
        self.adjust = nn.Linear(dim * 3, 6)
        # All controls begin with exactly the frozen base prediction.
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)

    def forward(self, a, b, base_logits):
        if (a.ndim != 2 or b.ndim != 2 or
                base_logits.shape != (len(a), 6) or len(b) != len(a)):
            raise ValueError("unaligned A/B/base shapes")
        ha = self.a(a)
        hb = self.b(b)
        pair = torch.cat((ha, hb, ha * hb), dim=-1)
        # Confidence is model-derived, not a gold correctness flag.
        uncertainty = (1.0 - base_logits.detach().softmax(-1).amax(-1, keepdim=True))
        relevance = torch.sigmoid(self.gate(pair))
        delta = self.cap * uncertainty * relevance * torch.tanh(self.adjust(pair))
        return base_logits.detach() + delta, delta


def _to_tensor(data, device, dtype=torch.float32):
    return torch.as_tensor(data, dtype=dtype, device=device)


def prepare_fold(ex, train_ids, held_ids, donor_seed):
    """Fit all transforms on TRAIN only; use same columns across arms."""
    train_ids = np.asarray(train_ids, dtype=np.int64)
    held_ids = np.asarray(held_ids, dtype=np.int64)
    if not len(train_ids) or not len(held_ids):
        raise ValueError("empty split")
    if set(train_ids.tolist()) & set(held_ids.tolist()):
        raise ValueError("train and held-out overlap")
    if set(ex["session"][train_ids]) & set(ex["session"][held_ids]):
        raise ValueError("held-out session seen during fitting")
    if set(ex["dialogue"][train_ids]) & set(ex["dialogue"][held_ids]):
        raise ValueError("dialogue leakage")

    a_train, b_train = base_and_b_features(ex, train_ids)
    a_held, b_held = base_and_b_features(ex, held_ids)
    donor_train = shuffle_b_donors(ex, train_ids, donor_seed)
    donor_held = shuffle_b_donors(ex, held_ids, donor_seed + 100000)
    _, sb_train = base_and_b_features(ex, train_ids, donor_train)
    _, sb_held = base_and_b_features(ex, held_ids, donor_held)

    for ids, donors in ((train_ids, donor_train), (held_ids, donor_held)):
        if not (np.all(ex["session"][ids] == ex["session"][donors])
                and np.all(ex["b_role"][ids] == ex["b_role"][donors])
                and np.all(ex["dialogue"][ids] != ex["dialogue"][donors])):
            raise RuntimeError("invalid B donor pool")
        if not set(donors.tolist()).issubset(set(ids.tolist())):
            raise RuntimeError("B donor crossed a fold partition")

    a_scaler = StandardScaler().fit(a_train)
    b_scaler = StandardScaler().fit(b_train)
    # IMPORTANT: the same real-B-fitted scaler applies to all B arms.
    out = dict(
        a_train=a_scaler.transform(a_train).astype(np.float32),
        a_held=a_scaler.transform(a_held).astype(np.float32),
        b_train=b_scaler.transform(b_train).astype(np.float32),
        b_held=b_scaler.transform(b_held).astype(np.float32),
        sb_train=b_scaler.transform(sb_train).astype(np.float32),
        sb_held=b_scaler.transform(sb_held).astype(np.float32),
        y_train=ex["label"][train_ids].astype(np.int64),
        y_held=ex["label"][held_ids].astype(np.int64),
        changed_held=ex["changed"][held_ids].astype(bool),
        held_dialogue=ex["dialogue"][held_ids].tolist(),
    )
    return out


def class_weights_from_train(y, device):
    counts = np.bincount(y, minlength=6)
    if len(counts) != 6 or (counts == 0).any():
        raise ValueError("all six emotion labels required in fold TRAIN")
    weight = len(y) / (6 * counts.astype(np.float32))
    return _to_tensor(weight, device)


def _batches(n, batch_size, seed, epoch):
    gen = torch.Generator(device="cpu").manual_seed(seed + epoch * 971)
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i:i+batch_size]


def train_a_only(fold, seed, device, epochs=BASE_EPOCHS):
    seed_all(seed)
    a = _to_tensor(fold["a_train"], device)
    y = _to_tensor(fold["y_train"], device, torch.long)
    net = AOnlyClassifier(a.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = class_weights_from_train(fold["y_train"], device)
    net.train()
    for epoch in range(epochs):
        for ids in _batches(len(y), BATCH_SIZE, seed + 1000, epoch):
            batch = ids.to(device)
            logits = net(a[batch])
            loss = F.cross_entropy(logits, y[batch], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite A-only training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
        p.grad = None  # discard gradients from the completed A-only stage
    return net


@torch.no_grad()
def infer(net, correction, a, b, device):
    at = _to_tensor(a, device)
    base = net(at)
    if correction is None:
        result, delta = base, torch.zeros_like(base)
    else:
        bt = _to_tensor(b, device)
        correction.eval()
        result, delta = correction(at, bt, base)
    return (result.argmax(-1).cpu().numpy().astype(np.int64),
            float(delta.abs().amax().item()),
            float((result.argmax(-1) != base.argmax(-1)).float().mean().item()))


def train_correction(net, fold, arm, seed, device,
                     epochs=CORRECTION_EPOCHS):
    if arm not in ARMS:
        raise ValueError("unsupported correction arm")
    # RESET before EACH arm: identical parameter initializations
    # and minibatch ordering across Self / Real / Shuffled controls.
    seed_all(seed + 400000)
    a = _to_tensor(fold["a_train"], device)
    source = (np.zeros_like(fold["b_train"]) if arm == "self" else
              fold["sb_train"] if arm == "shuffled_b" else fold["b_train"])
    b = _to_tensor(source, device)
    y = _to_tensor(fold["y_train"], device, torch.long)
    weights = class_weights_from_train(fold["y_train"], device)
    correction = ConditionalCorrection(a.shape[1], b.shape[1]).to(device)
    optimizer = torch.optim.Adam(correction.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    correction.train()
    net.eval()
    for epoch in range(epochs):
        for ids in _batches(len(y), BATCH_SIZE, seed + 5000, epoch):
            batch = ids.to(device)
            with torch.no_grad():
                base = net(a[batch])
            logits, delta = correction(a[batch], b[batch], base)
            loss = (F.cross_entropy(logits, y[batch], weight=weights)
                    + ANCHOR * delta.square().mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite correction training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    correction.eval()
    return correction


def run_one_seed(fold, seed, device, smoke=False):
    """Returns aligned predictions plus correction sensitivity statistics."""
    ep_a = 2 if smoke else BASE_EPOCHS
    ep_c = 2 if smoke else CORRECTION_EPOCHS
    base = train_a_only(fold, seed, device, epochs=ep_a)
    y_base, _, _ = infer(base, None, fold["a_held"], None, device)
    results = {"a_only": y_base}
    diagnostics = {}
    for arm in ARMS:
        branch = train_correction(base, fold, arm, seed, device, epochs=ep_c)
        held_b = (np.zeros_like(fold["b_held"]) if arm == "self" else
                  fold["sb_held"] if arm == "shuffled_b" else fold["b_held"])
        pred, max_abs, changed_rate = infer(
            base, branch, fold["a_held"], held_b, device
        )
        results[arm] = pred
        diagnostics[arm] = {
            "max_abs_logit_adjustment": max_abs,
            "fraction_predictions_changed_vs_frozen_a": changed_rate,
        }
        if arm == "real_b":
            repl, repl_abs, repl_changed = infer(
                base, branch, fold["a_held"], fold["sb_held"], device
            )
            results["real_b_replaced"] = repl
            diagnostics["real_b_replaced"] = {
                "max_abs_logit_adjustment": repl_abs,
                "fraction_predictions_changed_vs_frozen_a": repl_changed,
            }
    return results, diagnostics

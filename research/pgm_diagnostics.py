"""Pure, no-model PGM gradient geometry utilities.

Zero vectors are not interpreted as cosine=0: a task with no
dependency on a parameter group has undefined angle (None).
All summaries are descriptive; gradient disagreement != harmful
interference without controlled optimization experiments.
"""
import math

import torch


PAIRS = (("L1", "L2"), ("L1", "L3"), ("L2", "L3"))
TASK_NAMES = ("L1", "L2", "L3")


def gradient_geometry(vectors, eps=1e-12):
    """N aligned 1-D gradient tensors -> detached norms and cosines.

    Cosine is None if either vector has zero / nearly zero norm.
    Callers must align the SAME parameters in identical order.
    """
    if len(vectors) != 3:
        raise ValueError("expected exactly three PGM task gradients")
    if not all(isinstance(v, torch.Tensor) and v.ndim == 1 for v in vectors):
        raise ValueError("all gradients must be 1-D tensors")
    if any(v.numel() != vectors[0].numel() for v in vectors):
        raise ValueError("task gradients are not parameter-aligned")
    if not all(torch.isfinite(v).all().item() for v in vectors):
        raise ValueError("non-finite task gradient")
    detached = [v.detach().float() for v in vectors]
    norms = [float(torch.linalg.vector_norm(g).item()) for g in detached]
    cos = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        key = f"L{i+1}_L{j+1}"
        if norms[i] <= eps or norms[j] <= eps:
            cos[key] = None
        else:
            val = float(torch.dot(detached[i], detached[j]).item() / (
                norms[i] * norms[j]
            ))
            cos[key] = max(-1.0, min(1.0, val))
    return {"norms": dict(zip(TASK_NAMES, norms)), "cosines": cos}


def weighted_norm_shares(task_norms, weights, eps=1e-12):
    """Share of alpha_i ||g_i||, BEFORE directional cancellation."""
    values = [float(task_norms[name]) for name in TASK_NAMES]
    if len(weights) != 3:
        raise ValueError("weights must contain 3 values")
    w = [float(item) for item in weights]
    if any(not math.isfinite(x) or x < 0 for x in values + w):
        raise ValueError("invalid nonnegative norms/weights")
    amount = [a * b for a, b in zip(w, values)]
    total = sum(amount)
    return {
        name: (float(value / total) if total > eps else None)
        for name, value in zip(TASK_NAMES, amount)
    }


def safe_mean(values):
    """Mean over finite numbers; null when no observation exists."""
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def summarise_batch_geometry(items):
    """Aggregate gradient geometry across independent mini-batches."""
    if not items:
        raise ValueError("at least one batch required")
    result = {
        "batches": len(items),
        "mean_norms": {
            name: safe_mean([d["norms"][name] for d in items])
            for name in TASK_NAMES
        },
        "pairwise": {},
    }
    for x, y in PAIRS:
        key = f"{x}_{y}"
        observed = [
            d["cosines"][key] for d in items
            if d["cosines"][key] is not None
        ]
        result["pairwise"][key] = {
            "mean_cosine": safe_mean(observed),
            "conflicting_batches": sum(c < 0 for c in observed),
            "measured_batches": len(observed),
            "negative_fraction": (
                sum(c < 0 for c in observed) / len(observed)
                if observed else None
            ),
        }
    return result

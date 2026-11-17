"""V6.3: inspect exact class-logit effect of CSS SPF polynomial.

For the trained original CSS SPF:
  fused = polynomial + residual
  logits = Linear(fused)
         = W*polynomial + W*residual + classifier_bias

Because the last classifier is exactly linear, this additivity is
exact, not an approximate attribution method. However, per-class
prediction/probability effects cannot establish causal training benefit.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def decompose_linear_logits(linear_head, polynomial, residual):
    if polynomial.shape != residual.shape or polynomial.ndim != 3:
        raise ValueError("need aligned SPF tensors [batch,turn,hidden]")
    p = F.linear(polynomial, linear_head.weight, bias=None)
    r = F.linear(residual, linear_head.weight, bias=None)
    full = p + r
    if linear_head.bias is not None:
        full = full + linear_head.bias
        residual_only = r + linear_head.bias
    else:
        residual_only = r
    return {
        "poly_logits": p,
        "residual_logits_without_bias": r,
        "full_logits": full,
        "residual_only_logits": residual_only,
    }


@torch.no_grad()
def true_class_margin(logits, labels):
    """True-logit minus strongest rival; positive means correct top-1."""
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected 2D logits and 1D labels")
    if logits.shape[0] != labels.numel() or logits.shape[-1] < 2:
        raise ValueError("shape mismatch")
    labels = labels.long()
    if (not bool((labels >= 0).all().item())
            or not bool((labels < logits.shape[-1]).all().item())):
        raise ValueError("invalid labels")
    true_logit = logits.gather(-1, labels[:, None]).squeeze(-1)
    competitor = logits.clone()
    competitor.scatter_(1, labels[:, None], float("-inf"))
    return true_logit - competitor.max(-1).values


@torch.no_grad()
def per_turn_logit_statistics(full, residual_only, p_effect,
                              r_effect, labels, eps=1e-8):
    """List of train-turn statistics. No model, dataset, or threshold tuning.

    Logit effects are CLASS-CENTERED so constant offsets across all
    classes are not misinterpreted as discriminative information.
    """
    if not (full.shape == residual_only.shape == p_effect.shape
            == r_effect.shape):
        raise ValueError("logit component shape mismatch")
    if full.ndim != 2 or len(labels) != len(full):
        raise ValueError("expected [N,C] logits and [N] labels")
    if len(full) == 0:
        return []
    margins_full = true_class_margin(full, labels)
    margins_res = true_class_margin(residual_only, labels)
    pred_full = full.argmax(-1)
    pred_resid = residual_only.argmax(-1)
    p_centered = p_effect - p_effect.mean(dim=-1, keepdim=True)
    r_centered = r_effect - r_effect.mean(dim=-1, keepdim=True)
    p_norm = torch.linalg.vector_norm(p_centered.float(), dim=-1)
    r_norm = torch.linalg.vector_norm(r_centered.float(), dim=-1)
    tv = 0.5 * (
        torch.softmax(full, dim=-1) -
        torch.softmax(residual_only, dim=-1)
    ).abs().sum(-1)
    p_prob_full = torch.softmax(full, dim=-1).gather(
        1, labels.long()[:, None]
    ).squeeze(-1)
    p_prob_resid = torch.softmax(residual_only, dim=-1).gather(
        1, labels.long()[:, None]
    ).squeeze(-1)
    out = []
    for j in range(len(full)):
        out.append({
            "label": int(labels[j].item()),
            "full_pred": int(pred_full[j].item()),
            "residual_pred": int(pred_resid[j].item()),
            "full_margin": float(margins_full[j].item()),
            "residual_margin": float(margins_res[j].item()),
            "poly_true_margin_delta": float(
                (margins_full[j]-margins_res[j]).item()
            ),
            "centered_poly_logit_norm": float(p_norm[j].item()),
            "centered_residual_logit_norm": float(r_norm[j].item()),
            "centered_poly_residual_ratio": float(
                (p_norm[j] / r_norm[j].clamp_min(eps)).item()
            ),
            "probability_total_variation": float(tv[j].item()),
            "true_class_prob_delta": float(
                (p_prob_full[j]-p_prob_resid[j]).item()
            ),
        })
    return out


def summarize_turns(rows):
    """Descriptive means on rows, no bootstrap on already inspected data."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    mean = lambda key: sum(float(d[key]) for d in rows) / n
    f_ok = [d["full_pred"] == d["label"] for d in rows]
    r_ok = [d["residual_pred"] == d["label"] for d in rows]
    delta = [float(d["poly_true_margin_delta"]) for d in rows]
    return {
        "n": n,
        "full_accuracy_pct": 100*sum(f_ok)/n,
        "residual_only_accuracy_pct": 100*sum(r_ok)/n,
        "flip_count": sum(
            d["full_pred"] != d["residual_pred"] for d in rows
        ),
        "poly_fixed": sum(f and not r for f, r in zip(f_ok, r_ok)),
        "poly_broken": sum(r and not f for f, r in zip(f_ok, r_ok)),
        "mean_centered_poly_logit_norm": mean(
            "centered_poly_logit_norm"
        ),
        "mean_centered_residual_logit_norm": mean(
            "centered_residual_logit_norm"
        ),
        "mean_centered_poly_residual_ratio": mean(
            "centered_poly_residual_ratio"
        ),
        "mean_abs_poly_true_margin_delta": (
            sum(abs(v) for v in delta)/n
        ),
        "mean_signed_poly_true_margin_delta": sum(delta)/n,
        "fraction_true_margin_improved_by_poly": (
            sum(x > 1e-8 for x in delta)/n
        ),
        "fraction_true_margin_harmed_by_poly": (
            sum(x < -1e-8 for x in delta)/n
        ),
        "mean_probability_total_variation": mean(
            "probability_total_variation"
        ),
        "mean_abs_true_class_probability_change": (
            sum(abs(float(d["true_class_prob_delta"])) for d in rows)/n
        ),
    }

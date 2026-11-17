"""Read-only reconstruction of CSS SPF output as high-order + residual.

SPFusion.forward() in model.py is:
    poly = sqrt_signed(product of per-order (g_t z_t + g_a z_a + g_v z_v + z_const))
    poly = poly @ output_proj + output_bias
    res = residual_proj(cat(text,audio,vision))
    output = poly + res

This duplicates the exact forward arithmetic of the checked source.
No model changes. Per-utterance component sizes are diagnostics, NOT
causal attribution or a demonstration of useful high-order interactions.
"""
import torch


@torch.no_grad()
def decompose_spfusion(spf, t, a, v):
    if not (t.ndim == a.ndim == v.ndim == 3):
        raise ValueError("expected [B, dialogue_len, hidden_dim] inputs")
    if t.shape != a.shape or t.shape != v.shape:
        raise ValueError("modality embeddings must be aligned")
    B, L, _ = t.shape
    polynomial_product = torch.ones(
        B, L, spf.rank, device=t.device, dtype=t.dtype
    )
    # Original source constructs a [B,L,1] all-one tensor.
    ones = torch.ones(B, L, 1, device=t.device, dtype=t.dtype)
    order_terms = []
    gates_by_order = []
    for i in range(spf.poly_order):
        z_t = spf.act(t @ spf.weight_list_text[i])
        z_a = spf.act(a @ spf.weight_list_audio[i])
        z_v = spf.act(v @ spf.weight_list_vision[i])
        z_c = ones @ spf.weight_list_const[i]
        gates = spf.gate_act(spf.modality_gates[i])
        z_p = gates[0] * z_t + gates[1] * z_a + gates[2] * z_v + z_c
        polynomial_product = polynomial_product * z_p
        order_terms.append(z_p)
        gates_by_order.append(gates)
    stable = torch.sign(polynomial_product) * torch.sqrt(
        torch.abs(polynomial_product) + spf.eps
    )
    polynomial = stable @ spf.output_proj + spf.output_bias
    residual = spf.residual_proj(torch.cat([t, a, v], dim=-1))
    return {
        "polynomial": polynomial,
        "residual": residual,
        "reconstructed": polynomial + residual,
        "order_terms": order_terms,
        "gates_by_order": gates_by_order,
    }


@torch.no_grad()
def component_statistics(poly, residual, valid_mask):
    if poly.shape != residual.shape or poly.ndim != 3:
        raise ValueError("component tensors must have matching [B,L,D] shape")
    if valid_mask.shape != poly.shape[:2]:
        raise ValueError("mask shape does not match dialogue batch")
    mask = valid_mask.bool()
    if not bool(mask.any().item()):
        raise ValueError("no valid utterances")
    p = poly[mask].float()
    r = residual[mask].float()
    np = torch.linalg.vector_norm(p, dim=-1)
    nr = torch.linalg.vector_norm(r, dim=-1)
    cos = ((p * r).sum(-1) / (np * nr).clamp_min(1e-8))
    return {
        "n": int(mask.sum().item()),
        "mean_poly_norm": float(np.mean().item()),
        "mean_residual_norm": float(nr.mean().item()),
        "mean_poly_to_residual_norm_ratio": float(
            (np / nr.clamp_min(1e-8)).mean().item()
        ),
        "mean_poly_residual_cosine": float(cos.mean().item()),
        "fraction_poly_norm_greater_residual": float(
            (np > nr).float().mean().item()
        ),
    }


def summarize_predictions(true_labels, pred_full, pred_wo_poly, pred_wo_resid):
    """Paired diagnostic of inference-only removal, NOT trained ablation."""
    if not (len(true_labels) == len(pred_full)
            == len(pred_wo_poly) == len(pred_wo_resid)):
        raise ValueError("prediction and label arrays differ in length")
    if not true_labels:
        return None
    n = len(true_labels)
    full_good = [p == t for p, t in zip(pred_full, true_labels)]
    a_good = [p == t for p, t in zip(pred_wo_poly, true_labels)]
    b_good = [p == t for p, t in zip(pred_wo_resid, true_labels)]
    return {
        "n": n,
        "full_accuracy_pct": round(100 * sum(full_good) / n, 2),
        "without_polynomial_accuracy_pct": round(100 * sum(a_good) / n, 2),
        "without_residual_accuracy_pct": round(100 * sum(b_good) / n, 2),
        "without_polynomial_fixed": sum(
            not f and a for f, a in zip(full_good, a_good)
        ),
        "without_polynomial_broken": sum(
            f and not a for f, a in zip(full_good, a_good)
        ),
        "without_residual_fixed": sum(
            not f and b for f, b in zip(full_good, b_good)
        ),
        "without_residual_broken": sum(
            f and not b for f, b in zip(full_good, b_good)
        ),
    }

"""V5.2 opt-in PGM update-direction consistency experimental CONTROL.

Original CSS solves normalized-gradient MGDA for alpha but takes a raw-loss
weighted backpropagation step. This *experimental* alternative keeps the
same original alpha from PGM._project and multiplies each task loss by
1 / ||g_i||, with ONE global factor to match the magnitude of the
original raw-weighted gradient on PGM's selected shared parameters.

No architectural change, no additional labels, no inference change.
All coefficients are detached. It is an experiment, NOT proven superior.
"""
import math

import torch

from model import PGM


def derive_direction_aligned_coefficients(grads, alpha, eps=1e-8):
    """Return scaled detached loss weights with matched shared-grad norm.

    Original raw update: d_raw = sum_i alpha_i * g_i
    PGM target:          d_unit = sum_i alpha_i * g_i/(||g_i||+eps)
    Our update:          d_aligned = ||d_raw||/||d_unit|| * d_unit

    Guarantees d_aligned and d_raw have equal Euclidean norm on the
    EXACT same PGM shared parameter subset, at this same parameter state.
    Their *directions* can differ. This does not match the gradient
    magnitude in SPF/classification heads outside PGM's chosen subset.

    alpha and gradient norms are constants for this backward pass.
    """
    if len(grads) != 3 or len(alpha) != 3:
        raise ValueError("CSS PGM requires 3 task gradients and 3 alpha values")
    if not all(g.ndim == 1 for g in grads):
        raise ValueError("task gradients must be flattened vectors")
    if any(g.numel() != grads[0].numel() for g in grads):
        raise ValueError("task gradients have inconsistent parameter layout")
    a = alpha.detach()
    if not bool(torch.isfinite(a).all().item()) or bool((a < -1e-6).any().item()):
        raise ValueError("PGM alpha must be finite and nonnegative")
    if abs(float(a.sum().item()) - 1.0) > 1e-3:
        raise ValueError("PGM simplex coefficients do not sum to one")
    g = [v.detach() for v in grads]
    norms = torch.stack([torch.linalg.vector_norm(v) for v in g])
    if not bool(torch.isfinite(norms).all().item()):
        raise ValueError("nonfinite task gradient norm")
    if bool((norms <= eps).any().item()):
        raise ValueError(
            "one PGM task gradient is effectively zero; "
            "cannot define normalized target direction"
        )
    # Avoid stacking the three very large task-gradient vectors,
    # which unnecessarily increases peak GPU memory during training.
    raw = sum((a[i] * g[i] for i in range(3)))
    unit = sum((a[i] * g[i] / (norms[i] + eps) for i in range(3)))
    raw_norm = torch.linalg.vector_norm(raw)
    unit_norm = torch.linalg.vector_norm(unit)
    if (not bool(torch.isfinite(raw_norm).item())
            or not bool(torch.isfinite(unit_norm).item())
            or float(raw_norm.item()) <= eps
            or float(unit_norm.item()) <= eps):
        raise ValueError("degenerate PGM combined gradient direction")
    scale = raw_norm / unit_norm
    coeff = (a * scale / (norms + eps)).detach()
    if not bool(torch.isfinite(coeff).all().item()):
        raise ValueError("nonfinite direction-aligned loss coefficients")
    cosine = torch.dot(raw, unit) / (raw_norm * unit_norm)
    angle = math.degrees(math.acos(
        max(-1.0, min(1.0, float(cosine.item())))
    ))
    meta = {
        "direction_angle_deg": angle,
        "raw_shared_gradient_norm": float(raw_norm.item()),
        "unit_shared_gradient_norm": float(unit_norm.item()),
        "matching_scale": float(scale.item()),
        "original_alpha": [float(v) for v in a.cpu().tolist()],
        "effective_loss_coefficients": [float(v) for v in coeff.cpu().tolist()],
        "original_task_gradient_norms": [
            float(v) for v in norms.cpu().tolist()
        ],
    }
    return coeff, meta


class DirectionAlignedPGM(PGM):
    """Source-compatible PGM loss with normalized target direction.

    We reuse original _compute_task_losses, _compute_gradients and _project,
    so only the LOSS COMBINATION is experimental.
    """

    def __init__(self, nll_loss, kl_loss):
        super().__init__(nll_loss, kl_loss)
        self.last_diagnostics = None

    def forward(self, outputs, labels_, umask, shared_params):
        losses = self._compute_task_losses(outputs, labels_, umask)
        grads = self._compute_gradients(losses, shared_params)
        alpha = self._project(grads)
        coeff, self.last_diagnostics = derive_direction_aligned_coefficients(
            grads, alpha
        )
        new_loss = sum(w * loss for w, loss in zip(coeff, losses))
        return new_loss, alpha, losses

"""Training-only supervised contrastive regularization of CSS fused features.

This is a SMALL feasibility probe inspired by 2026 EMART (ACL) and
EII-SCL (Interspeech), not an implementation of either paper and NOT
a novelty claim. Uses only TRAIN labels; inference is unchanged.

The loss is class-balanced at the anchor level to prevent frequent
MELD Neutral utterances from dominating. All other real utterances
in the current minibatch are potential contrasts, INCLUDING those
from other dialogues, but no validation/test samples appear here.
"""
import torch
from torch.nn import functional as F


def class_balanced_supcon(fused, labels, valid_mask, temperature=0.2):
    """Return SupCon scalar, number of valid positive-pair anchors.

    fused: [B,L,D], labels [B,L] gold TRAIN labels,
    valid_mask: [B,L] real turns. Models make predictions WITHOUT labels.
    """
    if (fused.ndim != 3 or labels.ndim != 2 or
            labels.shape != fused.shape[:2] or
            valid_mask.shape != labels.shape):
        raise ValueError("fused=[B,L,D], label/mask=[B,L] expected")
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")
    valid = valid_mask.bool().reshape(-1)
    z = F.normalize(fused.reshape(-1, fused.shape[-1])[valid], dim=-1)
    y = labels.reshape(-1)[valid]
    n = len(y)
    if n < 2:
        return fused.sum() * 0.0, 0
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (y[:, None] == y[None, :]) & (~eye)
    eligible = same.any(-1)
    n_anchors = int(eligible.sum().item())
    if n_anchors == 0:
        return fused.sum() * 0.0, 0

    similarity = (z.float() @ z.float().T) / temperature
    similarity = similarity.masked_fill(eye, float("-inf"))
    log_denom = torch.logsumexp(similarity, dim=1)
    # Directly sum positive similarities; do NOT multiply a -inf
    # diagonal by zero (which would produce NaNs).
    pos_counts = same.sum(1).clamp_min(1)
    mean_pos = similarity.masked_fill(~same, 0.0).sum(1) / pos_counts
    per_anchor = log_denom - mean_pos

    # Each emotion class contributes equally among eligible anchors.
    yc = y[eligible]
    _, inverse, counts = torch.unique(
        yc, return_inverse=True, return_counts=True
    )
    weights = (1.0 / counts[inverse].float())
    weighted = per_anchor[eligible] * weights
    loss = weighted.sum() / weights.sum()
    return loss, n_anchors

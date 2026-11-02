"""Speaker-aware, history-gated feature fusion for CSS.

This module attends to preceding utterances split by speaker identity.
Its attention keys/values come ONLY from the provided features, speaker IDs,
and validity masks. No ground-truth emotion labels are used as input.

IMPORTANT: Original CSS uses bidirectional context encoders. Hence their
features can already contain future information; this module's past-only
masks do not make the full network causal.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.speaker_history import build_speaker_history_masks


class SpeakerHistoryFusion(nn.Module):
    """Residual history fusion with separate same/other speaker attention.

    Args:
        hidden_dim: Embedding width, 1024 in the original CSS.
        mode: 'both', 'self', or 'other' for research ablations.
        use_decay: Subtract softplus(decay) * temporal distance from logits.
        max_lag: Optional maximum preceding turn distance.
    """
    def __init__(self, hidden_dim, mode="both", use_decay=True, max_lag=None):
        super().__init__()
        if mode not in ("both", "self", "other"):
            raise ValueError("mode must be both, self, or other")
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.use_decay = use_decay
        self.max_lag = max_lag

        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_other = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learned per-utterance mixture; include the original representation
        # in the softmax to give the residual a reference option.
        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.raw_decay_self = nn.Parameter(torch.tensor(-2.0))
        self.raw_decay_other = nn.Parameter(torch.tensor(-2.0))
        # Begin close to baseline and allow training to learn a stronger effect.
        self.raw_strength = nn.Parameter(torch.tensor(-2.0))

    def _attend(self, scores, values, eligible, distances, decay):
        if self.use_decay:
            scores = scores - F.softplus(decay) * distances.to(scores.dtype)

        # All-empty histories are valid for a first utterance. Softmax
        # alone would treat a fully masked row as uniform; explicitly zero it.
        masked_scores = scores.masked_fill(~eligible, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights.masked_fill(~eligible, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(weights, values), eligible.any(dim=-1)

    def forward(self, features, speaker_ids, valid_mask):
        """Return [B,L,D] enhanced features. Input tensors are not mutated."""
        if features.ndim != 3:
            raise ValueError("features must have shape [B,L,D]")
        if features.shape[:2] != speaker_ids.shape or features.shape[:2] != valid_mask.shape:
            raise ValueError("speaker_ids and valid_mask must match [B,L]")
        if features.shape[-1] != self.hidden_dim:
            raise ValueError("unexpected feature width")

        same, other = build_speaker_history_masks(
            speaker_ids, valid_mask, max_lag=self.max_lag
        )
        if self.mode == "self":
            other = torch.zeros_like(other)
        elif self.mode == "other":
            same = torch.zeros_like(same)

        batch, length, _ = features.shape
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        pos = torch.arange(length, device=features.device)
        distances = (pos[:, None] - pos[None, :]).clamp_min(0).unsqueeze(0)
        self_ctx, self_has = self._attend(
            scores, v, same, distances, self.raw_decay_self
        )
        other_ctx, other_has = self._attend(
            scores, v, other, distances, self.raw_decay_other
        )
        mix = torch.softmax(self.gate(torch.cat(
            [features, self_ctx, other_ctx], dim=-1
        )), dim=-1)
        # The output of an empty branch must remain exactly zero.
        self_term = (
            mix[..., 1:2] * self_has.unsqueeze(-1)
            * self.proj_self(self_ctx)
        )
        other_term = (
            mix[..., 2:3] * other_has.unsqueeze(-1)
            * self.proj_other(other_ctx)
        )
        delta = torch.sigmoid(self.raw_strength) * (self_term + other_term)
        return features + delta * valid_mask.unsqueeze(-1).to(features.dtype)

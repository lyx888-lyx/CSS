"""V1 feasibility probe: speaker-relative changes BEFORE CSS dialogue encoders.

This module processes one modality's utterance features immediately after
unimodal projection, before any CSS dialogue-wide attention. It distinguishes
the speaker's own prior utterances from preceding other-speaker utterances
and compares them to the current utterance. No ground-truth emotion labels,
speaker emotions, or future *positions* are used by this module.

CAUTION: the original CSS dialogue encoders downstream remain bidirectional.
This is NOT an end-to-end causal ERC implementation or a validated novelty claim.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.speaker_history import build_speaker_history_masks


class PreContextSpeakerShift(nn.Module):
    """One-modality, past-only speaker-difference residual block.

    A small shift-evidence network compares current features with attentive
    historical summaries. Separate gates control self-history and other-history
    contributions. Missing histories contribute *exactly zero*.

    Args:
        dim: projected CSS feature width
        mode: self, other, or both (ablation)
        max_lag: restrict history to at most this many prior turn positions
    """

    def __init__(self, dim, mode="both", max_lag=None):
        super().__init__()
        if mode not in ("self", "other", "both"):
            raise ValueError("mode must be self, other or both")
        if dim < 2:
            raise ValueError("dim must be >=2")
        self.mode = mode
        self.max_lag = max_lag
        d_attn = max(1, dim // 4)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, d_attn, bias=False)
        self.key = nn.Linear(dim, d_attn, bias=False)
        self.self_diff = nn.Linear(dim, dim, bias=False)
        self.other_diff = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )
        # Deliberately nonzero but modest residual at initialization.
        self.raw_strength = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _summarize(scores, values, mask):
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(weights, values), mask.any(dim=-1)

    def forward(self, x, speaker_ids, valid_mask):
        if x.ndim != 3 or x.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("expected modality features [B, L, D]")
        if x.shape[:2] != speaker_ids.shape or x.shape[:2] != valid_mask.shape:
            raise ValueError("speaker ids and validity must be [B,L]")

        same, other = build_speaker_history_masks(
            speaker_ids, valid_mask, max_lag=self.max_lag
        )
        if self.mode == "self":
            other = torch.zeros_like(other)
        elif self.mode == "other":
            same = torch.zeros_like(same)

        normalized = self.norm(x)
        q = self.query(normalized)
        k = self.key(normalized)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        ctx_self, available_self = self._summarize(scores, normalized, same)
        ctx_other, available_other = self._summarize(scores, normalized, other)

        # A speaker's return can represent either inertia or a shift;
        # the difference is *feature-level evidence*, not a true emotion shift.
        own_difference = normalized - ctx_self
        other_difference = ctx_other - normalized
        gates = torch.sigmoid(self.gate(torch.cat(
            [normalized, own_difference.abs(), other_difference.abs()],
            dim=-1
        )))
        contribution = (
            gates[..., :1] * available_self.unsqueeze(-1)
            * self.self_diff(own_difference)
            +
            gates[..., 1:] * available_other.unsqueeze(-1)
            * self.other_diff(other_difference)
        )
        return x + (
            torch.sigmoid(self.raw_strength)
            * contribution
            * valid_mask.unsqueeze(-1).to(x.dtype)
        )

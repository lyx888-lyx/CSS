"""V9 ISER: frozen-CSS predictive transition refinement.

All model inputs are CSS predictions or intervening raw utterance features.
Gold labels are accepted ONLY by external training/evaluation losses.
The base CSS is bidirectional; explicit history lookup is past-only,
but the complete predictor is not an online causal model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("self", "conditional", "incremental", "shuffled")
EPS = 1e-8
LAMBDA = 0.5


def normalize_modalities(*modalities):
    """Per-utterance RMS-like normalization, with equal modal norm scales."""
    if any(x.ndim != 2 for x in modalities):
        raise ValueError("expected unpadded [turn, dimension] modalities")
    if len({len(x) for x in modalities}) != 1:
        raise ValueError("modalities have different dialogue length")
    return torch.cat([
        F.normalize(x.float(), dim=-1, eps=EPS) * math.sqrt(x.shape[-1])
        for x in modalities
    ], dim=-1)


def history_features(probabilities, speakers, modal_features):
    """For each turn, previous SAME speaker posterior and mean OTHER features.

    Only the OPEN interval (previous same speaker, current turn) is used.
    First speaking turns and consecutive same speaker turns receive no
    interactive correction. The returned eligibility mask is label-free.
    Inputs are one unpadded dialogue; no cross-dialogue state.
    """
    if probabilities.ndim != 2 or modal_features.ndim != 2:
        raise ValueError("probabilities and modal_features must be 2D")
    n, c = probabilities.shape
    if speakers.shape != (n,) or len(modal_features) != n:
        raise ValueError("unaligned dialogue")
    prev = probabilities.new_full((n, c), 1.0 / c)
    other = modal_features.new_zeros(modal_features.shape)
    repeat = torch.zeros(n, dtype=torch.bool, device=probabilities.device)
    eligible = repeat.clone()
    latest = {}
    for i, speaker in enumerate(speakers.tolist()):
        if speaker in latest:
            j = latest[speaker]
            prev[i] = probabilities[j]
            repeat[i] = True
            # j is the latest self turn, so intervening turns are other.
            if i > j + 1:
                other[i] = modal_features[j + 1:i].mean(dim=0)
                eligible[i] = True
        latest[speaker] = i
    return dict(previous=prev, other=other, repeat=repeat, eligible=eligible)


def predictive_distribution(previous, transition_scores):
    """Marginalize uncertain previous predicted state, not gold labels."""
    if previous.ndim != 2 or transition_scores.shape[-2:] != (
            previous.shape[-1], previous.shape[-1]):
        raise ValueError("invalid transition dimensions")
    matrix = transition_scores.softmax(dim=-1)
    return torch.matmul(previous.unsqueeze(-2), matrix).squeeze(-2)


def log_ratio(numerator, denominator):
    return numerator.clamp_min(EPS).log() - denominator.clamp_min(EPS).log()


class SelfTransition(nn.Module):
    """Marginal transition for repeat turns; not a causal self-only estimate."""
    def __init__(self, n_classes):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(n_classes, n_classes))

    def forward(self, previous):
        return predictive_distribution(previous, self.scores)


class InteractionTransition(nn.Module):
    """Input-conditioned adjustment to a FIXED learned marginal transition."""
    def __init__(self, self_scores, context_dim, hidden=32):
        super().__init__()
        self.register_buffer("self_scores", self_scores.detach().clone())
        self.n_classes = self_scores.shape[0]
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_classes ** 2)
        )
        # All conditional arms start at the exact marginal transition.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, previous, other):
        residual = self.net(other).reshape(-1, self.n_classes, self.n_classes)
        return predictive_distribution(previous, self.self_scores + residual)


def refine_logits(base_logp, previous, eligible, prior, self_model,
                  interaction=None, other=None, mode="incremental"):
    """Same label-free inference contract for each experimental mode.

    All corrections are applied to EXACTLY the same A...B...A eligible
    turns. Uncorrected rows are bit-identical to CSS base log probabilities.
    self:        log q_marginal - log pi_train
    conditional: log q_conditional - log pi_train
    incremental: log q_conditional - log q_marginal
    shuffled:    same incremental formula with unrelated TRAIN features
    """
    if mode not in MODES:
        raise ValueError("unknown V9 mode")
    if eligible.shape != (len(base_logp),):
        raise ValueError("invalid eligibility mask")
    q_self = self_model(previous)
    if mode == "self":
        q = q_self
        delta = log_ratio(q, prior)
    else:
        if interaction is None or other is None:
            raise ValueError("conditional transition requires context")
        q = interaction(previous, other)
        reference = prior if mode == "conditional" else q_self
        delta = log_ratio(q, reference)
    # Fixed lambda, no tuning or clipping based on development validation.
    adjusted = torch.where(eligible[:, None], base_logp + LAMBDA * delta,
                           base_logp)
    return adjusted, q


def shuffled_train_donors(recipient, train, seed):
    """Negative control using a fixed TRAIN bank only, no VALID donors.

    Each eligible recipient receives another TRAIN dialogue's feature mean.
    Same recipient dialogue is excluded. No labels participate.
    """
    candidates = torch.nonzero(train["eligible"], as_tuple=True)[0]
    if not len(candidates):
        raise ValueError("empty training donor bank")
    gen = torch.Generator().manual_seed(seed)
    result = torch.zeros_like(recipient["other"])
    for i in torch.nonzero(recipient["eligible"], as_tuple=True)[0].tolist():
        available = candidates[
            train["dialogue"][candidates] != recipient["dialogue"][i]
        ]
        if not len(available):
            raise ValueError("no donor in a different training dialogue")
        j = available[torch.randint(len(available), (1,), generator=gen).item()]
        result[i] = train["other"][j]
    return result

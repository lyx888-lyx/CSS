"""V2 feasibility model: new-information-prioritized first-order speaker state.

IMPORTANT:
- This is a lightweight, independent, CURRENT-UTTERANCE feature encoder,
  NOT a modification to CSS and NOT classical per-series GM(1,1).
- All dialogue history operations go strictly from t=0 to t=L-1.
- Only raw/precomputed current utterance feature vectors, speaker identity,
  and validity masks enter inference. Emotion labels are NEVER model inputs.
- 'other' is an exogenous context feature, NOT a causal emotion effect.
- The damping/update are grey-theory INSPIRED first-order dynamics, not
  a claim of novel differential equations.
"""
import torch
from torch import nn
from torch.nn import functional as F


MODES = ("current", "last", "ema", "adaptive", "adaptive_other")


class V2SpeakerStateProbe(nn.Module):
    """Five controlled arms sharing identical local encoder and classifier.

    Inputs use CSS dataloader conventions:
      text [L,B,1024], visual [L,B,342], audio [L,B,300],
      qmask [L,B,S] one-hot, umask [B,L].
    Outputs [B,L,C] logits.

    - current: zero historical feature channels
    - last: prior = last same-speaker current-utterance representation
    - ema: prior = fixed alpha=0.5 same-speaker representation EMA
    - adaptive: positive exponential decay towards shared state mean,
        then innovation-dependent convex posterior update
    - adaptive_other: adaptive with an additional separate aggregate
        of other speakers' utterances since the speaker last spoke.
    """

    def __init__(
        self, mode="current", hidden_dim=64, text_dim=1024,
        visual_dim=342, audio_dim=300, classes=7, speakers=9,
        dropout=0.15,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be one of " + str(MODES))
        if hidden_dim < 4 or speakers < 2:
            raise ValueError("hidden_dim>=4 and speakers>=2 required")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_speakers = speakers

        # NO dialogue-level self-attention or future feature aggregation.
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Grey-inspired scalar damping with a positive value, and
        # a trainable shared attractor for cold starts.
        self.raw_decay = nn.Parameter(torch.tensor(-2.0))
        self.population_state = nn.Parameter(torch.zeros(hidden_dim))
        self.innovation_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.other_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.other_projection.weight, mean=0.0, std=0.01)
        self.raw_other_gain = nn.Parameter(torch.tensor(-2.0))

        # Identical readout dimensions across ALL variants, so modes
        # differ in available historical evidence, not head topology.
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, text, visual, audio, qmask, umask, return_diagnostics=False):
        if umask.ndim != 2 or qmask.ndim != 3:
            raise ValueError("expected umask=[B,L] and qmask=[L,B,S]")
        batch, length = umask.shape
        if qmask.shape != (length, batch, self.n_speakers):
            raise ValueError("qmask shape inconsistent with umask or number of speakers")
        if (text.shape[:2] != (length, batch)
                or visual.shape[:2] != (length, batch)
                or audio.shape[:2] != (length, batch)):
            raise ValueError("modality features require [L,B,D]")
        valid = umask.bool()
        speaker_ids = qmask.argmax(-1).transpose(0, 1).long()  # B,L
        local = self.fuse(torch.cat([
            self.text_encoder(text.transpose(0, 1)),
            self.visual_encoder(visual.transpose(0, 1)),
            self.audio_encoder(audio.transpose(0, 1)),
        ], dim=-1))
        B, L, D = local.shape
        states = local.new_zeros(B, self.n_speakers, D)
        prior_seen = torch.zeros(B, self.n_speakers, dtype=torch.bool, device=local.device)
        last_t = torch.full(
            (B, self.n_speakers), -1, dtype=torch.long, device=local.device
        )
        # For EACH target speaker, store only other-speaker utterances
        # that arrived since this target last spoke. No future turns.
        other_context = local.new_zeros(B, self.n_speakers, D)
        other_seen = torch.zeros(B, self.n_speakers, dtype=torch.bool, device=local.device)
        outputs = []
        diag = {"has_own_history": [], "has_other_since_last": []}

        for t in range(L):
            h = local[:, t]
            is_real = valid[:, t]
            who = speaker_ids[:, t]
            speaker_select = F.one_hot(who, self.n_speakers).bool() & is_real[:, None]
            gather_idx = who[:, None, None].expand(-1, 1, D)
            state_prev = states.gather(1, gather_idx).squeeze(1)
            context_prev = other_context.gather(1, gather_idx).squeeze(1)
            has_own = prior_seen.gather(1, who[:, None]).squeeze(1) & is_real
            has_other = other_seen.gather(1, who[:, None]).squeeze(1) & is_real
            old_t = last_t.gather(1, who[:, None]).squeeze(1)
            gap = (t - old_t).clamp(min=1).to(h.dtype)

            if self.mode == "current":
                prior = torch.zeros_like(h)
                can_use_history = torch.zeros_like(has_own)
                other_flag = torch.zeros_like(has_other)
            elif self.mode in ("last", "ema"):
                prior = torch.where(has_own[:, None], state_prev, torch.zeros_like(h))
                can_use_history = has_own
                other_flag = torch.zeros_like(has_other)
            else:
                decay = torch.exp(-F.softplus(self.raw_decay) * gap[:, None])
                mu = self.population_state[None, :]
                own_prior = mu + decay * (state_prev - mu)
                own_prior = torch.where(has_own[:, None], own_prior, mu)
                if self.mode == "adaptive_other":
                    contribution = (
                        torch.sigmoid(self.raw_other_gain)
                        * torch.tanh(self.other_projection(context_prev))
                    )
                    prior = own_prior + torch.where(
                        has_other[:, None], contribution, torch.zeros_like(h)
                    )
                    can_use_history = has_own | has_other
                    other_flag = has_other
                else:
                    prior = own_prior
                    can_use_history = has_own
                    other_flag = torch.zeros_like(has_other)

            usable_prior = torch.where(
                can_use_history[:, None], prior, torch.zeros_like(prior)
            )
            innovation = torch.where(
                can_use_history[:, None], h - prior, torch.zeros_like(h)
            )
            # The same classifier sees the same number of inputs in all modes.
            meta = torch.stack([
                has_own.to(h.dtype) if self.mode != "current" else torch.zeros_like(gap),
                other_flag.to(h.dtype),
                torch.where(has_own, torch.log1p(gap), torch.zeros_like(gap))
                if self.mode in ("adaptive", "adaptive_other")
                else torch.zeros_like(gap),
            ], dim=-1)
            logits = self.classifier(
                torch.cat([h, usable_prior, innovation, meta], dim=-1)
            )
            outputs.append(logits)

            # Update AFTER producing the current turn's prediction.
            # Thus no current/future input can influence the prior for t.
            if self.mode == "current":
                state_new = h
            elif self.mode == "last":
                state_new = h
            elif self.mode == "ema":
                state_new = torch.where(
                    has_own[:, None], 0.5 * state_prev + 0.5 * h, h
                )
            else:
                # New-information priority: innovation-dependent update.
                update_input = torch.cat([
                    h, prior, (h - prior).abs(),
                    torch.log1p(gap)[:, None],
                    has_own.to(h.dtype)[:, None],
                ], dim=-1)
                update_rate = torch.sigmoid(self.innovation_gate(update_input))
                state_new = prior + update_rate * (h - prior)
            states = torch.where(
                speaker_select[:, :, None], state_new[:, None, :], states
            )
            prior_seen = prior_seen | speaker_select
            last_t = torch.where(
                speaker_select,
                torch.full_like(last_t, t),
                last_t,
            )

            # Every other speaker's utterance updates the addressee's
            # external-context accumulator. Reset the current speaker's
            # accumulator so it contains only INTERVENING other speech.
            other_active = is_real[:, None] & (~speaker_select)
            updated_context = 0.5 * other_context + 0.5 * h[:, None, :]
            other_context = torch.where(
                speaker_select[:, :, None],
                torch.zeros_like(other_context),
                torch.where(other_active[:, :, None], updated_context, other_context),
            )
            other_seen = torch.where(
                speaker_select,
                torch.zeros_like(other_seen),
                other_seen | other_active,
            )

            if return_diagnostics:
                diag["has_own_history"].append(has_own)
                diag["has_other_since_last"].append(has_other)
        result = torch.stack(outputs, dim=1)  # B,L,C
        if return_diagnostics:
            return result, {
                key: torch.stack(value, dim=1)
                for key, value in diag.items()
            }
        return result

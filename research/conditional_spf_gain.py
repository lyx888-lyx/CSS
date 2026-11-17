"""V8 pilot: modest high-order SPF gain and context-conditional gating.

The module uses ONLY features and speaker IDs observable at inference:
- within-turn text/audio/visual cosine disagreements;
- the three cosine distances to the latest PRECEDING different speaker;
- a binary "previous other speaker exists" indicator.

No true emotion labels, no dialogue-level group tags and no future turn
indices are used to construct the gate. IMPORTANT: the upstream original
CSS encoder is bidirectional, so overall CSS predictions remain
non-online, even though the gate's explicit cross-speaker lookup is causal.

Modes are controlled for fair comparisons:
  gain4: fixed polynomial multiplier 4; no gate parameters
  current: same multiplier with a trainable 7->1 scalar gate,
           last four (other-speaker) features hard-masked to zero
  other: same trainable architecture, uses other-speaker history
All conditional gates initialize at multiplier 4.0, with zero
initial linear coefficients and bias; base CSS parameters initialize
identically for all experimental arms.

This is a testable architecture hypothesis, NOT established novelty.
Dynamic modality gating and speaker-aware context are prior art.
"""
import torch
from torch import nn
from torch.nn import functional as F


def previous_different_speaker_indices(speaker_ids, valid_mask):
    """Return index of most recent PREVIOUS turn from ANOTHER speaker.

    Missing previous other speaker is represented by index -1.
    Padding is excluded. Works for 2-party and multiparty dialogues.
    """
    if speaker_ids.ndim != 2 or valid_mask.shape != speaker_ids.shape:
        raise ValueError("speaker_ids and valid_mask must be [B,L]")
    B, L = speaker_ids.shape
    valid = valid_mask.bool()
    # Tensor-only O(B*L^2) masking, no .item() CPU/GPU synchronization
    # inside batch loops. MELD's dialogues are short and padded.
    pos = torch.arange(L, device=speaker_ids.device)
    earlier = pos[None, :] < pos[:, None]  # [target, source]
    differing = speaker_ids[:, :, None] != speaker_ids[:, None, :]
    permitted = (
        earlier[None, :, :]
        & differing
        & valid[:, :, None]
        & valid[:, None, :]
    )
    positions = pos[None, None, :].expand(B, L, L)
    return positions.masked_fill(~permitted, -1).max(dim=-1).values


def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1, eps=1e-8)


def seven_context_features(text, audio, visual, speakers, valid):
    """[B,L,7] parameter-free inputs for the conditional gain."""
    if not (text.shape == audio.shape == visual.shape):
        raise ValueError("three projected modalities must share [B,L,D]")
    if text.ndim != 3 or speakers.shape != text.shape[:2]:
        raise ValueError("invalid feature/speaker shapes")
    if valid.shape != speakers.shape:
        raise ValueError("invalid valid mask")
    B, L, D = text.shape
    indices = previous_different_speaker_indices(speakers, valid)
    has_other = indices >= 0
    safe_indices = indices.clamp(min=0)
    index3 = safe_indices[:, :, None].expand(B, L, D)
    seen = [
        v.gather(dim=1, index=index3)
        for v in (text, audio, visual)
    ]
    cross = [
        torch.where(
            has_other,
            cosine_distance(now, prev),
            torch.zeros_like(has_other, dtype=now.dtype),
        )
        for now, prev in zip((text, audio, visual), seen)
    ]
    feature = torch.stack([
        cosine_distance(text, audio),
        cosine_distance(text, visual),
        cosine_distance(audio, visual),
        *cross,
        has_other.to(dtype=text.dtype),
    ], dim=-1)
    # Only valid utterance rows participate in an upstream loss.
    return feature.masked_fill(~valid.bool().unsqueeze(-1), 0.0)


class CrossSpeakerSPFGain(nn.Module):
    """Scalar high-order gain; seven input features, same parameter count.

    Scale is 4*(1+0.5*tanh(z)), bounded (2,6).
    At init, z=0, gain=4 for all turns. V8 control gain4 is
    therefore exactly the same as the conditional modes at init.
    """

    def __init__(self, mode):
        super().__init__()
        if mode not in ("current", "other"):
            raise ValueError("mode must be current or other")
        self.mode = mode
        self.proj = nn.Linear(7, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, text, audio, visual, speakers, valid):
        feature = seven_context_features(
            text, audio, visual, speakers, valid
        )
        if self.mode == "current":
            # Exact same 7->1 parameters, but disallow all
            # previous-other-speaker context input.
            feature = torch.cat([
                feature[..., :3],
                torch.zeros_like(feature[..., 3:]),
            ], dim=-1)
        z = self.proj(feature)
        gain = 4.0 * (1.0 + 0.5 * torch.tanh(z))
        return torch.where(
            valid.bool().unsqueeze(-1), gain, torch.ones_like(gain)
        )

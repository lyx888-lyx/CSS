"""V11: speaker-relative acoustic reaction before CSS contextual encoding.

This trains a *representation-level*, not class-logit-level, adjustment.
The explicit acoustic input uses only the current target, previous
same-speaker turn, and intervening OTHER speaker turns. Full CSS remains
bidirectional, thus this is NOT proof of online causality or influence.

Both modes have IDENTICAL parameter counts/initialization and apply only
when an intervening other speaker is present:
  self: current minus last same-speaker audio evidence
  reaction: self difference + intervening other evidence + interaction
No true emotion labels or gold change/stable indicators are inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("self", "reaction")


def acoustic_reaction_features(audio, speaker_ids, valid_mask):
    """Return [B,L,3D] raw change, intervening-other context, interaction.

    Previous same turn is strictly earlier; OTHER context lies strictly
    between that previous turn and the current turn. No sample-pair
    crosses dialogues. Padded rows are ignored. Supports >=2 speakers.
    """
    if audio.ndim != 3:
        raise ValueError("audio must be [B,L,D]")
    B,L,D=audio.shape
    if speaker_ids.shape != (B,L) or valid_mask.shape != (B,L):
        raise ValueError("mismatched [B,L] metadata")
    if audio.device != speaker_ids.device or audio.device != valid_mask.device:
        raise ValueError("all input tensors must be on same device")

    valid=valid_mask.bool()
    target=torch.arange(L,device=audio.device)
    src=target
    earlier=(src[None,:] < target[:,None])
    same=(speaker_ids[:,:,None] == speaker_ids[:,None,:])
    same_prev=(
        earlier[None,:,:]
        & same
        & valid[:,:,None]
        & valid[:,None,:]
    )
    indices=src[None,None,:].expand(B,L,L).masked_fill(~same_prev,-1)
    prev_idx=indices.max(dim=-1).values
    safe_prev=prev_idx.clamp_min(0)
    prev_audio=audio.gather(1,safe_prev[:,:,None].expand(B,L,D))

    in_interval=(
        (src[None,None,:] > prev_idx[:,:,None])
        & earlier[None,:,:]
        & (~same)
        & valid[:,:,None]
        & valid[:,None,:]
        & (prev_idx[:,:,None] >= 0)
    )
    counts=in_interval.sum(-1)
    eligible=(prev_idx>=0)&(counts>0)&valid
    weights=in_interval.to(dtype=audio.dtype)
    mean_other=torch.bmm(weights,audio)/counts.clamp_min(1).unsqueeze(-1)
    self_change=F.layer_norm(audio-prev_audio,(D,))
    other_response=F.layer_norm(mean_other-prev_audio,(D,))
    interaction=self_change*other_response
    features=torch.cat([self_change,other_response,interaction],dim=-1)
    features=features.masked_fill(~eligible.unsqueeze(-1),0.0)
    return features,eligible,prev_idx,counts


class SpeakerRelativeAcousticReaction(nn.Module):
    """Zero-output initialized acoustic residual, capacity matched arms."""

    def __init__(self, dim, mode, bottleneck=128):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"invalid V11 mode {mode}")
        self.mode=mode
        self.in_dim=dim
        self.project=nn.Sequential(
            nn.Linear(3*dim,bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck,dim),
        )
        # All original and modified models have identical logits at initial
        # parameters; both arms begin as exact original CSS.
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, audio, speakers, valid):
        feat,eligible,_,_=acoustic_reaction_features(audio,speakers,valid)
        if self.mode=="self":
            feat=torch.cat([feat[...,:self.in_dim],
                            torch.zeros_like(feat[...,self.in_dim:])],dim=-1)
        update=self.project(feat)*eligible.unsqueeze(-1).to(audio.dtype)
        return audio + update

"""Causal, speaker-typed masks for a future CSS relation encoder.

Every mask has shape [B, L, L] with mask[b, i, j] == True when
utterance j is an eligible historical source for target utterance i.
These masks do NOT make the full CSS backbone causal: the original
attention encoders are bidirectional.
"""
import torch


def build_speaker_history_masks(speaker_ids, valid_mask, max_lag=None):
    """Return (same_speaker_past, other_speaker_past) boolean masks.

    Args:
        speaker_ids: LongTensor [batch, length], identities within dialogues.
        valid_mask: BoolTensor [batch, length], True for real utterances.
        max_lag: Optional maximum number of intervening turn positions
            measured as i-j. If None, all preceding turns are eligible.
    """
    if speaker_ids.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("speaker_ids and valid_mask must both have shape [B, L]")
    if speaker_ids.shape != valid_mask.shape:
        raise ValueError("speaker_ids and valid_mask must have the same shape")
    if speaker_ids.device != valid_mask.device:
        raise ValueError("speaker_ids and valid_mask must share a device")
    if max_lag is not None and (not isinstance(max_lag, int) or max_lag < 1):
        raise ValueError("max_lag must be a positive integer or None")

    _, length = speaker_ids.shape
    pos = torch.arange(length, device=speaker_ids.device)
    distance = pos[:, None] - pos[None, :]  # target position i minus source j
    past = (distance > 0).unsqueeze(0)
    if max_lag is not None:
        past = past & (distance <= max_lag).unsqueeze(0)

    valid = valid_mask.bool()
    eligible = past & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    same = speaker_ids.unsqueeze(-1) == speaker_ids.unsqueeze(-2)
    return eligible & same, eligible & ~same

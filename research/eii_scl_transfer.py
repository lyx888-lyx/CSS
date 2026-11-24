"""Paper-inspired EII-SCL transfer objectives for CSS (arXiv:2607.17366v1).

No changes to the inference graph. All pairs are within a single dialogue.
The paper leaves hard-window gradients unspecified. The dynamic arm uses
fixed identity Q/K projections and detached, integer window membership;
it is a documented reconstruction, NOT verified official EII-SCL code.
"""
import torch
from torch.nn import functional as F

ARMS = ("supcon", "eii_fixed", "eii_dynamic")
TAU = .07
ALPHA = .02
FIXED_WINDOW = 2


def pair_masks(labels, speakers, valid):
    """[B,L,L] masks: no padding, self-pairs or cross-dialogue comparisons."""
    if labels.ndim != 2 or speakers.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("labels, speakers and valid must be aligned [B,L]")
    valid = valid.bool()
    n = labels.shape[1]
    distinct = ~torch.eye(n, dtype=torch.bool, device=labels.device)[None]
    eligible = valid[:, :, None] & valid[:, None, :] & distinct
    same_speaker = speakers[:, :, None] == speakers[:, None, :]
    same_label = labels[:, :, None] == labels[:, None, :]
    return eligible, same_speaker, same_label


@torch.no_grad()
def dynamic_windows(features, same_speaker_pairs):
    """Eq. (6)-(7) with Wq=Wk=I; no learned projections or STE claimed.

    Paper uses unscaled projected dot product; the identity specialization
    therefore also uses unscaled dot product, not cosine attention.
    """
    if features.ndim != 3 or same_speaker_pairs.shape != (
            features.shape[0], features.shape[1], features.shape[1]):
        raise ValueError("dynamic window tensor shape mismatch")
    f = features.detach().float()
    # Stable softmax on same-speaker non-self keys. Handle no-peer rows.
    scores = f @ f.transpose(1, 2)
    has_peer = same_speaker_pairs.any(-1)
    masked = scores.masked_fill(~same_speaker_pairs, -torch.inf)
    masked = torch.where(has_peer[..., None], masked, torch.zeros_like(masked))
    attn = torch.softmax(masked, dim=-1) * same_speaker_pairs
    positions = torch.arange(f.shape[1], device=f.device)
    distance = (positions[:, None] - positions[None, :]).abs()
    expected = (attn * distance).sum(-1)
    # Avoid floor off-by-one from tiny floating point errors at integers.
    return torch.floor(expected + 1e-6).long()


def contrastive_loss(features, labels, qmask, umask, arm, tau=TAU,
                     fixed_window=FIXED_WINDOW, return_stats=False):
    """CSS fused features [B,L,D]; labels/mask [B,L]; qmask [B,L,S].

    supcon: conventional mean of positive log-probabilities.
    eii_*: literal summed-positive probability ratio (Eq. 10-11).
    For EII, same-emotion cross-speaker samples are easy negatives per Eq. 9.
    Skip anchors without positives; average all valid eligible anchors.
    EII hard weights remain differentiable (no detach), per literal formula.
    """
    if arm not in ARMS or tau <= 0 or fixed_window < 0:
        raise ValueError("invalid objective, temperature or fixed window")
    if (features.ndim != 3 or features.shape[:2] != labels.shape or
            umask.shape != labels.shape or qmask.ndim != 3 or
            qmask.shape[:2] != labels.shape):
        raise ValueError("unaligned features/labels/speaker/mask")
    valid = umask.bool()
    speakers = qmask.argmax(-1)
    eligible, same_speaker, same_label = pair_masks(labels, speakers, valid)
    z = F.normalize(features.float(), dim=-1, eps=1e-8)
    sim = (z @ z.transpose(1, 2)).clamp(-1, 1)
    hard = torch.zeros_like(eligible)
    windows = None
    if arm == "supcon":
        positive = eligible & same_label
        active = positive.any(-1)
        # Shift cosine exponent by 1 for bounded, stable exp.
        logits = (sim - 1.) / tau
        denom = (logits.exp() * eligible).sum(-1)
        per_anchor = denom.clamp_min(1e-30).log() - (
            (logits * positive).sum(-1) / positive.sum(-1).clamp_min(1))
    else:
        positive = eligible & same_speaker & same_label
        active = positive.any(-1)
        positions = torch.arange(labels.shape[1], device=labels.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        if arm == "eii_fixed":
            windows = torch.full_like(labels, fixed_window)
        else:
            windows = dynamic_windows(features, eligible & same_speaker)
        hard = eligible & same_speaker & ~same_label & (
            distance[None] <= windows[:, :, None])
        easy = eligible & ~positive & ~hard
        exp = ((sim - 1.) / tau).exp()
        pos_sum = (exp * positive).sum(-1)
        # Literal Eq. 10: gradient flows through similarity AND weight.
        neg_sum = (exp * easy).sum(-1) + (
            exp * hard * ((1. - sim) / 2.)).sum(-1)
        per_anchor = ((pos_sum + neg_sum).clamp_min(1e-30).log()
                      - pos_sum.clamp_min(1e-30).log())
    loss = per_anchor[active].mean() if active.any() else features.sum() * 0.
    if not return_stats:
        return loss
    with torch.no_grad():
        stats = {
            "valid_anchors": int(valid.sum().item()),
            "eligible_anchors": int(active.sum().item()),
            "hard_negative_pairs": int((hard & active[..., None]).sum().item()),
            "window_sum": int(windows[active].sum().item()) if windows is not None else 0,
            "window_count": int(active.sum().item()) if windows is not None else 0,
        }
    return loss, stats


class SupervisedContrastHook:
    """Capture pre-classifier fused feature via hook; consume exactly once.

    This avoids changing CSS forward outputs or PGM's original 3-task losses.
    No hook remains installed for validation/inference.
    """
    def __init__(self, model, arm):
        if arm not in ARMS:
            raise ValueError("invalid auxiliary arm")
        self.arm = arm
        self.feature = None
        self.stats = {}
        self.reset()
        self.handle = model.all_output_layer.register_forward_pre_hook(self.capture)

    def reset(self):
        self.stats = dict(valid_anchors=0, eligible_anchors=0,
                          hard_negative_pairs=0, window_sum=0, window_count=0,
                          batches=0, raw_aux_loss_sum=0.)

    def capture(self, module, inputs):
        if self.feature is not None:
            raise RuntimeError("previous fusion feature was not consumed")
        self.feature = inputs[0]

    def __call__(self, model, labels, qmask, umask):
        if self.feature is None:
            raise RuntimeError("CSS classifier hook did not capture a feature")
        features, self.feature = self.feature, None
        loss, stats = contrastive_loss(
            features, labels, qmask, umask, self.arm, return_stats=True)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite auxiliary contrastive loss")
        for k, v in stats.items():
            self.stats[k] += v
        self.stats["batches"] += 1
        self.stats["raw_aux_loss_sum"] += float(loss.detach().item())
        return ALPHA * loss

    def close(self):
        self.handle.remove()
        self.feature = None

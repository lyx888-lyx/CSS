"""Train-only robustness loss for cross-speaker history in V1.1.

Counterfactual here denotes a controlled FEATURE-PATH ABLATION, not
human emotional causation. Training targets may supervise a loss mask;
they are never passed into the model's forward features.
"""
from contextlib import contextmanager

import torch


def stable_after_other_training_mask(labels, speaker_ids, valid_mask):
    """Boolean [B,L]: speaker returns after others and retains prior emotion.

    labels and speaker IDs are [B,L]. This is strictly for TRAIN loss;
    no true labels can be used at validation/test prediction time.
    """
    if labels.ndim != 2 or speaker_ids.shape != labels.shape or valid_mask.shape != labels.shape:
        raise ValueError("labels, speaker IDs, mask must share [B,L]")
    bsz, seq_len = labels.shape
    result = torch.zeros_like(valid_mask, dtype=torch.bool)
    valid = valid_mask.bool()
    # Non-vectorized reference is transparent; MELD/IEMOCAP dialogues short.
    for b in range(bsz):
        previous = {}
        for t in range(seq_len):
            if not bool(valid[b, t]):
                continue
            speaker = int(speaker_ids[b, t])
            if speaker in previous:
                j = previous[speaker]
                if t - j > 1 and bool(labels[b, j] == labels[b, t]):
                    result[b, t] = True
            previous[speaker] = t
    return result


@contextmanager
def without_v1_other_residual(model):
    """Zero only the output of V1's other_diff projection in every modality."""
    blocks = getattr(model, "precontext_shift", None)
    if not isinstance(blocks, torch.nn.ModuleDict):
        raise ValueError("V1.1 requires the V1 precontext ModuleDict")
    handles = []
    def zero(_module, _inputs, output):
        return torch.zeros_like(output)
    try:
        for block in blocks.values():
            handles.append(block.other_diff.register_forward_hook(zero))
        yield
    finally:
        for handle in handles:
            handle.remove()


def masked_categorical_kl(teacher_prob, student_prob, mask):
    """KL(teacher || student), averaged over selected real utterances."""
    if teacher_prob.shape != student_prob.shape or teacher_prob.shape[:2] != mask.shape:
        raise ValueError("teacher, student and mask shapes disagree")
    eps = 1e-8
    t = teacher_prob.detach().clamp_min(eps)
    s = student_prob.clamp_min(eps)
    per_turn = (t * (t.log() - s.log())).sum(dim=-1)
    select = mask.to(per_turn.dtype)
    return (per_turn * select).sum() / select.sum().clamp_min(1)


def v11_training_consistency_loss(model, inputs, labels, speaker_ids, valid_mask):
    """Compare eval-mode full-V1 vs eval-mode other-history-path removal.

    Assumes inputs=(textf,visuf,acouf,umask,qmask,lengths).
    The supervised PGM pass is separately computed in TRAIN mode.
    Both predictions in this path-specific consistency loss use EVAL
    mode, preventing dropout noise from masquerading as history effect.
    The full-path student is differentiable; the masked-path teacher
    is detached. No true labels are included in model inputs.
    """
    eligible = stable_after_other_training_mask(labels, speaker_ids, valid_mask)
    if not bool(eligible.any()):
        # Do not run extra forwards for batches with no eligible returns.
        return next(model.parameters()).sum() * 0.0, 0

    was_training = model.training
    try:
        model.eval()
        student_prob = model(*inputs)[4]  # with gradients
        if student_prob.shape[:2] != labels.shape:
            raise ValueError("student probabilities and labels mismatch")
        with torch.no_grad():
            with without_v1_other_residual(model):
                teacher_prob = model(*inputs)[4]
        loss = masked_categorical_kl(teacher_prob, student_prob, eligible)
    finally:
        model.train(was_training)
    return loss, int(eligible.sum().item())

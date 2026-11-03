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


def v11_training_consistency_loss(model, inputs, student_prob, labels, speaker_ids, valid_mask):
    """Teacher: remove other residual; student: original full V1 forward.

    Assumes inputs=(textf,visuf,acouf,umask,qmask,lengths), each before
    model.forward. Teacher uses eval mode to disable stochastic dropout;
    student_prob was computed in training mode with normal dropout.
    """
    if student_prob.ndim != 3 or labels.shape != student_prob.shape[:2]:
        raise ValueError("student probabilities and labels mismatch")
    eligible = stable_after_other_training_mask(labels, speaker_ids, valid_mask)
    if not bool(eligible.any()):
        return student_prob.sum() * 0.0, 0

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            with without_v1_other_residual(model):
                teacher_prob = model(*inputs)[4]
    finally:
        model.train(was_training)
    return masked_categorical_kl(teacher_prob, student_prob, eligible), int(eligible.sum().item())

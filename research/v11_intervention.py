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
    # Move only the small discrete metadata to CPU once. Avoid synchronizing
    # GPU one scalar at a time for every dialogue position in 100 epochs.
    y = labels.detach().cpu().tolist()
    sp = speaker_ids.detach().cpu().tolist()
    valid = valid_mask.detach().bool().cpu().tolist()
    rows = []
    for b in range(bsz):
        previous = {}
        selected = [False] * seq_len
        for t in range(seq_len):
            if not valid[b][t]:
                continue
            speaker = int(sp[b][t])
            if speaker in previous:
                j = previous[speaker]
                if t - j > 1 and y[b][j] == y[b][t]:
                    selected[t] = True
            previous[speaker] = t
        rows.append(selected)
    return torch.tensor(rows, dtype=torch.bool, device=valid_mask.device)


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


def measure_auxiliary_gradient_ratio(supervised_loss, consistency_loss, coefficient, params):
    """One-batch diagnostic: ||lambda*g_cons|| / ||g_PGM|| on selected V1 weights.

    Measures only gradients of the three V1 other_diff weight matrices,
    *before* optimizer weight decay. It must not write .grad or update
    parameters and must be called before loss.backward(). This is a
    diagnostic, not a proof that an objective is useful or negligible.
    """
    params = tuple(params)
    if not params:
        raise ValueError("provide trainable V1 other-difference parameters")
    if coefficient < 0:
        raise ValueError("coefficient must be nonnegative")
    base_grad = torch.autograd.grad(
        supervised_loss, params, retain_graph=True, allow_unused=True
    )
    aux_grad = torch.autograd.grad(
        consistency_loss, params, retain_graph=True, allow_unused=True
    )

    def norm(grads):
        squares = [
            g.detach().float().square().sum()
            for g in grads if g is not None
        ]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt().item())

    base = norm(base_grad)
    aux = norm(aux_grad)
    scaled = float(coefficient) * aux
    return {
        "pgm_grad_norm": base,
        "raw_kl_grad_norm": aux,
        "weighted_kl_grad_norm": scaled,
        "weighted_to_pgm_grad_ratio": scaled / max(base, 1e-12),
    }

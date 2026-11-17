# V5: Source-aligned CSS PGM gradient audit — NO training

## Research question, not a speculative module

Original **Cross-Space Synergy (CSS), AAAI 2026** jointly addresses:
- High-order multi-modal interaction via **SPFusion**, and
- Conflicting multi-objective optimization via **Pareto Gradient Modulator (PGM)**.

Our previous V0/V1/V1.1/V2/V3/V4 experiments did not produce a
convincing MELD VALID performance gain. We are **not** trying a new
attention/gate/class prior now. We first ask:

> Are the three original CSS losses in conflict on the precise
> parameter subset PGM examines, and how does this geometry differ
> across the modules on which the three objectives act?

This diagnostic **cannot by itself prove a weakness of PGM** or
identify a better optimizer. It is a prerequisite for that claim.

## Source facts checked against THIS repository

In `model.py`, `PGM._compute_task_losses` calculates:

- `L1`: fused NLL
- `L2`: sum of three unimodal NLL terms
- `L3`: sum of three unimodal vs fused KL terms

The KL fused softmax target `kl_p_all` is **not detached**
(`F.softmax(all_final_out / temp)`), so L3 gradients propagate to
both the fusion teacher and unimodal students.

In `train_MELD.py::train_or_eval_model`, `shared_params`
used for the PGM simplex weight calculation include, **in source
order**:

1. `shared_transformer`
2. `textf_input`, `acouf_input`, `visuf_input`
3. `text_to_audio`, `text_to_vision`,
   `bias_gate_audio`, `bias_gate_vision`
4. `t_t`, `a_a`, `v_v`
5. `t_t_gate`, `t_gate`, `a_a_gate`, `a_gate`,
   `v_v_gate`, `v_gate`
6. `features_reduce_t`, `features_reduce_a`,
   `features_reduce_v`

It does **not** include `last_gate` (SPFusion), the four
classification heads, or the speaker-embedding parameter.
The *computed weights* are then applied to the three losses,
whose gradients update **all** task-connected parameters during
normal training. Exclusion from PGM's weight-computation vector
does NOT imply exclusion from backpropagation.

Original `PGM._project` constructs a cosine Gram matrix of
normalized task gradients and runs SciPy SLSQP on a simplex.
We invoke the original method (no parallel reimplementation).

## Diagnostic design

`scripts/audit_css_pgm_gradients.py`:

- Loads the **original** previously VALID-selected, architecture-off,
  MELD CSS checkpoint solely as a parameter snapshot.
- Checks exact original MELD split seed 10068:
  1037 TRAIN dialogues / 115 VALID dialogues.
- Samples a fixed subset of TRAIN dialogues using sample seed
  **73011**, default **8 batches x 8 dialogues**.
  Never touches the 115 VALID utterance cohort or official TEST.
- Runs **no optimizer, no backward weight update, no training**.
  By default sets model to `train()` so normal dropout is active,
  but freezes parameters in time; dropout seed **61027**.
  This is a **training-like gradient snapshot, NOT an actual
  historical CSS training-trajectory record**.
- Exactly computes L1/L2/L3 via original `PGM._compute_task_losses`.
- Uses `torch.autograd.grad` on exactly the original PGM parameter
  list (same source order) and on SPF (extra analysis only).
- Logs task gradient norms, three pairwise cosine similarities,
  number/fraction of batches with negative cosines, and PGM
  simplex weights from original `_project`.
- Logs the *pre-cancellation* relative magnitude
  `alpha_i * ||g_i|| / sum_j alpha_j * ||g_j||`.
  This is NOT a causal influence estimate; gradient vectors can
  cancel directionally.
- Breaks geometry down into:
  `cross_modal_transformer`,
  `modality_input_and_bias`,
  `unimodal_context`,
  `gates_and_reductions`,
  and **SPF excluded from PGM weights**.
- Correctly marks undefined angles (e.g., **L2 has no gradients
  on SPF parameters**) as `null`, not as zero.
- On exactly the same sampled TRAIN utterances, logs per-class
  mean L1/L2/L3 losses, train-subset classification accuracy,
  and NLL's local output-logit gradient norm
  `||p - one_hot(y)||`. **These are NOT per-class parameter
  gradient conflicts.** Small class support is explicitly shown.

Source-reference utility in `research/pgm_diagnostics.py`,
CPU synthetic tests in
`tests/test_v5_css_pgm_gradient_audit.py`.

## Run — no retraining, fast smoke first

```bash
cd /code/CSS
conda activate cmdpad
git -c http.version=HTTP/1.1 fetch origin
git switch analysis/v5-css-pgm-gradient-audit

# Test only new V5 code; keep legacy tests untouched.
python -m unittest discover -s tests \
  -p 'test_v5_css_pgm_gradient_audit.py' -v

# Start with two TRAIN dialogues, one batch:
python -u -m scripts.audit_css_pgm_gradients \
  --checkpoint results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt \
  --batch-size 2 --max-batches 1 \
  --output results/v5_pgm_gradient_smoke.json
```

If smoke succeeds, run the **one fixed diagnostic protocol**:

```bash
python -u -m scripts.audit_css_pgm_gradients \
  --checkpoint results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt \
  --batch-size 8 --max-batches 8 \
  --sample-seed 73011 --dropout-seed 61027 --mode train \
  --output results/v5_css_pgm_gradient_audit.json \
  2>&1 | tee results/v5_css_pgm_gradient_audit.log
```

By design, 64 sampled TRAIN dialogues are a preliminary view.
No claim about full-dataset frequency or class-specific optimizer
harm should be made if Fear/Disgust support is sparse.

If GPU memory is insufficient, first report the failing command
and memory. Using `--batch-size 4 --max-batches 8` reduces
the sample cohort and changes the interpretation; always
record the actual protocol.

## How to interpret the output

- **Frequent L1 vs L2 negative cosine in selected PGM space:**
  some task conflict exists on the sampled training snapshot.
  Does not imply PGM fails; the original point of PGM is to
  handle it.
- **Frequent L1 vs L3 conflict on SPF but no L2-SPF gradient:**
  structurally expected partly because L2 does not touch SPF;
  supports investigating how PGM weights transfer from selected
  shared params to excluded task-specific params.
- **Large raw-gradient norm differences** despite similar
  PGM alpha: projection is solved on unit gradients, while
  alpha multiplies unnormalized losses. This *may* matter
  but is only a descriptive potential mismatch.
- **Small or unstable conflict fractions:** do not build a
  proposed gradient surgery module; the hypothesized
  optimization bottleneck lacks evidence.
- **High Fear/Disgust per-class L1 despite scarce observations:**
  compatible with difficult classification, NOT proof of
  class-specific gradient conflicts or evidence that
  a class-aware optimizer will help.
- Any apparent gradient mismatch must be followed by a
  pre-registered CONTROLLED baseline comparison, multiple
  independent seeds/splits, and second dataset. Do not use
  the repeatedly inspected 115 VALID dialogues to tune
  another optimizer or report generalizable gains.

## Integrity and known constraints

The CSS implementation processes full dialogues with
bidirectional attention. This is *not* a strictly online
prediction model. No labels are fed into model inference.

The checkpoint's epoch was chosen with VALID historically;
we only **use original TRAIN inputs/labels for this audit**.
No official TEST loading, no validation logits/labels, no
optimizer step, no modifications to the original CSS code,
and no attempt to manufacture an improvement in F1.

**GO only if** the geometry reveals a concrete,
repeatable mismatch that the original PGM does not already
address, and a controlled test has a way to isolate it.
Otherwise NO-GO and reconsider representation/context issues.

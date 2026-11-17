# V5.1 — PGM's normalized projection vs actual CSS backprop direction

## Fixed evidence from V5 TRAIN-only 8-batch snapshot

User-run V5 on 64 sampled MELD TRAIN dialogues (666 utterances) from
an original CSS valid-selected checkpoint, without any model updates,
VALID observations, or official TEST.

Original PGM shared-parameter gradient cosines:

| Pair | Mean cosine | Negative batches |
|---|---:|---:|
| fused NLL L1 vs sum-unimodal NLL L2 | +0.2979 | 0/8 |
| fused NLL L1 vs KL L3 | -0.0322 | 4/8 |
| sum-unimodal NLL L2 vs KL L3 | +0.0944 | 0/8 |

PGM weights (mean): [0.3398, 0.2659, 0.3942].
**Pre-cancellation** weighted raw gradient norm shares:
[0.3367, 0.6411, 0.0222].

On sampled TRAIN turns: Fear L1=2.083, acc=24% (25 turns);
Disgust L1=1.566, acc=46.15% (13 turns).
Those figures are not enough to infer generalization or
class-specific PGM gradient conflicts. Do not use them to
cherry-pick label weighting.

## The precise, testable question

In source `model.py`:

```python
G = torch.stack([g / (g.norm() + 1e-8) for g in grads])
Q = G @ G.t()
alpha = scipy.optimize.minimize(... Q ...).x
total_loss = sum(w * l for w, l in zip(alpha, losses))
```

PGM solves an approximate normalized-gradient multiobjective
minimum-norm direction. Set `u_i = g_i / ||g_i||`.

The **SLSQP reference direction in selected shared params** is:

```
d_unit = sum_i alpha_i * u_i
```

However, the final loss actually used for backprop creates:

```
d_raw = sum_i alpha_i * g_i
      = sum_i alpha_i * ||g_i|| * u_i
```

Consequently, the `alpha_i` that minimizes a norm for
`d_unit` need not minimize the norm of `d_raw`.
Since CSS's three task gradient norms are strongly unequal,
we must quantify whether `d_unit` and `d_raw`
actually point in different directions. They *could*
still align if task gradients are collinear.

This is **not** enough to claim a bug or novel correction:
loss normalization / unit-gradient methods already exist;
CSS may intentionally trade off original loss magnitudes.

## Exact reconstruction from saved V5 JSON; no GPU required

V5 already saved each batch's gradient norms `n_i`,
pairwise unit-gradient cosine Gram matrix `Q`, and
original PGM alpha weights `alpha_i`.

Without re-running CSS, compute

```
a = [alpha_1, alpha_2, alpha_3]
b = [alpha_1*n_1, alpha_2*n_2, alpha_3*n_3]
cos(d_unit, d_raw) =
    (a^T Q b) / sqrt((a^T Q a) * (b^T Q b))
angle_deg = arccos(cos) * 180 / pi
```

This reconstructs the **exact two directions in PGM's
selected shared-parameter space** up to normal floating
precision. It is NOT an angle in all model parameters.

Also compare each task's *unit-gradient alignment* with
the reference and actual directions:

```
align_unit_i = (Q a)_i / sqrt(a^T Q a)
align_raw_i  = (Q b)_i / sqrt(b^T Q b)
```

This can expose whether the raw-loss update tracks KL's
direction much less than the normalized projection would,
despite KL receiving a large alpha.

## Read-only run

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v51-pgm-direction-alignment

python -m unittest discover -s tests \
  -p 'test_v51_pgm_direction_alignment.py' -v

python -u -m scripts.audit_pgm_direction_alignment \
  --input results/v5_css_pgm_gradient_audit.json \
  --output results/v51_pgm_direction_alignment.json
```

No checkpoint is loaded. No PyTorch, network, data, labels,
optimizer steps, or official TEST are used by the V5.1 script.
Just JSON numeric calculations; CPU is sufficient.

### Deciding next action

- If average angle is small (e.g. near 0-10 degrees), stop
  pursuing "projection/backprop mismatch" as a practical
  hypothesis. Low KL raw norm share alone then does not
  prove a meaningful geometric inconsistency.
- If angle is consistently substantial (>20-30 degrees
  across batches), we have an internally measurable
  **optimization-definition discrepancy**, not proof
  of hurt F1. It motivates a controlled study of
  normalization-consistent gradients / alternative
  loss-scale handling, versus original CSS.
- A corrected scheme is NOT automatically Pareto optimal
  globally or better at classifying emotions. It needs
  fair controls, matched seeds and budgets, independent
  validation splits and both MELD/IEMOCAP.
- At this snapshot, model is in training dropout mode
  but has frozen parameter values from a selected epoch;
  not the actual historical gradient trajectory.
- All analyses of the old 115 VALID dialogues are exploratory
  and must not be used to optimize a new loss coefficient.

This step fulfills a critical research standard: *observe
and quantify the mechanism first, then decide whether
modification has a justified, falsifiable hypothesis*.

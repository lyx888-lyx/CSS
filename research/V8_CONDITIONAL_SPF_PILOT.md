# V8 — A controlled attempt to improve CSS metrics using conditional higher-order evidence

Status: **one prespecified training pilot; results NOT RUN**.
Date: 2026-11-17. Dataset: original CSS MELD features.
This is **an experimental baseline/idea**, not established novelty
or a guarantee of a metric improvement.

## Motivation strictly from user-run evidence

The fixed V7 3-seed original-CSS-SPF vs retrained residual-only
MELD paired ablation on 846 TRAIN/87 existing VALID dialogues
found residual-only minus full-SPF W-F1 (pp):

| Group | 10068 | 21431 | 31907 | Mean | Seed SD |
|---|---:|---:|---:|---:|---:|
| overall | +0.29 | +0.11 | +0.13 | +0.18 | 0.10 |
| stable | +0.00 | -0.45 | +0.36 | -0.03 | 0.41 |
| changed | +1.06 | +0.39 | +0.66 | +0.70 | 0.34 |
| changed_after_other | +0.98 | +0.82 | +1.17 | +0.99 | 0.18 |

**Important:** Three training seeds on the SAME previously inspected
VALID cohort are not three independent held-out datasets. Even a
consistent +0.18pp does not establish high-order SPF is unnecessary
globally. The user wants a REPEATABLE improvement beyond CSS,
not a post-hoc maximization of this cohort.

Earlier V6.3 checked original CSS SPF at an independently trained
CSS checkpoint on 668 seen TRAIN turns: mean centered polynomial/
residual class-logit norm ratio 0.0549, only 8 top1 flips if the
poly term is removed; mean absolute true-label margin delta 0.106.
Polynomial influences probabilities and margin, but its class
discriminative effect is relatively small. It is not universally
helpful or provably useless.

## Hypothesis / strict comparisons

Original CSS SPF is:

    h = residual(T,A,V) + poly(T,A,V)

Here we test a narrowly precommitted change:

    h = residual(T,A,V) + gain_i * poly(T,A,V)

We do NOT change original PGM, per-class losses, dataset features,
speaker embeddings, source Transformer, base network or classifier.

The fixed gain factor is **4.0**, selected PRIOR to running V8
from descriptive TRAIN-only V6.3 logit ratio of ~0.055:
a factor of 4 raises the nominal polynomial/residual class-logit
ratio to roughly 0.22, not ~1. This is a mechanistic
*working constant*, NOT a tuned optimum and NOT a grid search.
It might HURT classification by amplifying bad polynomial evidence.

To isolate whether any improvement is simply from amplifying
high-order contribution versus modality / other-speaker context,
we compare EXACTLY these **five arms** at the same saved CSS
features, seed and split:

1. Original full SPF = residual + 1*poly — EXISTING checkpoint
2. Retrained residual-only SPF = residual — EXISTING checkpoint
3. V8 `gain4`: residual + 4*poly, no new parameters — NEW train
4. V8 `current`: residual + g_i*poly, g_i from ONLY current
   utterance's three text/audio/visual cosine disagreements — NEW
5. V8 `other`: same parameterized gate additionally uses
   each modality's cosine distance to the **latest preceding
   other-speaker turn** and a presence flag — NEW

Trainable gate:
- Input is seven scalar features: three within-turn modality
  disagreements; three modality-wise current-to-previous-other
  distances; and 0/1 history presence.
- `current` and `other` have EXACTLY the same 7->1 linear
  parameter count; `current` masks the four historical inputs.
- `g_i = 4*(1+0.5*tanh(Linear(features)))` is in (2,6).
- Linear weights/bias begin at **exactly zero** so at epoch 0
  current/other both EXACTLY match the gain4 baseline. All
  original CSS modules are initialized BEFORE the V8 gate, so
  shared seeded initial parameters are identical to the original.
- Speaker context is generated from PRE-context projected
  text/audio/visual features, not from true emotion labels;
  only **previous** different-speaker turns are selected and
  padded turns are excluded. There is no gold change/stable tag
  in the forward pass.
- CSS encoder remains BIDIRECTIONAL; explicit gate history
  is prior-only, but the model as a whole is NOT an online
  causal predictor. Do not falsely claim online MERC.
- Current-only vs other comparisons isolate whether explicit
  preceding OTHER-speaker history adds something beyond
  modality disagreement.
- This is scalar *polynomial contribution* routing, not
  per-class true emotion gating. It cannot observe ground truth
  labels at inference.

**WARNING on novelty:** dynamic fusion gates, speaker context
and modality complementary information already have 2026 prior
art (e.g., DF-GCN KBS 2026, DnR WACV 2026), and this
pilot does NOT establish independent novelty. A paper would
need mechanism-level differentiation and independent results.

## Fixed training protocol

- MELD data `data/meld_multimodal_features.pkl`,
  identical to original CSS.
- 846 TRAIN dialogues / 87 **already-viewed** VALID
  dialogues, fixed split seed 36427.
- Original CSS architecture/SPF, original PGM and
  Adam learning rate 5e-6, batch size 32,
  max 100 epochs; best checkpoint selected only by
  validation Weighted F1 (same as established research runner).
- Train seed 10068 for pilot (matched original CSS
  and residual-only checkpoint on this same split).
- No new VALID split consumed. Three new models trained.
- Prediction labels for changed/after-other grouping
  used **only after model inference**, never for input
  features or loss weighting.
- Original official TEST has been used historically
  to reproduce CSS, but V8 will do no TEST inference
  or score-based selection.
- NO tuning factor4, tanh range, learning rate, dropout,
  epochs, or split after seeing pilot outcome.

## GO/NO-GO gate for ACTUAL metrics

The goal is a repeatable overall W-F1 increase versus
original CSS; tracked change groups must also be retained.

**Exploratory GO condition** (not statistical validation):
- V8 `other` should exceed original full SPF by
  **>=+1.0pp overall** on the same 87-dialogue VALID,
  and exceed BOTH `gain4` and `current` by
  **>=+0.5pp overall**, without loss in Changed and
  Changed-after-other weighted F1 relative to original.
  Report per-class supports; rare-class gains may be noise.
- If `other` wins, preregister **two more fixed training
  seeds 21431/31907**, an independent IEMOCAP development
  protocol, and simpler age/lag controls. These must be
  prepared before running and do not reuse the 87-dialogue
  VALID for hyperparameter changes.
- If gain4/current do as well as other, then there is
  **no evidence of incremental other-speaker effect**.
  Do not label it a cross-speaker innovation.
- If ALL arms fail or changed groups decline,
  STOP this gate method; do not scan factors such
  as 2/3/5/8 or select new validation seeds to seek a gain.
- Even if GO on old validation, publication-level gains
  require multi-seed/cross-dataset independent confirmation
  using matched controls, uncertainty measures and fair
  no-leakage checkpoint selection.

## Pull, test, smoke and run

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v8-conditional-spf-pilot

python -m unittest discover -s tests \
  -p 'test_v8_conditional_spf.py' -v

# One-epoch smoke (NOT evidence of performance):
python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 36427 \
  --v61-clean-split --v61-spf-mode full \
  --v52-pgm-mode original --v8-spf-mode other \
  --speaker-mode off --precontext-mode off \
  --epochs 1 --run-name v8_other_smoke \
  --checkpoint-dir results/v8_other_smoke

# Full same-split original-vs-3-variants comparison:
CSS_PYTHON=python bash scripts/run_v8_conditional_spf_pilot.sh
```

Result: `results/v8_conditional_spf_seed10068_split36427/five_arm_validation.json`
and three new checkpoints + training logs. Existing original
CSS and residual-only V6.1 checkpoints reused as matched
same-seed/same-split controls; do NOT compare to original CSS
73.59 or paper TEST F1 from different protocols.

## What to send back

The full `V8 group:` summary for all six groups, and
selected epochs. If smoke fails, send first traceback and
first error. No need to run any official TEST.


## User-run V8 results — 2026-11-17: **NO-GO**

Fixed protocol: MELD same 846 TRAIN / 87 previously viewed VALID
dialogues (875 VALID utterances), seed10068, original CSS/PGM,
fixed 100-epoch budget. The three newly-trained arms were compared
against the two existing V6.1 same-seed matched baselines.

| Arm | Overall W-F1 | Change-group W-F1 | Changed-after-other W-F1 | Best epoch |
|---|---:|---:|---:|---:|
| Original full CSS SPF | 76.14 | 66.79 | 70.88 | 87 |
| Retrained residual-only | **76.43** | **67.85** | **71.86** | 84 |
| Gain4 fixed x4 polynomial | 76.02 | 66.82 | 70.42 | 99 |
| Current-only modality gate | 76.07 | 66.67 | 70.66 | 97 |
| Prior-other-speaker modality gate | 75.96 | 66.70 | 70.67 | 97 |

Other-speaker gate vs original full SPF:
- Overall **-0.18 pp** (instead of required >=+1.00).
- Changed **-0.09 pp**.
- Changed-after-other **-0.21 pp**.

Other-speaker vs simpler controls:
- Current-only: **-0.11 pp** overall.
- Fixed gain4: **-0.06 pp** overall.
- Retrained residual-only: **-0.47 pp** overall,
  Changed-after-other **-1.19 pp**.

Other-group results: first 75.35% vs full 75.72%;
stable unchanged 87.68%; stable-after-other unchanged 90.12%.
There is no evidence of a benefit from this particular method
of conditioning high-order feature strength on the most recent
different-speaker modality evidence.

All three new arms selected checkpoints at late epochs
(97/99), but this cannot justify post-hoc training-budget or
gain-factor sweeps on the exhausted validation cohort.
Dataset split/seed and single pilot do NOT demonstrate
cross-dataset or cross-seed superiority/inferiority.

**Precommitted stop rule applies: NO-GO.**
Do not try x2/x3/x5 gains, learnable scaling range changes,
speaker/context gating tweaks, altered epochs, modified LR
or repeated splits on the 87-dialogue VALID to chase a score.
Preserve the negative result.

**Updated research choice:** CSS's SPF has now been interrogated
through frozen deletion (V6), fair retraining (V6.1),
dialogue bootstrap (V6.2), class-logit attribution (V6.3),
fixed three-seed retraining (V7) and controlled stronger/
conditional high-order experiments (V8), without a credible
gain beyond the original CSS. This is evidence against
continuing incremental SPF scaling/gating, **NOT evidence**
that cross-speaker information or high-order interactions
are universally useless. For the user goal of actual
MERC W-F1 improvement, next prioritize independently
verified 2026 top-venue speaker/context-dynamics papers,
ensure their code/data/checkpoint selection are reproducible,
and use a stronger research-question-aligned baseline.
Avoid official TEST during method search and avoid comparing
heterogeneous feature/extraction and validation protocols.

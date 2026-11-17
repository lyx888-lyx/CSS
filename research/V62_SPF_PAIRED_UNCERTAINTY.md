# V6.2 — quantify uncertainty on the ALREADY observed V6.1 result

## Research position

V6 original SPF decomposition on 668 TRAIN utterances:
- Polynomial:residual feature norm ratio ~0.22
- Deleting polynomial from the FROZEN checkpoint changed 0/668 predictions
- Polynomial-only frozen classifier accuracy 16.32%
- SPF order-specific 3-modality gates ~ sigmoid(1)

V6.1 then FAIRLY retrained both variants from scratch for 100 epochs,
with identical initialization, optimizer and 846 TRAIN / 87 NEW VALID
dialogues, wholly excluding the older 115 and 104 validation cohorts.
The resulting 875 new-VALID utterances gave:

| Group | N | Original full SPF W-F1 | Retrained residual-only | Delta |
|---|---:|---:|---:|---:|
| all | 875 | 76.14 | 76.43 | +0.29 pp |
| first | 242 | 75.72 | 75.31 | -0.41 pp |
| stable | 291 | 87.68 | 87.68 | 0.00 pp |
| changed | 342 | 66.79 | 67.85 | +1.06 pp |
| stable_after_other | 196 | 90.12 | 90.65 | +0.53 pp |
| changed_after_other | 273 | 70.88 | 71.86 | +0.98 pp |

Best checkpoint epochs full=87 vs residual-only=84.

**Key:** These are small differences, in ONE training seed and ONE
new validation split. It is wrong to say that SPF is proved redundant
or that residual-only fusion generalizes better.

## V6.2: NOT A NEW EXPERIMENTAL MODEL

To measure uncertainty of the EXISTING result, V6.2:
1. Reloads the two saved original-V6.1 checkpoints.
2. Checks exact checkpoint provenance and training/validation split
   to ensure full SPF and residual-only predictions are paired
   on the SAME 87 dialogues (no official TEST inference).
3. Produces all-utterance and stable/changed/after-other
   matched correct->wrong and wrong->correct migration counts.
4. Reports per-class sample support, correctly predicted counts
   and prediction frequencies.
5. Calculates a **paired cluster bootstrap**, resampling WHOLE
   dialogue IDs with replacement, never independent utterances,
   to estimate descriptive percentile 95% CIs for the residual
   minus original weighted-F1 difference. Fixed `seed=64217`,
   `reps=1500`, no tuning or model selection.
6. Checks that full and residual predictions exactly reproduce
   the user's previously reported 76.14% and 76.43% W-F1
   before computing any uncertainty.

**Important statistical limitations:**
- A 95% percentile bootstrap confidence interval is NOT proof
  of equivalence when it overlaps zero.
- With only 87 dialogue units and rare classes, interval coverage
  and subgroup CIs are uncertain.
- Dialogue bootstrap captures one source of sampling variability,
  but NOT model-training initialization seed variation, data split
  selection variation, or correlations between related dialogues.
- Weighted F1 is a nonlinear statistic and class weights change
  when resampled.
- This validation cohort is ALREADY inspected. It is not an
  independent confirmation, and these results should NOT guide
  model architecture/hyperparameter selection.

## Run once: no training or checkpoint updates

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v62-spf-paired-uncertainty

python -m unittest discover -s tests \
  -p 'test_v62_paired_uncertainty.py' -v

python -u -m scripts.audit_v61_paired_uncertainty \
  --full-checkpoint results/v61_spf_paired_split36427_trainseed10068/checkpoints/v61_full_valid_best.pt \
  --residual-checkpoint results/v61_spf_paired_split36427_trainseed10068/checkpoints/v61_residual_only_valid_best.pt \
  --replicates 1500 \
  --bootstrap-seed 64217 \
  --output results/v61_spf_paired_split36427_trainseed10068/v62_paired_uncertainty.json
```

Output prints:
- W-F1 full/residual per group, paired fixes/breaks;
- 95% dialogue-resampling CIs for differences;
- per-class support, fixes/breaks;
- full JSON to the above path.

## A principled decision after V6.2

If the paired interval is broad and overlaps zero, V6.1 does not
justify choosing a new high-order architecture. We can still
**report** that full SPF showed no observable gain on the
tested configuration. Follow with a limited preregistered,
independent cross-seed / cross-dataset ablation to assess
whether this weak marginal contribution replicates. No tuning
on the 87 inspected VALID dialogues.

Even if the interval excludes zero, it is NOT sufficient for
novelty or a broad claim across initializations. To write a
strong paper, require a concrete mechanism (e.g. learned
residual shortcut dominance) and differentiate from prior
work on adaptive high-order multimodal fusion.

No assistant-side GPU training or TEST scoring occurred.

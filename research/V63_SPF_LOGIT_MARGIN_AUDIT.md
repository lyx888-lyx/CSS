# V6.3 — Is the SPF polynomial actually used by the trained classifier?

## Evidence available BEFORE this diagnostic

From user's V6 TRAIN-only source reconstruction:
- `SPFusion` exactly reconstructs `poly + residual` (error zero).
- Feature norm ratio `||poly|| / ||residual||` ≈ 0.21–0.23.
- Frozen-model removal of polynomial changed top1 0/668 TRAIN utterances.
- Four layers' static per-order T/A/V sigmoid gates all ≈ 0.731.

From user's V6.1 paired FROM-SCRATCH retraining:
- MELD with 846 TRAIN / 87 clean new VALID and 100 epochs, seed10068.
- Original full SPF 76.14 vs residual-only 76.43 overall W-F1.
- 87 validation dialogues / 875 utterances; paired corrections 4,
  new mistakes 2 (**six correctness-state changes**; the total number\n  of distinct class-label predictions may be larger if both models\n  are wrong but choose different wrong classes).
- Changed W-F1 66.79 vs 67.85.

From user's V6.2 paired WHOLE-DIALOGUE bootstrap (1500 reps):
- Overall delta(residual-full) +0.29pp, CI [-0.34,+0.97]pp.
- Changed +1.06pp, nominal CI [+0.04,+2.39]pp; 3 fixes,
  0 newly broken correct labels.
- The changed-group interval is exploratory with only 3
  correctness fixes, subgroup overlap and multiple comparisons.
  No global significance or architectural performance superiority
  may be claimed.

## What V6.3 resolves

Top1 invariance cannot tell us whether SPF polynomial changes
the classifier's confidence, logits or TRUE-class decision margins.

The original CSS ends SPF with `all_output_layer`, a **linear**
classification head. This yields an exactly additive identity:

```
fused = polynomial + residual
logits_full = W @ polynomial + W @ residual + b
logits_without_poly = W @ residual + b
poly_logit_contribution = W @ polynomial
```

Classify **the learned full-SPF model**, not a separate
residual-only checkpoint. Unlike retraining ablation, this
is a *within-model explanatory decomposition*.

Measure:
- `||W poly - mean_classes(W poly)|| / ||W resid - mean_classes(W resid)||`
  for each utterance. Class-centering strips uniform logit shifts
  that cannot change predictions or softmax.
- TRUE-label margin `logit[y] - max_{c != y} logit[c]` before
  and after polynomial contribution. A nonzero margin change
  can coexist with unchanged top1 class.
- Absolute signed change in true-class margin, fraction of
  utterances with positive vs negative margin influence.
- Total variation distance of the softmax distributions
  `0.5 * sum_c |p_full(c) - p_no_poly(c)|`.
- Paired top1 flips, fixes/breaks, subgroup and class counts.
- Exact SPF forward and class-head logit reconstruction checks.
- Distance of the learned global SPF gates to `sigmoid(1)`
  (the source initialization). This alone is NOT an
  estimate of total parameter learning or gate usefulness.

**No data contamination:** use only the EXACT 64 V6 previously
sampled MELD TRAIN dialogues, selected by seed73117 from the
V5.2 933-TRAIN cohort. These 64 are explicitly retained in
V6.1's 846-TRAIN cohort and never enter ANY old/new VALID.
This consumes no additional future validation candidates.

Load **the V6.1 full-SPF checkpoint**:
`results/v61_spf_paired_split36427_trainseed10068/checkpoints/v61_full_valid_best.pt`.
No other model or checkpoint is evaluated. No training,
parameter changes, official TEST, or held-out VALID inference.

## Server commands

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v63-spf-logit-margin-attribution

python -m unittest discover -s tests \
  -p 'test_v63_spf_logit_margin.py' -v

python -u -m scripts.audit_spf_logit_margin \
  --checkpoint results/v61_spf_paired_split36427_trainseed10068/checkpoints/v61_full_valid_best.pt \
  --batch-size 8 \
  --output results/v63_spf_logit_margin_audit.json \
  2>&1 | tee results/v63_spf_logit_margin_audit.log
```

No GPU available? add `--cpu`. No new training.
Expected 64 dialogues / 668 utterances; the script fails
rather than silently sample something else.

## Strict interpretation

1. If the polynomial class-CENTERED logits, true-class margin
   contributions and softmax TV are near zero, **even though
   feature norms are nonzero**, that would be more direct
   evidence that learned polynomial features have small
   *classifier-level influence* at this checkpoint than
   simply comparing raw feature norms.
2. If margins/softmax change substantially even though no
   top1 decision flips, the branch is NOT invisible to
   the classifier. A claim of `SPF is unused` would be
   too strong. Such behavior may impact confidence,
   calibration and training.
3. If the polynomial improves true-label margins for some
   classes and hurts others, examine descriptive subgroup
   evidence with supports; this is not causal validation.
4. This is an audit of ONE checkpoint on TRAIN only. No
   novel method is proposed; no fine-tuning on earlier
   inspected 87/104/115 VALID dialogues; no official TEST.
5. Do NOT rush to add dynamic gates: CSS has upstream
   sample-dependent gates. 2026 top-conference literature
   review and matched-capacity/multi-seed experiments are
   still needed before claiming an independent contribution.

## Research outcome to aim for

Distinguish:
- **low class-discriminative logit effect** from 
- **nonzero margin/probability effect without top1 flips**.

Neither by itself explains WHY SPF learns this way; that
would require training-time representation/gradient traces
and independent confirmation. This step helps choose the
right mechanistic question BEFORE adding another module.

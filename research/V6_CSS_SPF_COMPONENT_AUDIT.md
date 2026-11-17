# V6 — CSS SPF source-aligned component diagnosis (NO NEW TRAINING)

## Why we stop V5.2

The fixed paired 60-epoch original-CSS-vs-PGM-direction-aligned
comparison on a new, non-overlapping 104-dialogue MELD validation cohort
showed:
- Overall weighted F1: **71.33 -> 68.39 (-2.94 pp)**.
- Stable weighted F1: 80.12 -> 79.41 (-0.71 pp).
- Changed weighted F1: **62.15 -> 56.52 (-5.63 pp)**.
- Changed-after-other: **64.58 -> 59.07 (-5.51 pp)**.

The V5.1 geometric observation (~38.42 degree mean mismatch between
PGM normalized minimum-norm direction and the original raw-weighted
loss gradient) was real for eight sampled TRAIN batches, but fixing
it **did not improve** this paired experiment.

Conclusion: do **NOT** tune scales or learning-rate hyperparameters
on that new validation cohort to recover a gain. Moreover, V5.2
matches gradient norms only on PGM-selected parameters, not on SPF,
heads, or Adam's adaptive step. The result does not isolate a pure
whole-network direction effect or prove PGM incorrect.

## New original-paper question: what does SPF actually contribute?

CSS's **SPFusion** in `model.py` uses a factorized high-order
polynomial plus a learned linear residual:

```
z_{k} = sigmoid(g_{k,T}) * gelu(T W_{k,T})
      + sigmoid(g_{k,A}) * gelu(A W_{k,A})
      + sigmoid(g_{k,V}) * gelu(V W_{k,V}) + (1 W_{k,const})
poly = signed_sqrt(prod_k z_k) @ W_out + b_out
residual = Linear([T, A, V])
fused = poly + residual
logits = all_output_layer(fused)
```

**Important source detail:** `modality_gates` are learned
per-order 3-element **global parameters**, not per-utterance
reliability predictors. Their values do not change with the
utterance or the emotional context at inference. This is a
source-grounded potential limitation, **not** proof that
dynamic gates improve CSS (or that the original authors make
any false claim).

### Read-only TRAIN-only diagnostic

- Use V5.2 **original CSS control checkpoint**, saved at
  `results/v52_pgm_paired_split24117_trainseed10068/checkpoints/v52_original_valid_best.pt`.
  This is the 58-epoch VALID-selected ORIGINAL model, **not**
  the direction-aligned treatment checkpoint.
- Reconstruct exactly `SPFusion.forward` from its own
  `last_gate` input representations via a forward hook,
  and verify the reconstruction matches the real SPF output.
- For each sampled TRAIN utterance, measure polynomial norm,
  residual norm, their ratio and cosine, and global
  learned modality gates for each polynomial order.
- Run two diagnostic *inference-only* component removal probes:
  `all_output_layer(residual)` ("without polynomial") and
  `all_output_layer(poly)` ("without residual"), compared
  with the original full prediction. Report paired error
  repairs/new errors and stable/changed/after-other breakdowns.
- These probes deliberately do **not** retrain classifier heads
  to account for changes in feature distribution. **They do
  not constitute causal or fairly retrained ablations.**
  Do NOT conclude "SPF unnecessary" merely because it is small
  or because deleting it at inference helps/hurts.
- Outputs contain per-class and subgroup diagnostics with
  counts. Small Fear/Disgust counts should not be used to
  select new loss factors.

#### Validation/test isolation

The previously inspected 115-dialogue VALID cohort (seed10068)
and the newly used 104-dialogue V5.2 VALID cohort (split24117)
are both EXCLUDED. We sample only from the 933 V5.2 TRAIN
dialogues with fixed seed **73117**.

This is a frozen-model source-mechanism diagnostic, not another
attempt to maximize F1.

## Run commands

```bash
cd /code/CSS
conda activate cmdpad

git fetch origin
git switch analysis/v6-css-spf-contribution-audit

python -m unittest discover -s tests -p 'test_v6_css_spf_audit.py' -v

# Small functional smoke, TRAIN ONLY:
python -u -m scripts.audit_css_spf_components \
  --checkpoint results/v52_pgm_paired_split24117_trainseed10068/checkpoints/v52_original_valid_best.pt \
  --batch-size 2 --max-batches 1 \
  --output results/v6_spf_smoke.json

# Full fixed audit, 8 batches of 8 sampled TRAIN dialogues:
python -u -m scripts.audit_css_spf_components \
  --checkpoint results/v52_pgm_paired_split24117_trainseed10068/checkpoints/v52_original_valid_best.pt \
  --batch-size 8 --max-batches 8 --sample-seed 73117 \
  --output results/v6_css_spf_audit.json \
  2>&1 | tee results/v6_css_spf_audit.log
```

## Evidence we need before a new architecture

1. **Source-grounded limitation**: global per-order gates
   cannot adapt their coefficients to the current speaker
   turn, contrary to the behavior of *sample-conditioned*
   reliability gating. CSS has OTHER sample-dependent
   gates upstream, so this does **not** imply CSS as a
   whole lacks adaptive modality weighting.
2. **Functional behavior**: inspect whether polynomial
   and residual both influence class predictions, and
   whether their influence varies over stable/changed
   turns or categories. This is only a preliminary
   consistency check.
3. **Novelty search**: compare class/speaker-conditioned
   dynamic gates and high-order fusion methods published
   in 2026 (prefer top conference or journal with code).
   Dynamic modality attention/gating alone is prior art.
4. **Real model comparison (later, if justified)**:
   retrained original SPF vs a conceptually grounded
   alternative at equal initialization/compute/seeds
   on new independent dialogue splits and IEMOCAP;
   fair retraining is mandatory before any performance
   claim. Do not tune repeatedly on 104 already viewed
   validation dialogues.

## How to interpret the outcome

- If the polynomial and residual both contribute
  significantly and their effects differ between
  changed and stable categories, a **conditional
  interaction** question may be worth exploring.
- If the polynomial barely changes prediction but
  has substantial norms, that suggests potential
  redundancy, not proof of poor optimization.
- If rare-class samples are underrepresented,
  describe uncertainty instead of promising improvement.
- If these observations fail to produce a specific
  falsifiable mechanism, **do not** launch V6
  network training.

No assistant-side GPU execution or test-run claims.

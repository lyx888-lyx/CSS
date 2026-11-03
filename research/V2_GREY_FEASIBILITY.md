# V2 feasibility: classical grey GM(1,1) and speaker-specific emotion dynamics

Created 2026-11-03, after negative CSS V0/V1/V1.1 experiments.
**This branch contains NO new model and NO training procedure.** Only a
speaker-history coverage audit and a research plan.

## What GM(1,1) actually means

Let nonnegative scalar observations from ONE speaker be `x^(0)(1..n)`,
ordered chronologically and measured at compatible intervals.

1. First-order accumulated generating operation (AGO):
   `x^(1)(k) = sum_{i=1}^k x^(0)(i)`.
2. Background values:
   `z^(1)(k) = 0.5 [x^(1)(k-1)+x^(1)(k)]`.
3. Estimate development coefficient `a` and grey input `b`
   in `x^(0)(k)+a*z^(1)(k)=b` using least squares.
4. The whitening equation is:
   `d x^(1)(t)/dt + a x^(1)(t) = b`.
5. Use its fitted time response to forecast the NEXT observation.

In this context, minimum 4 already-observed same-speaker scalar
points is a common classical small-data fitting requirement.
That means prediction at dialogue position k needs at least
**4 prior, same-speaker, same-dialogue** observations. Having 4 total
including current is not eligible for an honest prior forecast.

Classical GM(1,1) is sensitive to sequence form: approximately
exponential trajectories are better suited than abrupt emotional
flips. Emotion labels are categorical and do not directly form a
valid nonnegative scalar time series. Signed valence and ordinary
CSS embedding vectors also do not meet classic assumptions without
an additional, carefully audited transformation. A neural
grey-inspired state-space recurrence is NOT 'standard GM(1,1)'.

## Critical MELD data issue

The user's prior MELD validation subgroup analysis found
1044 utterances, 314 speaker-dialogue first occurrences, so
mean utterances/speaker/dialogue = 1044/314 ≈ 3.32.
This does NOT establish the percentage of targets with at least 4
prior same-speaker utterances; an empirical coverage audit is needed.
Long tracks may exist but might cover few classification targets.

## Read-only/no-training coverage audit

The script only reads speaker identity one-hots and dialogue lengths
from the MELD **TRAIN data**, split into fixed train/validation
dialogue partitions. It does not load TEST or use true emotion labels.

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch research/v2-grey-feasibility
python -m unittest discover -s tests -v
python -m scripts.audit_gm_history_feasibility \
  --split-seed 10068 --minimum-history 4 \
  --output results/grey_history_feasibility.json
```

Prints train/validation speaker-dialogue track counts, histogram of
same-speaker available past observations at each target, proportion
of utterances eligible for classic GM forecast, and number of such
targets following an intervening other-speaker turn.

Do **not** fit or evaluate classic GM on validation/test labels. For
future experiments, use only TRAIN labels for supervised model
fitting, and at inference derive historic emotion signals from
previous predictions or available current-turn non-label features.

## Candidate architectures, ordered by scientific risk

A. **Classical GM(1,1) historical-signal diagnostic**.
   Select a well-defined nonnegative scalar per speaker turn, e.g.
   predicted class probability from a leakage-free, local-utterance
   model. Fit GM on each speaker's *past* 4–6 observations and predict
   the next turn's probability; measure eligibility and forecasting
   error. Compare with last-value persistence, exponential moving
   average and simple AR(1). Not a MERC model or publishable novelty
   alone. For cold-start, fallback to local evidence.

B. **Grey-inspired first-order speaker state with exogenous speaker
   signals**. A conceptual model:
   `d z_A(t)/dt = -a_A(t) z_A(t) + b_A(t)
                   + sum_{B != A} gamma_AB(t) u_B(t)`.
   `z_A`: low-dimensional learned emotion trajectory (not label).
   `u_B`: observed prior utterance embedding for another speaker
   (NOT other person's ground-truth emotion).
   Positive damping `a_A`; interaction coefficient `gamma_AB`
   can depend on context. Only previous utterances may act as inputs
   for a past-only trajectory prediction.
   This is a **first-order state-space / multivariate grey-inspired
   equation**, not the standard GM(1,1). Shared parameters and
   cold-start fallback avoid per-dialogue least-squares instability.
   'Driver' coefficients do not prove B emotionally causes A.
   Original CSS is bidirectional; do not claim end-to-end causality
   by attaching this module to its post-context features.

C. **Prediction residual and change detection**.
   Compare a prior predicted state `z_hat_A(t^-)` with a current
   local text/audio/vision observation `z_A_obs(t)`:
   `r_A(t) = z_A_obs(t) - z_hat_A(t^-)`.
   Feed current local observation, prior, discrepancy magnitude,
   model uncertainty and speaker-gap metadata into a classifier.
   Distinguish smooth drift, abrupt shifts and unreliable predictions.
   Do NOT interpret the residual alone as an emotion-cause signal.

## Mandatory experimental controls

- Coverage first; if few turns are eligible for classic GM fitting,
  do **not** treat a classical per-speaker GM as a central mechanism.
- Forecast vs last-value, moving average and first-order AR; otherwise
  apparent gains could be explained by simpler temporal smoothing.
- For classification: original CSS, V1-other, learned state without
  grey assumptions, grey model without B, grey model with B,
  and confidence fallback, same seed/split/valid-only selection.
- Measure changed and stable subgroup performance and especially
  changed-after-other, without using true emotion labels for
  inference. New auxiliary 'shift' target can use TRAIN labels only.
- Repeat multiple seeds, compare full/small-coverage subsets and
  possibly IEMOCAP. Prior CSS full dialogue attention is
  bidirectional, so keep future-information and deployment claims
  honest.
- Do not choose mechanism solely from the 115 validation dialogues.
  Conclude negative evidence promptly if assumptions fail.

## Related research and novelty constraints (2026)

- CSS, AAAI 2026, robust multimodal backbone and PGM:
  https://ojs.aaai.org/index.php/AAAI/article/view/39602
- EICA, ICASSP 2026, inertia/contagion relational constraints:
  https://doi.org/10.1109/ICASSP55912.2026.11463143
- SLCNet, Information Fusion 2026, speaker-specific long-term
  contexts including silent moments:
  https://www.sciencedirect.com/science/article/pii/S1566253525008474
- EMART, ACL 2026, emotion wheel / affective geometry:
  https://aclanthology.org/2026.acl-long.1875/
- DF-GCN, 2026 preprint, neural ODE-driven dynamic graph:
  https://arxiv.org/abs/2603.22345
- GM(1,1) limitations (2024):
  https://doi.org/10.1155/2024/9961208
- Grey models small sample, typically >=4 observations:
  https://doi.org/10.3390/axioms10040278

Thus both first-order dynamics and inter-speaker dependence have prior
art. The only plausible contribution is a well-validated
**speaker-specific, cold-start-safe, grey-structured prior that
separates autonomous trend from exogenous cross-speaker evidence**
under explicit conditions; mere equations or extra gates are not new.

## Decision gate

Use the empirical history audit before writing any new model.
If classical GM eligibility is poor, prioritize a globally trained,
first-order grey-inspired recurrence (shared parameters), not a
per-dialogue GM(1,1) estimator. Neither route deserves full
100-epoch training before cheap forecasting baselines are compared.

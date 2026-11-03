# V2 grey-model decision gate: CPU forecasting benchmark, NOT a CSS extension

Date: 2026-11-03. Context: V0, V1, V1-other and V1.1 did not improve
MELD validation emotion-shift scores. Do not tune TEST or run
another 100-epoch CSS experiment without a falsifiable dynamic hypothesis.

## Actual coverage audit (user-run, 35 unit tests passed)

| Partition | Dialogues | Utterances | Speaker/dialogue histories | >=4 PREVIOUS observations | Eligibility |
|---|---:|---:|---:|---:|---:|
| TRAIN | 1037 | 10054 | 2844 | 2565 | 25.51% |
| VALID | 115 | 1044 | 314 | 253 | 24.23% |

Of eligible turns, 1555 TRAIN and 143 VALID occur after intervening
other-speaker speech. Most utterances (791/1044 VALID) are not
eligible for a conventional per-speaker GM(1,1) one-step forecast.
Classic GM(1,1) also requires nonnegative scalar observations, a
well-behaved trend, and a meaningful equally spaced time variable.
Per-speaker utterance numbers do not necessarily match elapsed time.

**Decision:** Classic GM(1,1) cannot serve as the MAIN module.
It might be a specialized forecast prior only when history is long
enough; use a globally trained cold-start-safe first-order state
model if the mathematical structure turns out empirically useful.

## New inexpensive benchmark

`scripts/benchmark_gm_probabilities.py` addresses: **Does classic
GM forecast a speaker's next emotion-related numeric signal better
than simple temporal alternatives?**

Protocol:
1. Fit a text-only `StandardScaler+LogisticRegression` on *only*
   1037 TRAIN dialogues / 10054 TRAIN utterances, using the supplied
   `videoText` features and ground truth TRAIN emotion labels.
2. Transform VALID utterances independently to get local emotion
   probability vectors, 7 classes.
3. At a target VALID utterance, history input comprises **only**
   the preceding 4 same-speaker probability vectors. No true
   VALID emotion labels are passed as time-series values or model
   inputs; the actual current utterance probabilities are not used
   to forecast the current turn.
4. Evaluate matched eligible targets for:
   - Last observed probability (persistence).
   - Exponential moving average alpha=0.5.
   - Regularized local classwise AR(1), per-class ridge slope.
   - Unconstrained classic classwise GM(1,1), clipped/normalized.
   - Guarded GM(1,1): per-class fallback when level ratio violates
     the standard admissibility heuristic.
   - Current-utterance text-only logistic (different information
     setting! Not a history-only forecasting baseline).
5. Compute target-label accuracy, weighted F1, multiclass Brier
   score and negative log likelihood on exactly the same eligible
   VALID utterances. Labels are used ONLY for final evaluation.
6. Report the fraction of class probability series passing the
   standard GM level-ratio heuristic, plus forecast clipping and
   fallback counts.

Class probabilities are numerically positive and normalized, but
do not guarantee near-exponential trend. The uploaded dataset's
upstream `videoText` feature generation may include dialogue
context; we have not audited that. **Do not call the results causal,
end-to-end online, or comparable to CSS 73.59% weighted F1.**
This benchmark tests only whether a gray dynamics prior is worth
further attention on the limited eligible subset.

Commands:

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch research/v2-grey-forecast-baselines
python -m unittest discover -s tests -v
python -u -m scripts.benchmark_gm_probabilities \
  --split-seed 10068 --window 4 --max-iter 180 \
  --output results/grey_local_forecast_benchmark.json
```

CPU only. If the first model warns that it did not converge,
report that warning along with the results. This run never uses
MELD's official TEST utterances or labels and has no GPU training.
Validation labels are used only for scoring after forecasting.
No existing CSS checkpoints are overwritten.

## Scientific interpretation and next choice

If GM and guarded GM **do not beat** last-value/EMA/AR(1) on
log-loss and Brier across the eligible targets, do not build
the V2 mechanism around standard GM(1,1). Even if they do,
the smaller subset and irregular turn spacing still motivate
a cold-start-safe globally learned model. Do not treat 253
eligible targets as enough to claim generalization.

A candidate V2 recurrence inspired by grey 'whitening dynamics'
but NOT identical to standard GM(1,1):

  ds_A/dt = -softplus(a_A) s_A + b_A
            + sum_{B!=A} g_{BA} u_B(t)

Use an *event-time* discretization, speaker state initialized to
global learned prior, and separate exogenous B inputs. At the
current A utterance, compare the predicted prior state with a
unimodal/trimodal **current-only** observation encoder. Classify
with the current evidence, history prediction and residual; the
model must not feed true prior emotions into inference. Study
absolute turn gap and silence events explicitly when available.

A constrained first-order dynamic model and cross-speaker
couplings have strong prior art. Related 2026 publications:
- CSS, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39602
- EICA, ICASSP 2026: https://doi.org/10.1109/ICASSP55912.2026.11463143
- DialogueSSD, IEEE TAFFC 2026:
  https://ieeexplore.ieee.org/abstract/document/11663388/
- CDAGL, ICASSP 2026, ODE speaker dynamics:
  https://ieeexplore.ieee.org/document/11464400/

**Novelty bar:** "put first-order ODE/grey model on emotion features"
is NOT a submission-ready contribution. Need mathematical
identifiability or constrained parameterization, coverage-aware
cold start, intervention/forecast validity, no label leakage and
multiple-seed controlled improvements over strong temporal
baselines and 2026 publications.

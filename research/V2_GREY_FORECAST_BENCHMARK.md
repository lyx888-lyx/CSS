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

## Actual user-run result (2026-11-03): CLASSICAL GM ROUTE REJECTED

CPU validation benchmark completed. The local text-only logistic
classifier on 10,054 TRAIN utterances emitted a
`ConvergenceWarning` at `max_iter=180`. On 253 GM-eligible
VALIDATION utterances, computed from prior speaker utterance
predicted probabilities (not true prior labels):

| History forecast | Accuracy % | Weighted F1 % | Log loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| last-value persistence | 39.13 | 37.65 | 3.3537 | 1.0081 |
| EMA alpha=.5 | 39.53 | 36.32 | **2.1201** | **0.8210** |
| local AR(1) | 36.76 | 34.83 | 4.0259 | 0.9811 |
| raw classwise GM(1,1) | 37.15 | 33.81 | 5.2799 | 1.0537 |
| level-ratio-guarded GM(1,1) | 39.13 | 37.65 | 3.3481 | 1.0074 |
| current-text-only logistic (different info) | 62.06 | 59.92 | 1.6695 | 0.5995 |

Only **28 of 1771 classwise series (1.58%)** passed the GM
level-ratio admissibility heuristic. Unconstrained GM forecasts
required clipping for **382/1771** class components (21.6%).
Guarded GM effectively falls back to persistence. The current-text
logistic uses the target's CURRENT utterance and should never be
compared as a same-information history-only forecast. None of these
figures are directly comparable to CSS full-dialogue 73.59% VALID
weighted F1.

**Interpretation:** Under this fixed experimental setup, standard
per-speaker classwise GM(1,1) is both low-coverage (~24.2% of
VALID turns have >=4 earlier own-speaker observations) and an
uncompetitive historical probability forecaster. Due to
non-converged upstream logistic, tiny eligible subset and
unverified upstream text-feature context dependence, this is NOT a
universal theorem that GM(1,1) never works on emotion; rather
there is insufficient basis for making it a central MERC mechanism.

**Stop:** No further direct classical GM tuning or growing the
validation hyperparameter search. Do not try arbitrary classwise
translations or larger rolling windows to rescue this model.

### Candidate V2 scientific question (NOT yet a validated innovation)

Can an **event-time, cold-start-safe, globally learned, first-order
grey-structured speaker prior** separate (i) autonomous within-speaker
emotional persistence, (ii) exogenous signals from other speakers
since their last turn and (iii) current observed multimodal evidence,
without use of ground-truth historical emotion labels at inference?

One possible time-dependent *proposal* (not standard GM(1,1)):

```
prior_A(t) = exp(-alpha*gap) * post_A(t_prev)
           + (1 - exp(-alpha*gap))*mu
           + sum_{j in (t_prev,t), speaker(j)!=A}
                   exp(-alpha*(t-j)) * W_other * u_j

obs_A(t)   = f_current(text_t, audio_t, visual_t)
residual   = obs_A(t) - prior_A(t)
p(y_t)     = classifier(obs_A, prior_A, residual, history_available)
post_A(t)  = posterior_update(prior_A, obs_A)
```

Here `gap` is turn count, not verified wall-clock time. `mu` is a
learned population prior for cold-start, `alpha>0` a shared
damping parameter, and `u_j` an actual earlier other-speaker
utterance feature (not the speaker's true emotion label). All
state transitions and updates must be performed in chronological
order using **uncontextualized, current-utterance-only** features,
not original CSS full-dialogue bidirectional attention outputs.
The external term is an ASSOCIATION, not proof of emotional
causation.

Before GPU training:
1. Validate dimensions, masks, chronology, no future label/features
   leakage, and exact behavior with 0/1/2 speaker turns in synthetic
   CPU unit tests.
2. A low-cost controlled classification comparison on the fixed
   train/valid dialogue split: current-only; +own lag; +EMA;
   +learned first-order state; +exogenous-other state, where each
   shares the same supervision, training budget and current feature
   encoder. Check overall/stable/changed/after-other and
   cold-start/non-cold-start disjoint subsets.
3. If adding the exogenous term provides no reproducible benefit,
   STOP. No new 100-epoch CSS run.
4. If it survives >=3 seeds/splits, evaluate a full MERC
   backbone, fair CSS same-protocol baseline, and a second dataset.
   Keep official TEST untouched until final architecture selection.

**2026 novelty caveats, source-checked:**
- EICA (ICASSP 2026) already models emotion inertia / contagion:
  https://doi.org/10.1109/ICASSP55912.2026.11463143
- CDAGL (ICASSP 2026) already uses dynamic graph ODEs for
  speaker-specific context:
  https://ieeexplore.ieee.org/document/11464400/
- DialogueSSD (IEEE TAFFC 2026) combines state-space encoders and
  sentiment shift supervision:
  https://doi.org/10.1109/TAFFC.2026.3726761
- CAS-ODE (IEEE Signal Processing Letters 2026) explores
  continuous conversation dynamics:
  https://doi.org/10.1109/LSP.2026.3699337

A simple first-order ODE, learned history gate, dynamic graph or
shift-classification head alone does NOT establish 2026 top-venue
novelty. The above candidate is only a falsifiable research program,
not a claim of top-tier novelty or improved emotion recognition.

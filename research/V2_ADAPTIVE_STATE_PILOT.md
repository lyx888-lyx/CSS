# V2 — New-information-prioritized speaker state (low-cost feasibility pilot)

**Status:** independent model code + unit tests + validation-only runner.
No model is trained by the assistant; GPU results must be obtained by
the user's server. This is NOT a CSS backbone patch or proof of novelty.
Use as a *scientific decision gate*, not as a polished paper method.

## Why this study

Previous user's MELD experiments:
- Original CSS strict dialogue-valid F1 73.59; V1-other retrained
  73.55, V1.1 consistency lambda2.5 60.16.
- Classical GM(1,1) history eligibility 253/1044 VALID (24.23%);
  28/1771 per-class historical probability components satisfy
  level ratio (1.58%).
- Historical GM forecast weighted F1 33.81 vs last-value 37.65;
  EMA gives better Brier/log loss on the 253 eligible targets.
- Therefore do NOT fit a per-dialogue classical GM(1,1).
  Investigate globally learned, first-order, cold-start-safe dynamics,
  always against simpler history controls.

## Model: five fair arms

Implementation in `research/v2_state_model.py`, trained with
`scripts/train_v2_state_probe.py`. All arms share the SAME current-only
multimodal encoder (text 1024, vision 342, audio 300 -> hidden 64)
and same classifier input dimension and readout topology.
No dialogue-wide attention, no true label as model input. History
state resets at each full dialogue and processes turns strictly in order.

| Mode | Context provided to readout |
| --- | --- |
| `current` | Current multimodal utterance only |
| `last` | Current + last same-speaker embedding |
| `ema` | Current + same-speaker EMA (weight=0.5) |
| `adaptive` | Current + first-order decay-to-population prior; current innovation controls state update |
| `adaptive_other` | As adaptive, plus a separate decayed aggregate of *other speakers' intervening utterances* |

The last model uses, conceptually:

```
p_own(t) = mu + exp(-softplus(alpha)*gap)*(state_A_prev-mu)
p_A(t) = p_own(t) + sigmoid(beta) * tanh(W_other * cross_A(t))
e_A(t) = current_only_encoder(text, audio, vision)
innovation = e_A(t) - p_A(t)
rate = sigmoid(G(e_A, p_A, abs(innovation), gap, history_present))
state_A_post = p_A(t) + rate * innovation
logits_A = C(e_A(t), available_prior, available_innovation, availability_meta)
```

Prediction happens **before** current state update.
`cross_A` resets whenever A speaks; meanwhile it accumulates
non-A utterance features that occur after the last A turn.
`gap` counts intervening turn positions, *not actual elapsed time*.
This is a learned **first-order grey-inspired state-space recurrence**,
NOT standard GM(1,1), a verified theoretical advance, or evidence
that one interlocutor causally influences another.

All arms train with the same multiclass supervised cross-entropy
(no auxiliary emotion-shift labels), AdamW, selected by the
same fixed 115-dialogue VALID weighted F1. Standard split seed 10068,
train seed 10068. There is **no official TEST access**.

## RUN STEP 1: unit tests + one smoke pass

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v2-adaptive-speaker-state-pilot
python -m unittest discover -s tests -v
python -u -m scripts.train_v2_state_probe \
  --mode adaptive_other --seed 10068 --split-seed 10068 \
  --epochs 1 --max-train-batches 2 \
  --out-dir results/v2_smoke_adaptive_other
```

This creates a *smoke-only* checkpoint, not a valid training result.
Please report exceptions and the `SELECTED mode=...` line.
Only after passing this code-level test, run the five-arm suite.

## RUN STEP 2: matched five-arm 12-epoch pilot

Run in tmux, using the exact Python executable of the active conda
environment (no assumptions about miniconda install directory):

```bash
cd /code/CSS
conda activate cmdpad
PY_EXE=$(which python)
tmux new-session -d -s v2_states \
  "cd /code/CSS && CSS_PYTHON='$PY_EXE' V2_EPOCHS=12 bash scripts/run_v2_state_suite.sh"
```

Only **12 epochs for each of five arms**, not 100 CSS epochs.
All modes use identical hyperparameters; 12 epochs is an exploratory
budget (no claimed convergence). Tuning all five modes on one split
may overfit; this is a method-selection pilot, NOT final benchmark.

To inspect after completion:

```bash
cd /code/CSS
R=$(cat results/v2_speaker_state_suite_last_run.txt)
cat "$R/status.txt"
cat "$R/comparison.txt"
```

Logs `$R/<mode>/train.log`, best validation checkpoints
`$R/<mode>/valid_best.pt`, full per-class and subgroup reports
`$R/<mode>/summary.json`, structured combined `$R/summary.json`.
On failure, `status.txt` says FAILED and earlier files remain.

Each report includes accuracy, weighted/macro F1 for:
`all`, `first`, `stable`, `changed`,
`stable_after_other`, `changed_after_other`, plus per-class F1
and short-history buckets 0,1,2,>=3 prior same-speaker utterances.
True labels only define these groups AFTER prediction.

## Interpretation and go/no-go gate

- `last` and `ema` ask if history helps beyond current evidence.
- `adaptive` vs `ema` asks if learned first-order update helps,
  not whether classical GM(1,1) works.
- `adaptive_other` vs `adaptive` tests whether other-speaker
  intervening features add value, especially on emotion changes.
- A strong `all` but lower `changed` score fails the core goal.
- Claims about 0/1/2-turn cold starts require actual subgroup results.
- An improvement at one seed is an exploratory signal ONLY; follow
  with >=3 seeds, corresponding baseline runs, IEMOCAP and appropriate
  2026 state-space/ODE/interaction research comparisons.
- If adaptive does not beat EMA, or other does not improve adaptive,
  DO NOT build a large CSS integration. Freeze the negative result.

Crucial: The dataset's precomputed RoBERTa/audio/video feature
generation has not been audited for possible UPSTREAM context
information. The V2 code itself never mixes future utterances, but
that does NOT prove raw features are deployable online. Do not
claim full-stream causal inference without source feature audit.

## Existing 2026 overlap (novelty checks)

- EICA, ICASSP 2026, emotion inertia / contagion:
  https://doi.org/10.1109/ICASSP55912.2026.11463143
- CDAGL, ICASSP 2026, ODE dynamic speaker graph:
  https://ieeexplore.ieee.org/document/11464400/
- DialogueSSD, IEEE TAFFC 2026, state-space + shift auxiliary task:
  https://doi.org/10.1109/TAFFC.2026.3726761

No generic first-order memory gate, exponential decay, or external
speaker features suffice for a novel paper contribution. V2's
purpose is to test whether a GREY-INSPIRED inductive bias deserves
further mathematically distinct, reproducible research.

## User-run complete 12-epoch five-arm pilot (2026-11-03)

The user reports completed full train/validation split,
single seed=10068, split seed=10068, 115 validation dialogues,
with official TEST untouched.

| Mode | Overall weighted F1 | Stable | Changed | Changed-after-other | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 72.12 | 79.35 | 60.83 | 61.02 | 6 |
| last | 72.29 | 81.70 | 60.95 | 60.73 | **12** |
| ema | **72.48** | **82.24** | **61.47** | **62.26** | 4 |
| adaptive | 71.90 | 81.14 | 60.14 | 60.30 | 6 |
| adaptive_other | 72.11 | 80.28 | 60.92 | 61.52 | **12** |

Observation: at this one seed, EMA beats the current-only model on
overall weighted F1 (+0.36 pp), stable (+2.89 pp), changed (+0.64 pp)
and after-other changed (+1.24 pp). Neither learned adaptive state
nor the exogenous-other branch beats EMA.

Adding other-speaker input to separately trained adaptive:
overall +0.21 pp, changed +0.78 pp,
changed-after-other +1.22 pp, but still worse than EMA.
These are model comparisons, **not** intervention estimates of an
other-speaker effect. `last` and `adaptive_other` select the
last available epoch (12); thus results might be training-budget
limited. Do not run a full 100-epoch CSS integration on this evidence.

Short-history bucket weighted F1:

| Mode | 0 prior own turns | 1 | 2 | >=3 |
| --- | ---: | ---: | ---: | ---: |
| current | **78.63** | 71.38 | 72.66 | 66.30 |
| last | 76.27 | **73.42** | **75.24** | 66.78 |
| ema | 75.50 | 73.37 | 73.01 | **68.79** |
| adaptive | 76.99 | 71.03 | 71.69 | 68.00 |
| adaptive_other | 77.29 | 70.95 | 74.89 | 66.74 |

These bins are disjoint by amount of *previous own-speaker history*.
The different `0` scores arise from independently trained
parameters; they are not evidence that a history branch acted
on turns with no own-speaker history. In adaptive_other, a speaker
can have prior OTHER-speaker evidence even at first own turn.

### Read-only next diagnostic: paired error migration + external path

We added:
- `scripts/diagnose_v2_state_paths.py`: read four trained
  checkpoints and evaluate on the SAME 115 validation dialogues,
  alongside a temporary inference intervention that zeros the
  `adaptive_other` model's learned `other_projection` output.
  Reports 5 model score rows and paired `fixed`/`broken` counts
  for EMA vs adaptive models and external-path ablation. Also
  reports learned damping and raw other-speaker gain at selected
  checkpoint.
- `tests/test_v2_state_paths.py`: synthetic CPU tests that
  validate cohort matching, feature-path hook restoration and
  paired error migration.

```bash
cd /code/CSS
conda activate cmdpad
git pull --ff-only origin experiment/v2-adaptive-speaker-state-pilot
python -m unittest discover -s tests -v
R=$(cat results/v2_speaker_state_suite_last_run.txt)
python -m scripts.diagnose_v2_state_paths \
  --suite "$R" --output "$R/paired_errors.json"
```

Do not overinterpret the external-vector intervention. The same
trained classifier still sees availability flags for other-speaker
history, and its full model state parameters remain trained for
other inputs. This intervention is NOT a separately trained
`adaptive` variant or human emotional causal effect.
Only proceed with multi-seed/longer-training controls if evidence
and the research question justify them; freeze V2 adaptive
hypothesis if EMA continues to dominate.


## Paired V2 validation diagnosis (2026-11-04) — NEGATIVE

User ran `scripts/diagnose_v2_state_paths.py` on the SAME 1,044 VALID
utterances across trained checkpoints; official TEST was untouched.

| Variant | Overall weighted F1 | Stable | Changed | Changed-after-other |
| --- | ---: | ---: | ---: | ---: |
| current | 72.12 | 79.35 | 60.83 | 61.02 |
| ema | **72.48** | **82.24** | **61.47** | **62.26** |
| adaptive | 71.90 | 81.14 | 60.14 | 60.30 |
| adaptive_other | 72.11 | 80.28 | 60.92 | 61.52 |
| adaptive_other with other_projection output zeroed, inference only | 72.25 | 80.42 | 61.00 | 61.35 |

Paired correctness migration on SAME 1044 utterances:

| Pair (source -> target) | Wrong-to-correct | Correct-to-wrong | Net correct | Predicted labels changed |
| --- | ---: | ---: | ---: | ---: |
| current -> ema | 32 | 31 | +1 | 81 |
| ema -> adaptive | 25 | 31 | -6 | 80 |
| ema -> adaptive_other | 38 | 44 | -6 | 120 |
| adaptive -> adaptive_other | 28 | 28 | 0 | 96 |
| adaptive_other -> zero external projection | 6 | 5 | +1 | **16** |

`adaptive_other` trained parameters:
- Positive decay `softplus(raw_decay)=0.125943`.
- Other gain `sigmoid(raw_other_gain)=0.143835`.
- `other_projection.weight` Frobenius norm = 3.272761.

Interpretation: the trained external history projection affected
only 16/1044 (=1.53%) **argmax label decisions** in the zero-path
inference probe. The remaining 1028 labels may still have changed
probability distributions; this diagnostic does NOT measure
distributional effect or all other-history information. In this
probe, removing the external vector fixed 6, broke 5, and changed
5 incorrect predictions to other incorrect labels. Since the
external path can contribute through training and metadata remains,
this is **not** the same as training an other-free architecture.

The `adaptive` and `adaptive_other` separately trained-model
comparison (net 0 correct, 96 changed labels) cannot isolate the
causal contribution of external utterances. Despite last and
adaptive_other selecting epoch 12 (potential training-budget
limits), there is **no robust one-seed evidence** that this
adaptive first-order speaker dynamic beats EMA or enhances
emotion-shift recognition. EMA's weighted F1 advantage over
current maps to only **one additional correctly classified
utterance** at this seed. No reliable overall or change-specific
improvement has been established.

**Decision: STOP adding gates, history-difference terms, large
lambda, or training a 100-epoch CSS+V2 model.** Keep EMA and
current as cheap reference baselines, archive this independent V2
probe as negative evidence, and reformulate the method problem
before new experiments. If eventually rerunning, preregister
training budget and 3+ seeds with checks on early/late best
epochs. Do not present a single seed or inference-only path
change as an architectural innovation.

For additional *read-only* subgroup evidence already in the saved
`paired_errors.json` (NO retraining):

```bash
cd /code/CSS
R=$(cat results/v2_speaker_state_suite_last_run.txt)
python - "$R/paired_errors.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
pair = "adaptive_other -> adaptive_other_zero_external_vector"
print("After zeroing external projection (source = trained V2 full):")
for g in ("all", "stable", "changed", "stable_after_other", "changed_after_other", "first"):
    v = d["paired"][pair][g]
    print(g, "n=", v["n"], "pred_changes=", v["predicted_label_changed"],
          "fixed=", v["fixed"], "broken=", v["broken"])
PY
```

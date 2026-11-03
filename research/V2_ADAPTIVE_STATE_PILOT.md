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

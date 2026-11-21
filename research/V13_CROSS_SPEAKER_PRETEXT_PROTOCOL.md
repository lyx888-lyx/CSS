# V13 — A-B-A predictive self-supervision (CSS-only)

Date: 2026-11-21. All work remains in **`/code/CSS`**.

## Evidence before V13, not a claim of success

CSS AAAI 2026 paper IEMOCAP TEST: ACC 75.42%, weighted F1 75.66%.
User V12 source-faithful full-TRAIN reproduction: ACC 75.17%, W-F1 75.28%.
V12 original/Self/Reaction W-F1:
**75.28 / 74.38 / 74.06**, where checkpoints were chosen by TEST,
as in the original author source.
Thus additive acoustic self/cross representation is **NO-GO**.
V11's apparent session-4 VALID W-F1 improvement failed to generalize
to the complete author-protocol training comparison.

V13 changes the TRAINING OBJECTIVE, not the CSS inference architecture.
The hypothesis is falsifiable: auxiliary prediction using intervening
B's features should help beyond predicting from prior A alone.

## Training-only objective

For every utterance A_i with previous A_j and one or more
intervening B turns, gather **only prior projected audio features**
for the prediction source:
- `a_j` = original CSS 1024-D acoustic projection for previous A
- `b` = mean of acoustic projections of B utterances STRICTLY
  between A_j and A_i
- `a_i` = actual current A projection, used only as
  a `detach()`/stop-gradient learning TARGET, never in source.

A shared-size auxiliary network (Linear(2048,256), GELU,
Linear(256,1024)) predicts a normalized target using InfoNCE
over the eligible A targets within the current mini-batch,
temperature fixed at 0.2:

- `self` control: input `[a_j, 0]`; same total parameters
  as cross (half of the first-layer inputs structurally zero).
- `cross` treatment: input `[a_j, b]`; explicit intervening
  other-speaker representation.

Only if >=2 eligible targets are in the minibatch do we
compute InfoNCE. Otherwise the loss is exact zero.
Additional term: `0.05 * InfoNCE` added **AFTER original
PGM has computed its three original gradient-balancing weights**.

Training uses the ORIGINAL CSS model forward, loss terms, PGM,
optimizer, SDT feature input and official TRAIN partition.
The pretext head only exists in training and is stored separately
from CSS inference weights: **zero test-time architecture change**.
This also makes the 75.28% V12 base checkpoint a valid frozen
source-comparable baseline (separate full-run training from scratch
for self/cross at seed61078).

No emotion gold labels, past gold emotion changes, future
A audio in the predictor SOURCE, or TEST metadata enters
the auxiliary objective. Its target is the observed A audio,
not a supervised label. Because original CSS's dialogue
transformer attends bidirectionally, and because correlation
need not mean causation, do NOT claim evidence of physical
emotion influence merely from higher classification F1.

## Fixed full-TRAIN source-comparable benchmark

- IEMOCAP original full TRAIN sessions 1–4: 120 dialogues,
  5810 labeled utterances; official TEST Session 5: 1623.
- Original CSS parameters: seed61078, 200 epochs, batch32,
  lr1e-4, weight decay1e-5, dropout0.5, hidden1024,
  heads8, rank16, SPF order3, temp1 and weighted PGM.
- The original trainer was changed only by an OPTIONAL keyword:
  `train_or_eval_model(..., auxiliary_loss_fn=None)`.
  Default None preserves the original training code, loss
  and gradient computation exactly.
- `self` and `cross` models have identical ORIGINAL CSS
  initialization, identical self/cross head initialization
  and parameter counts, and identical full-data sampler
  initialization. The new head's random initialization is
  isolated, restoring CPU/GPU RNG states afterwards.
- Coefficient 0.05, temperature 0.2, bottleneck256 and
  200 epochs are FIXED exploratory parameters. **No TEST
  hyperparameter tuning or choosing a model using TEST**.
- Historical source chooses the best TEST W-F1 checkpoint
  every epoch. V13 faithfully follows source protocol for
  benchmark numeric comparability BUT must flag every result
  `test_used_to_select_checkpoint=true` and
  `clean_independent_test=false`.
- Session 5 TEST has already been seen in V11 and V12.
  Thus even positive full-TRAIN numbers are **NOT independent
  confirmatory evidence**. Future publication claims need
  an independently locked protocol / new dataset or
  evaluation data not involved in method design.

## Run, CSS only

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v13-cross-speaker-pretext

# 1. Check tensor logic & training hook (no IEMOCAP TEST)
python -m unittest discover -s tests \
  -p 'test_v13_cross_speaker_pretext.py' -v

# 2. Checks full TRAIN and the previously saved V12 original baseline.
#    Reports the number of eligible A-B-A targets from TRAIN only.
python -u -m scripts.run_v13_cross_speaker_pretext --preflight

# 3. One full TRAIN epoch, no TEST loader/inference/scoring.
python -u -m scripts.run_v13_cross_speaker_pretext \
  --arm self --smoke
python -u -m scripts.run_v13_cross_speaker_pretext \
  --arm cross --smoke

# 4. Optional source-faithful historical benchmark, explicitly
#    acknowledging TEST-based best-epoch selection:
python -u -m scripts.run_v13_cross_speaker_pretext \
  --arm self --acknowledge-test-based-selection
python -u -m scripts.run_v13_cross_speaker_pretext \
  --arm cross --acknowledge-test-based-selection

python -m scripts.summarize_v13_fulltrain
```

Existing V12 full original CSS baseline required:
`results/v12_original_css_author_protocol/author_test_selected/original/report.json`
with W-F1 75.28%.

V13 outputs under:
`results/v13_cross_speaker_pretext/`
- `smoke/self/smoke.json`
- `smoke/cross/smoke.json`
- `author_test_selected/self/best.pt`, `history.json`, `report.json`
- `author_test_selected/cross/best.pt`, `history.json`, `report.json`

Code refuses to overwrite an existing run, records per-epoch
historical TEST selection transparently, and keeps all reports.

## Precommitted interpretation

For historical author-style comparison, primary measure is
weighted F1 relative to original CSS 75.28%. The paper's
75.66% is the historical published reference.

Cross-speaker incremental evidence specifically requires
`cross > self` (ideally consistently across *independently
selected* training seeds and datasets), in addition to
`cross > original CSS`. If self improves and cross does not,
the evidence supports within-speaker predictive representation
learning, NOT added other-speaker influence.

A single positive TEST-selected score (even over 75.66%)
**is not independent or publishable proof**. It remains a
hypothesis until new, prospectively locked evaluation and
no test-informed method selection. If both are negative,
record NO-GO and STOP: do not sweep coefficient, temperature,
bottleneck, epochs or speaker window on the already viewed
IEMOCAP Session-5 TEST. At that point assess the design
mechanistically on TRAIN only and use an independently
available corpus before another novelty claim.


## User-run V13 FULL original TRAIN results — 2026-11-21 (FINAL NO-GO)

The user ran both fixed self/cross V13 arms under the source-faithful
IEMOCAP protocol: TRAIN Ses01–04, 5,810 utterances; TEST Ses05, 1,623;
seed61078, 200 epochs; TEST weighted-F1 selected checkpoint per epoch.
The CSS baseline is the preserved V12 original CSS checkpoint/report.
These are **TEST-selected historical benchmark scores**, not untouched
TEST generalization.

| Arm | TEST-selected epoch | TEST ACC | TEST W-F1 | TEST macro-F1 | Δ W-F1 vs CSS |
|---|---:|---:|---:|---:|---:|
| Original CSS | 106 | 75.17 | 75.28 | 74.30 | 0.00 pp |
| V13 Self pretext | 127 | 74.61 | 74.75 | 73.88 | -0.53 pp |
| V13 Cross pretext | 127 | 74.68 | 74.83 | 73.95 | -0.45 pp |

Published CSS historical IEMOCAP W-F1=75.66%. Neither V13 arm
exceeds the **reproduced** CSS W-F1=75.28, let alone 75.66.
Cross's marginal advantage over Self is +0.08 pp W-F1;
paired correctness has only +1 net correct utterance for Cross
versus Self (12 fixed, 11 broken, 26 total top1 prediction flips).
This is far too weak to establish incremental cross-speaker information.

Paired correctness relative to CSS:
- Self: 21 fixed, 30 broken, 9 net fewer correct, 57 top1 flips.
- Cross: 25 fixed, 33 broken, 8 net fewer correct, 65 top1 flips.

Per-class F1 (CSS / Self / Cross):
- happy: 64.94 / 65.18 / 64.38
- sad: 82.30 / 81.26 / 81.34
- neutral: 77.47 / 76.57 / 76.48
- angry: 68.95 / 69.59 / 70.59
- excited: 82.65 / 80.63 / 80.71
- frustrated: 69.50 / 70.03 / 70.23

**Decision: V13 STOP / NO-GO.** Do not retune the auxiliary loss
coefficient 0.05, InfoNCE temperature 0.2, bottleneck 256,
A-B-A history window, optimizer, random seed, or checkpoint epoch
using these already observed TEST patterns. The auxiliary task
predicts acoustics, not emotion; its relationship to emotion
recognition is a *possible explanation* for the negative result,
not a verified mechanism.

Preserve exact user-generated reports on the server:
- `results/v13_cross_speaker_pretext/author_test_selected/self/report.json`
- `results/v13_cross_speaker_pretext/author_test_selected/cross/report.json`
- `results/v12_original_css_author_protocol/author_test_selected/original/report.json`

Next scientific step (not V13 parameter tuning): formulate a
distinct TRAIN-only hypothesis and test its incremental cross-speaker
information on a locked TRAIN-internal session-aware protocol.
Keep final source-comparable training on all original TRAIN sessions,
but note all future IEMOCAP Session-5 evaluation has already been
exposed during V11/V12/V13 and cannot be called untouched.

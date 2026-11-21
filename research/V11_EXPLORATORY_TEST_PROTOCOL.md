# V11: one-time exploratory official IEMOCAP TEST evaluation

## User decision (2026-11-21)

User explicitly wants to inspect how V11 performs on the official
IEMOCAP TEST instead of treating its +0.55 pp session-4 VALID
difference as a paper-level improvement. This is a reasonable
**exploratory comparison**, but it consumes independent TEST
status for this model-development line: it MUST NOT be followed by
tuning V11 to improve the visible TEST outcome while later
reporting that result as untouched TEST generalization.

## Exactly what is being compared

V11 already trained on session 1–3 TRAIN; checkpoints were selected
using session 4 VALID. Three frozen checkpoints:
- `results/v10_iemocap/paired/original_valid_best.pt`
- `results/v11_iemocap_acoustic_reaction/paired/self_valid_best.pt`
- `results/v11_iemocap_acoustic_reaction/paired/reaction_valid_best.pt`

All three share the same 90-dialogue training and 30-dialogue
validation cohort, seed 61078, 200-epoch optimization budget
and validation-only selection; model changes are solely the
self/reaction acoustic adapter. Their session-4 validation
W-F1 was 68.41 / 68.67 / 68.96 respectively.

Official TEST is IEMOCAP session 5 in the existing SDT
`data/iemocap_multimodal_features.pkl`. Test predictions
are computed for all 3 frozen checkpoints with no training
or score-based selection. This yields a fair **same-protocol**
TEST comparison of the three models but not the final paper's
full 1–4 TRAIN -> 5 TEST comparison.

CSS Table 1 reports 75.66% TEST weighted F1 and 75.42% ACC
on its paper-specific full-data evaluation protocol, which
must not be directly compared as an architecture-only win/loss
against these 1–3-trained V11 checkpoints.
Original source's test-selected training routine also has
a different checkpoint-selection provenance.

## Commands (CSS only)

```bash
cd /code/CSS
conda activate cmdpad
git pull --ff-only origin experiment/v11-speaker-relative-acoustic-reaction

python -m unittest discover -s tests \
  -p 'test_v11_test_protocol.py' -v

# Validate all 3 checkpoint configs, original train/dev indices,
# source selection metadata and exact model state keys.
# Does NOT load, iterate or score TEST.
python -u -m scripts.evaluate_v11_test_once --preflight

# Explicit, exploratory one-time official TEST inspection
python -u -m scripts.evaluate_v11_test_once --evaluate-test
```

Outputs:
`results/v11_iemocap_acoustic_reaction/test_exploratory_once/report.json`
with individual TEST predictions and labels, class-wise recall
(displayed as class ACC) and F1, overall ACC/W-F1/Macro-F1,
subgroup scores, and paired corrected/broken/top1 flips.

The script writes
`results/v11_iemocap_acoustic_reaction/test_exploratory_once/test_access_started.json`
before loading the official TEST partition. It blocks a second
run in the same output directory. If an infrastructure problem
causes failure mid-TEST, preserve the marker/log and document
the reason before any necessary repeat. Never repeatedly
evaluate scores to select the best method.

If only wanting a formally untouched final benchmark and
not a development diagnostic, **DO NOT** execute
`--evaluate-test` yet; first preregister and complete
the final architecture, seed-selection policy and full
TRAIN protocol, then perform the only independent TEST run.

## Interpretation decision

If Reaction TEST W-F1 exceeds original under identical 1–3
training conditions, label it an exploratory same-protocol
TEST difference only. If it does not exceed, report the
negative result as-is. Do not change architecture, dropout,
model-epoch selection or coefficients based on TEST numbers.

The previously seen V11 VALID improvement fails the
precommitted full success condition because
Changed-after-other decreased 49.75 -> 47.99. A positive
TEST overall alone cannot establish the intended cross-
speaker change mechanism, nor statistical or multiple-seed
robustness.

If preparing a future publishable final model after viewing
this TEST, the IEMOCAP Session-5 benchmark is no longer
an independent untouched test for THIS method development
line. A confirmatory independent cohort / clearly qualified
evaluation is necessary.

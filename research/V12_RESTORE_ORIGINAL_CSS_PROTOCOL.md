# V12 — Restore the EXACT ORIGINAL CSS IEMOCAP paper-comparable training protocol

Date: 2026-11-21. Project: **ONLY /code/CSS**.

## Why we stopped V10/V11 session-4 validation scores

CSS paper Table 1 gives IEMOCAP W-F1 75.66%, ACC 75.42%.
User earlier reproduced original CSS on the official held-out Session-5
TEST with W-F1 **75.28%**. New V11 current experiment used TRAIN
sessions 1–3 (4298 utterances), selected on VALID session 4
(1512 utterances), and tested Session 5 (1623 utterances).
That protocol is **NOT** the paper's original training protocol.
V11 exploratory TEST: Original 69.60 / Self 70.75 / Reaction 70.20
W-F1. It is NOT fair to compare 70.75 directly against paper 75.66
because V11 withheld 26.02% of original training utterances.

This V12 branch restores the original author source
`train_IEMOCAP.py` IEMOCAP protocol for fair **author-style**
comparison among original CSS, V11 Self and V11 Reaction.

## Exact source-verified original protocol

The source `train_IEMOCAP.py`:
- Calls `get_IEMOCAP_loaders(valid=0.0, batch_size=32)`.
  Thus every official TRAIN dialogue (120 dialogues, 5810 utterances)
  is used for training. It does not hold out Session 4.
- Source author DEFAULTS: seed61078, epochs200, lr1e-4,
  weight decay1e-5, dropout0.5, batch32, hidden1024,
  8 heads, rank16, polynomial order3, temperature1.
- Class weights 1/[.086747,.144406,.227883,.160585,.127711,.252668].
  Original SPF and PGM unchanged, all three SDT feature modalities.
- Uses `SubsetRandomSampler` across the 120 original TRAIN dialogue
  indices. V12 uses the *same* source `get_IEMOCAP_loaders` and
  *same* `trainer.train_or_eval_model`. It even makes the
  original empty-VALID call per epoch, for reproducibility.
- Source scores TEST every epoch, selects the best TEST
  weighted F1, with test Accuracy to break tied W-F1.
  V12 reproduces this rule EXACTLY, but explicitly labels
  the checkpoint and all reports as **TEST-selected**.

**Critical publication caveat:** Original test-best checkpoint
selection is TEST leakage. The point of this V12 path is to
restore **numeric comparability with the author benchmark**,
NOT to pretend the scores are independent unbiased evaluation.
User has already viewed one V11 TEST comparison. No further
method/loss/epoch choices should be made based on these
V12 TEST scores if claiming clean generalization.
For a research-paper-quality independent evaluation, develop a
separate prespecified selection protocol and independent cohort.

## Compared arms, same full protocol

- `original`: original CSS model; no experimental module.
- `self`: identical CSS model + V11 acoustic adapter with only
  within-speaker A(previous)->A(current) contrast active.
- `reaction`: identical CSS model + V11 acoustic adapter with
  intervening B acoustic evidence and elementwise interaction.

V11 module was previously evaluated on 1–3 sessions; this
branch **does not change its architecture or hyperparameters**
in response to previously viewed TEST scores.

Code ensures original CSS base initial parameters and global
random training stream are identical across all three arms
at seed61078; extra V11 adapter weights use isolated RNG
so they do not change original CSS DataLoader sampler
order or dropout random sequence before training.
The two V11 adapters have matched parameter counts.

The saved weights from V10/V11 session-4 protocol are
**not reused**; training starts from scratch for the entire
120 TRAIN dialogue cohort.

## Run in /code/CSS

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v12-original-css-author-protocol

python -m unittest discover -s tests \
  -p 'test_v12_author_protocol.py' -v

# No TEST scoring; verifies original 120 dialogues / 5810 turns
python -u -m scripts.run_v12_author_protocol --preflight

# One TRAIN epoch to verify model, PGM and gradients, without TEST scoring
python -u -m scripts.run_v12_author_protocol --arm reaction --smoke

# 1. Run original baseline FIRST, using historical test-best
#    checkpoint selection as in original paper source.
python -u -m scripts.run_v12_author_protocol \
  --arm original --acknowledge-test-based-selection

# Check whether it returns near the user's prior 75.28% W-F1.
# If original is still down at ~69%, STOP and report its log.
# Do NOT start Self/Reaction training before understanding why.

# 2. Only after author-style CSS baseline is reproduced:
python -u -m scripts.run_v12_author_protocol \
  --arm self --acknowledge-test-based-selection

python -u -m scripts.run_v12_author_protocol \
  --arm reaction --acknowledge-test-based-selection

# Summarize TEST-selected same-protocol results (NO new scoring)
python -m scripts.summarize_v12_author_protocol
```

The full author-style benchmark calls official TEST every epoch.
This is the source's original behavior. You explicitly opt in
with `--acknowledge-test-based-selection`. It is deliberately
distinct from the STRICT validation-only code in
`scripts/train_research.py` and the prior V11 session-4
exploratory protocol.

If interrupted, preserve partial `history.json` and the best
checkpoint. The script refuses to overwrite an existing run.
We do not silently restart with a different seed or reuse TEST
history to select a better operating point.

Results:

```text
results/v12_original_css_author_protocol/
  smoke/reaction/smoke.json
  author_test_selected/original/
    best.pt
    history.json
    report.json
  author_test_selected/self/
    best.pt
    history.json
    report.json
  author_test_selected/reaction/
    best.pt
    history.json
    report.json
```

## Decision

**First checkpoint:** the original CSS author-style reproduced
IEMOCAP TEST W-F1 needs to return to about 75.28%, close to the
published 75.66%; otherwise the new method cannot be honestly
compared numerically against paper Table 1 and the source
environment/protocol must be debugged BEFORE changing methods.

Only after the original is recovered is it meaningful to
compare the three V12 arms under this same original protocol.
A positive TEST-selected delta can motivate a fresh strict,
pre-registered independently selected protocol, but is
insufficient proof of robust contribution or conference novelty.
The previous V11 Session-4 results remain separate and labeled.

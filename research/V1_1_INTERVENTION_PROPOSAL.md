# V1.1 Research Decision — Speaker-specific history sensitivity (proposal only)

Date: 2026-11-03. **V1.1 pilot code is now implemented, but has NOT been run on MELD.**
This document records evidence, potential novelty overlap, falsifiable hypotheses,
and a controlled implementation order; it is not a claim of originality.

## Completed evidence (MELD; fixed 115-dialogue validation, train seed 10068)

| Model / intervention | Validation W-F1 | Stable W-F1 | Changed W-F1 |
|---|---:|---:|---:|
| Independently trained original CSS | 73.59 | 82.25 | 63.69 |
| Independently trained V1 both | 73.24 | 80.76 | 63.53 |
| V1 with trained both, self path disabled at inference | 73.57 | 80.72 | 64.32 |
| Independently **retrained** V1 other-only | 73.55 | 81.59 | 63.35 |

**Crucial:** The apparent changed-score gain from turning off the self
projection **at inference** did not persist after training the other-only
architecture independently. Other-only is within 0.04 percentage
points of CSS overall, but has 0.34 pp lower changed F1 and 0.66 pp
lower stable F1. Therefore no current V1 design has shown a robust
benefit. Do NOT rerun arbitrary mode combinations or report the
intervention's subgroup F1 as a valid standalone method.

The matched CSS vs V1-both error analysis found 10 fixed, 14 broken,
32 different output classes among 1044 validation utterances, with
stable group 1 fixed/6 broken. Single seed and a small validation set.

## Research question (not yet answered)

Does the model exploit **spurious cross-speaker history** for an
utterance whose own-speaker emotion persists, and can training-time
speaker-specific history sensitivity constraints reduce these
unnecessary dependencies without losing useful information for
genuine emotional shifts?

This is narrower than historical relevance attention: the target is
a *measurable change in predictions under a controlled intervention*,
not just another self/other attention gate.

## Candidate V1.1 — intervention consistency, to be piloted

1. Start from the **V1 other-only** architecture (fresh training, not
   continuation from its checkpoint). All three projected modalities
   use the existing V1 other-speaker residual; original SPF and PGM
   remain untouched.
2. For each training dialogue, obtain normal predictions
   `p_full` and a second prediction `p_mask` from the same parameters
   with the V1 **other-speaker residual projection set to zero**.
   This is an intervention on one *feature path only*, **not**
   physical deletion of another participant's full utterance, and
   is NOT a causal identification procedure. The CSS contextual
   encoders remain bidirectional.
3. For TRAIN utterances only, identify **stable-after-other** returns
   by comparing the current label with the same speaker's prior label,
   while requiring at least one intervening other-speaker turn.
   Use gold labels only as a **training loss mask**, never as a model
   feature and never at validation/inference.
4. Apply a small consistency penalty on **stable-after-other** returns:
   `L_consistency = KL(stopgrad(p_mask_eval) || p_full_eval)`.
   Compute BOTH auxiliary distributions in evaluation mode (Dropout
   disabled) to avoid confounding history intervention with stochastic
   dropout. Continue the original PGM supervised objective on its
   separate TRAIN-mode pass. The extra regularizer requires two extra
   forwards (only one has gradient), so training will be slower.
   **Do not** force changed samples to be sensitive to history, since
   a change need not have been triggered by another speaker.
5. Inference uses the normal model once; no masking, no previous
   gold labels, no second-pass overhead.

Hypothesis: suppress accidental sensitivity to other history on stable
utterances; possibly improve `stable` without hurting `changed`.
Do **not** infer emotion contagion from such an effect.

### Necessary controls before any improvement claim

- Matched fresh training of CSS, V1-other, and V1-other+consistency
  under identical fixed dialogue split/seed, validation-based
  checkpoint selection, and no test evaluation.
- `lambda=0` control and an unconditioned consistency control.
  Avoid tuning a large hyperparameter grid on just 115 dialogues.
- Report validation macro and weighted F1, subgroup accuracy/F1,
  per-class confusion, first-return counts, and robust difference
  across >=3 seeds if pilot is encouraging.
- Verify regularizer does **not** look at future labels for input;
  its TRAIN loss mask only is allowed. Validation labels used only
  for post hoc analyses.
- If there is no positive impact on `stable` or overall F1, reject
  the hypothesis and do not escalate the module.
- **Important distribution limitation:** masking the V1 extra history
  path does not remove contextual information carried by the
  pre-existing bidirectional CSS transformer. Therefore this is
  branch-dependence regularization, not a counterfactual estimate of
  one speaker's influence on another.

## Literature overlaps: 2026 work

1. PERC (Findings EACL 2026) separates intra/inter contextual and
   personality representations and contrastively aligns them:
   https://aclanthology.org/2026.findings-eacl.172/
2. EICA (ICASSP 2026) models emotion inertia/contagion alignments:
   https://doi.org/10.1109/ICASSP55912.2026.11463143
3. HiRoC (published Aug 25, 2026) selects relevant history, uses
   speaker-typed routing, and calibrates cross-modal disagreement:
   https://link.springer.com/article/10.1007/s44443-026-01163-x
4. SCoPE (2026 preprint) conditions on speaker emotion priors,
   predicts emotion shifts and dynamically weights evidence:
   https://arxiv.org/abs/2607.20445
5. ECERC (ACL 2025) studies evidence/cause attention in multimodal ERC:
   https://aclanthology.org/2025.acl-long.102/
6. SAAM-EC (EMNLP 2026) uses sufficiency-aware adversarial evidence
   masking in **emotion-cause pair extraction** (different task):
   https://aclanthology.org/2026.emnlp-main.971/

History filtering, shift-aware gating, and generic counterfactual
augmentation are **NOT** novel by themselves. This pilot must show a
specific training-time mechanism and independent benefits, and a
thorough literature audit is required before claiming novelty.

## Implementation status and first server action

The current pilot implementation includes:

- `research/v11_intervention.py`: stable-after-other TRAIN loss mask;
  temporary removal of V1 other-history residual; masked categorical KL
  with detached masked-path teacher.
- `train_MELD.py`: opt-in auxiliary TRAIN loss **added after** original
  PGM task weighting. With lambda=0, the original model loss is unchanged.
  Prints `V11_TRAIN_DIAG epoch=... stable_after_other_used=... mean_consistency_kl=...`
  for verification.
- `scripts/train_research.py`: option
  `--v11-consistency-lambda`, requires `--speaker-mode off`
  and `--precontext-mode other`; the original TEST remains unexamined.
- `tests/test_v11_intervention.py`: synthetic mask, hook, KL gradient,
  training-only and empty-mask checks.

**First run: TEST THE CODE, not 100 epochs.**

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch research/v1-1-intervention-proposal
python -m unittest discover -s tests -v
python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 10068 \
  --speaker-mode off --precontext-mode other \
  --v11-consistency-lambda 0.05 --epochs 1 \
  --run-name v11_consistency_smoke_s10068
```

Confirm the epoch output includes `V11_TRAIN_DIAG` with
`stable_after_other_used > 0`, valid dialogues=115, and a saved
checkpoint. A 1-epoch F1 is NOT an effectiveness claim.

Only after this passes should a full matched 100-epoch run be made,
and compared to the already trained V1-other checkpoint under the
same dialogue split and training seed. Any apparent improvement must
survive additional seeds and dataset checks.

**This is a feasibility pilot, NOT a verified independent novelty or
submission-ready contribution.**

## Additional preflight after first real MELD smoke test (2026-11-03)

Observed from user:
- `stable_after_other_used=2085` TRAIN utterances;
- `mean_consistency_kl=0.00002323` in epoch 1;
- `lambda=0.05`, so rough mean weighted KL
  `lambda * KL = 1.1615e-6`, versus total train loss `1.8627`;
- epoch-1 validation W-F1=32.80%, not an efficacy result.

**Important:** Comparing numerical loss values alone does NOT reveal how
strongly the consistency term updates parameters. To avoid spending
100 epochs with a potentially negligible regularizer, we added
`--v11-probe-gradient`. The diagnostic samples the first TRAIN batch
with at least one selected stable-after-other utterance each epoch.
It computes two gradients *without changing parameters*:
`g_pgm = d(L_PGM)/d(theta)` and
`g_aux = lambda * d(KL)/d(theta)`, where theta is the three V1
`other_diff.weight` matrices. It reports the ratio of L2 norms
`||g_aux|| / ||g_pgm||`. This is only a **one-batch, selected-parameter**
diagnostic, not a whole-training gradient estimate.

Run once with **a different checkpoint name** from the prior smoke:

```bash
cd /code/CSS
conda activate cmdpad
git pull --ff-only origin research/v1-1-intervention-proposal
python -m unittest discover -s tests -v
python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 10068 \
  --speaker-mode off --precontext-mode other \
  --v11-consistency-lambda 0.05 --v11-probe-gradient \
  --epochs 1 --run-name v11_gradient_probe_s10068
```

Report the `V11_GRAD_DIAG` and `V11_TRAIN_DIAG` lines. If the
auxiliary gradient ratio is extremely small, the proposed penalty may
be inert at coefficient 0.05. The next research step should be
pre-specified loss scaling or redesign, **not** an arbitrary sweep over
the same small validation set. Conversely, a nontrivial gradient is
necessary but insufficient evidence of usefulness.

No 100-epoch V1.1 run has been verified at this point.

## Decision after measured auxiliary gradients — single V1.1 lambda=2.5 pilot

Actual one-batch first-epoch gradients on three `other_diff.weight` matrices:

| Component | Norm |
| --- | ---: |
| PGM gradient | 3.63007151e-02 |
| Raw consistency KL gradient | 1.35916183e-04 |
| Weighted KL at lambda=0.05 | 6.79580917e-06 |
| Weighted KL / PGM | 1.87208686e-04 = 0.01872% |

The first observed loss KL=0.00002323, selected stable-after-other
training utterances=2085. These observations demonstrate the auxiliary
loss is active but *very weak on this batch*; this is not evidence
that the underlying regularizer is useless in later epochs.

A single **exploratory**, fixed lambda=2.5 is chosen:
`0.05 * (0.01 / 0.000187208686) ≈ 2.67`, rounded down to 2.5.
At the SAME first-batch raw gradients, that would be ~0.936% of PGM
gradient norm. This **does not** predict training gradients or accuracy.
We commit to one run at lambda=2.5, not a broad validation-set
hyperparameter search. This target of ~1% is a heuristic only.

Run the following from the checked-out branch (only ONE model trains):

```bash
cd /code/CSS
conda activate cmdpad
git pull --ff-only origin research/v1-1-intervention-proposal
python -m unittest discover -s tests -v
tmux new-session -d -s css_v11_pilot \
  'cd /code/CSS && bash scripts/run_v11_lambda2p5_pilot.sh'
```

The script preflights GPU/data and finds previously trained CSS and
V1-other checkpoints in `results/v1_meld_pair_last_run.txt`.
It trains 100 epochs with `--v11-probe-gradient` (once per epoch),
selects checkpoint via VALID weighted F1, **never selects on TEST**,
then re-evaluates CSS, V1-other, V1.1 on the identical validation split
and writes a group comparison. No baseline retraining.

To read results (including failures) after the tmux session exits:

```bash
cd /code/CSS
R=$(cat results/v11_lambda2p5_pilot_last_run.txt)
cat "$R/status.txt"
cat "$R/summary.txt"
tail -n 15 "$R/train.log"
```

The run directory contains:
- `protocol.txt` with fixed lambda and baseline checkpoint paths
- `train.log` with per-epoch KL/gradient/training + validation results
- `checkpoints/v11_lambda2p5_s10068_valid_best.pt`
- `validation_comparison.json` / `.log` for three checkpoints
- `summary.txt` with overall/stable/changed/after-other weighted F1
- `status.txt` recording RUNNING, COMPLETE, or FAILED

A failure in the summary does not destroy the checkpoint; diagnose
and recover only the summary without retraining. A one-seed result
is insufficient for a final improvement claim.


## 2026-11-03 full lambda=2.5 MELD pilot — NEGATIVE result / PAUSE

User completed all 100 epochs under the **validation-only** protocol,
seed=10068, split seed=10068, validation 115 dialogues, official TEST
untouched. Saved original checkpoint and logs remain valid.

| Variant | Overall VALID weighted F1 | Stable | Changed | Changed-after-other |
| --- | ---: | ---: | ---: | ---: |
| CSS | 73.59 | 82.25 | 63.69 | 64.42 |
| V1-other independent training | 73.55 | 81.59 | 63.35 | 64.23 |
| V1.1 consistency lambda=2.5 | **60.16** | **69.68** | **48.50** | **52.08** |

Delta V1.1 vs CSS: **-13.43 pp overall**, -12.57 pp stable,
-15.19 pp changed, -12.34 pp changed-after-other.

Run directory:
`results/v11_lambda2p5_pilot/20261103_081906`

- `train.log`: epoch-by-epoch train and validation metrics, auxiliary
  KL, gradient ratio on first eligible batch per epoch
- `validation_comparison.json`: independently re-evaluated metrics for
  CSS, V1-other, V1.1 on identical validation dialogues
- `checkpoints/v11_lambda2p5_s10068_valid_best.pt`: selected V1.1 model
- `summary.txt`: final cross-model results

**Decision:** STOP further lambda increases and blind lambda sweeps.
The first-epoch gradient-norm ratio on just three other_diff weights
cannot calibrate a training-wide loss coefficient. Selecting lambda=2.5
from that one diagnostic was insufficiently justified. A negative
single-run result does not isolate the reason (loss strength, teacher
quality, optimization conflict, or objective mismatch).

Crucial conceptual limitation: V1.1's 'intervention' zeros only the
added other_diff projections. Original CSS bidirectional context
attention *still mixes all prior and future speakers*. Therefore the
zero-path teacher is NOT a genuine other-speaker-absent counterfactual,
and the objective does not guarantee preservation of discriminative
emotion information. KL can be reduced by making output
distributions insensitive to this path, not necessarily more correct.

**Next action:** read existing training logs for best epoch, validation
trajectory and how per-batch KL/gradient ratio evolved; no training
required. Preserve failure as negative evidence. Avoid claiming that
a single regularizer setting disproves all speaker-history robustness
learning. If research continues, redesign the intervention at a
well-defined information boundary and measure its validity before
running further full experiments.

Read-only server commands:

```bash
cd /code/CSS
R=$(cat results/v11_lambda2p5_pilot_last_run.txt)
grep '^Selected epoch=' "$R/train.log"
grep '^epoch=' "$R/train.log" | awk 'NR==1 || NR%10==0 || NR==100'
grep '^V11_GRAD_DIAG' "$R/train.log" | awk 'NR==1 || NR%10==0 || NR==100'
grep '^V11_TRAIN_DIAG' "$R/train.log" | awk 'NR==1 || NR%10==0 || NR==100'
```

These results do not justify reporting V1.1 as a method improvement.

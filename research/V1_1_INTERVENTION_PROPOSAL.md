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

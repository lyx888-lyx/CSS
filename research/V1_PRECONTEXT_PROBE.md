# V1 feasibility probe: pre-context speaker-relative differences

**Stage:** exploratory first step after the original-author-protocol MELD
comparison. V0 CSS+history had TEST-selected weighted F1 **66.45%**
versus the original CSS **66.95%** (seed 10068; both best epoch 57,
full training data and TEST-based epoch selection). That test result
motivates *stopping V0 as an improvement claim*, but it must **not**
serve as a model-selection target for V1.

## The new, testable hypothesis

V0 uses features after original CSS full-dialogue contextual attention
and SPF. Its nominal self/other histories are therefore context-mixed.
V1 asks whether *speaker-relative changes can be extracted BEFORE
the original CSS contextual encoders* from separate projected text,
audio and visual representations.

For each modality:
1. construct strict **preceding** same-speaker and other-speaker masks;
2. attend independently to both histories without ground-truth labels;
3. obtain (current - own-history) and (other-history - current)
   **feature differences**;
4. gate their contributions with a residual to the current representation;
5. pass the modified modality representations into **unchanged CSS**
   temporal/cross-modal encoders, SPF and PGM training.

This V1 is **not yet a top-conference paper innovation**. PERC
(Findings EACL 2026) separates self/other context; EICA (ICASSP 2026)
studies emotional inertia/contagion; SCoPE (2026 preprint) has
shift-aware history/evidence fusion. We must demonstrate a distinct,
reproducible benefit rather than simply claim novelty of these ideas.

The features are *projected raw utterance representations*, but the
unchanged CSS encoder downstream is **bidirectional**. Thus the
overall architecture is **not causal/online**.

## Files and compatibility

- `research/precontext_shift.py`: new pre-context module.
- `model.py`: `precontext_shift_mode` argument, **None by default**
  and new module initialized only after all original CSS layers.
- `scripts/train_research.py`: `--precontext-mode off|self|other|both`;
  `--speaker-mode off` must be used with V1 to prevent stacking V0
  on top of it. The independent-dialogue-validation protocol from the
  fixed-split branch is unchanged.
- `tests/test_precontext_shift.py`: masked positions, forward/backward,
  no-future-read inside the block, seeded baseline parameter parity.

The original author's `train_MELD.py` and original CSS model
configuration are **not** modified for this V1 experiment. No test
dataset is evaluated during V1 development.

## Step 1: server smoke tests (do **not** start full 100 epochs yet)

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v1-precontext-shift-probe
python -m unittest discover -s tests -v
python -m scripts.train_research --help
```

Run a 1-epoch **baseline** and a 1-epoch **V1** with a fixed
MELD validation dialogue split (`--split-seed 10068`):

```bash
python -u -m scripts.train_research --dataset MELD --seed 10068 \
  --split-seed 10068 --speaker-mode off --precontext-mode off \
  --epochs 1 --run-name v1_smoke_css_s10068

python -u -m scripts.train_research --dataset MELD --seed 10068 \
  --split-seed 10068 --speaker-mode off --precontext-mode both \
  --epochs 1 --run-name v1_smoke_precontext_s10068
```

Expected `train_dialogues=1037`, `valid_dialogues=115`, no
test evaluation, new checkpoint files (old files not overwritten).
One-epoch scores are only a code check.

## Step 2: if Step 1 succeeds

Compare a full 100-epoch CSS-off baseline and V1 on the same split.
Do not reuse the original TEST-selected scores; for formal comparison,
use an exactly matched training protocol and an independent
validation-set-selected checkpoint. In addition to overall F1, analyze
`changed`, `changed_after_other`, `first`, class-specific scores
and parameter count.

If V1 still shows no reliable benefit, **stop** and reconsider its
interaction hypothesis instead of stacking arbitrary modules.
Only after verification across multiple seeds/splits and MELD/IEMOCAP
should any stronger originality or performance claim be considered.

## Related 2026 works (start novelty audit here)

- PERC, *Who You Are, What You Say: Intra- and Inter- Context
  Personality for Emotion Recognition in Conversation*,
  Findings of EACL 2026:
  https://aclanthology.org/2026.findings-eacl.172/
- EICA, *An Emotional Inertia-Contagion-Aware Alignment for Emotion
  Recognition in Conversations*, ICASSP 2026:
  https://doi.org/10.1109/ICASSP55912.2026.11463143
- SCoPE, *Shift-Aware Speaker-Conditioned Priors for Emotion
  Recognition in Conversations*, 2026:
  https://arxiv.org/abs/2607.20445

This architecture is a **controlled V1 pilot**, not a novelty assertion
or a state-of-the-art result.

## Step 3: Matched full 100-epoch validation pair (only after smoke test)

The first smoke test completed successfully at epoch 1; its
VALID weighted F1=32.80% is **not** an architecture-quality estimate.
To compare the original CSS and V1 fairly, use the same new-protocol
initialization/reset procedure for both models. Do **not** substitute
the older, pre-RNG-reset CSS baseline checkpoint.

Start the pair with one detached tmux session (sequential, same GPU):

```bash
cd /code/CSS
git pull --ff-only origin experiment/v1-precontext-shift-probe
python -m unittest discover -s tests -v
tmux new-session -d -s css_v1_pair \
 'cd /code/CSS && bash scripts/run_v1_meld_pair.sh'
tmux ls
```

Defaults: train seed 10068, validation split seed 10068,
100 epochs for each model, 1037 train dialogues + 115 validation
dialogues. This runs original CSS first, V1 precontext both second.
No test-set evaluation is performed.

After completion, regardless of whether the tmux session still exists:

```bash
cd /code/CSS
R=$(cat results/v1_meld_pair_last_run.txt)
cat "$R/status.txt"
cat "$R/summary.txt"
tail -n 10 "$R/css.log"
tail -n 10 "$R/v1.log"
```

`summary.txt` and `summary.json` contain validation F1, accuracy,
best epoch, paired delta and checkpoint locations. The summary also
checks the exact same validation dialogue indices, split seed and
training seed were used. `status.txt` reports RUNNING, FAILED or
COMPLETE. Files live under a timestamped `results/v1_meld_pair/`
directory and existing V0 / author-protocol checkpoints are preserved.

Only after this matched first pair should we decide whether
multi-seed, speaker-shift subgroup or modality-level ablation experiments
are warranted.

## Step 4: Explain V1's actual effect before designing V1.1

The first matched validation pair (train seed=10068, split seed=10068,
115 dialogues, 1044 utterances) produced:

| Subset | CSS weighted F1 | V1 weighted F1 | V1-CSS |
| --- | ---: | ---: | ---: |
| overall | 73.59 | 73.24 | -0.35 pp |
| first speaker turn | 76.69 | 77.55 | +0.86 pp |
| stable vs own previous turn | 82.25 | 80.76 | -1.49 pp |
| changed vs own previous turn | 63.69 | 63.53 | -0.16 pp |
| stable after another speaker | 86.27 | 84.72 | -1.55 pp |
| changed after another speaker | 64.42 | 64.53 | +0.11 pp |

The class-level F1 decreases for Disgust (-2.43 pp) and Anger
(-2.19 pp) are descriptive only and may be unstable at one split/seed.

Interpretation: the V1 pre-context difference block is not yet a
verified improvement for emotion transitions. In particular, a positive
0.11 pp change on `changed_after_other` is too small to support an
inter-speaker influence claim. Also, first speaker appearances can
have **other-speaker** history, even though they have no own history.

A no-retraining, validation-only *paired error migration* diagnosis now
quantifies how many utterances V1 corrected, broke or changed to
different wrong classes. It does not use the official TEST split.

```bash
cd /code/CSS
git pull --ff-only origin experiment/v1-precontext-shift-probe
python -m unittest discover -s tests -v
R=$(cat results/v1_meld_pair_last_run.txt)

python -m scripts.diagnose_v1_errors \
  --css "$R/checkpoints/css_valid_best.pt" \
  --v1 "$R/checkpoints/v1_precontext_valid_best.pt" \
  --output "$R/paired_errors.json"
```

The report prints `fixed` (CSS wrong -> V1 correct), `broken`
(CSS correct -> V1 wrong), `predicted_label_changed`, and net
correctness for all, stable, changed, first, after-other groups and
each true emotion class. Its unit tests use only synthetic arrays.
Do not over-interpret one seed; subsequent V1.1 interventions should
be selected **only on validation**, ideally with multiple seeds and
dialogue-level uncertainty estimates.

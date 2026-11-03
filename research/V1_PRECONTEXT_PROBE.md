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

# CSS + Speaker-History Gate (Experimental v0)

This is a **hypothesis-testing baseline**, not a demonstrated paper innovation.

## What it does

- Preserve the original CSS modality-aware and interaction-aware encoders, SPF, and PGM.
- Optionally insert `SpeakerHistoryFusion` after SPF and before the final multimodal classifier.
- For each target utterance, separately attend to **past** utterances from (a) its own speaker and (b) other speakers.
- Apply learned, nonnegative exponential-logit penalties for turn distance and per-utterance gates for combining the histories.
- Use a small residual addition rather than replacing the original representation.
- No true emotion labels are ever passed to this module.

**Important:** The original CSS encoders use *bidirectional* full-dialogue attention. Consequently, the final CSS features may already contain information from later turns. The new module is past-only **at its own input level**, but does not make the full model causal or real-time.

## A fair experiment requires a fresh baseline

The old published-protocol results (IEMOCAP 75.28%, MELD 66.95%) select the best epoch using test F1. **Do not compare those numbers directly** to this runner's independent-validation results.

`scripts/train_research.py` uses a seeded **dialogue-level** train/validation partition from the official training dialogues, by default 10% validation. It chooses the best epoch by validation weighted F1 and does not evaluate the test set during training. Test evaluation is an explicit action after architecture and hyperparameter selection.

### 1. Fast check

From repository root, after switching to this branch and activating your `css` environment:

```bash
python -m unittest discover -s tests -v
python -m scripts.train_research --help
```

The smoke tests use CPU toy tensors; they do not verify official data training.

### 2. One-epoch installation/gradient check

```bash
python -u -m scripts.train_research --dataset MELD --speaker-mode off --epochs 1 --run-name strict_baseline_smoke
python -u -m scripts.train_research --dataset MELD --speaker-mode both --epochs 1 --run-name speaker_v0_smoke
```

Do not call a one-epoch run a scientific result. Expect output to include **validation** scores and a checkpoint under `checkpoints/`; it should **not** print test-set F1.

### 3. Experiment plan (after checking the first-epoch logs)

Run the strict CSS baseline and matching variants under the same seed, split and hyperparameters:

```bash
python -u -m scripts.train_research --dataset MELD --speaker-mode off --run-name meld_strict_base
python -u -m scripts.train_research --dataset MELD --speaker-mode self --run-name meld_self
python -u -m scripts.train_research --dataset MELD --speaker-mode other --run-name meld_other
python -u -m scripts.train_research --dataset MELD --speaker-mode both --no-time-decay --run-name meld_both_no_decay
python -u -m scripts.train_research --dataset MELD --speaker-mode both --run-name meld_both_decay
```

All runs use the same model defaults as the public MELD training script and the same dialogue-level 10% validation partition. These are *initial* ablations, not a complete robustness study. Replicate with multiple seeds, also run IEMOCAP, and report validation-selected test metrics only after your method is fixed.

### 4. Only when the experiment is frozen: held-out test

```bash
python -m scripts.train_research --eval-only --checkpoint checkpoints/meld_strict_base_valid_best.pt
python -m scripts.train_research --eval-only --checkpoint checkpoints/meld_both_decay_valid_best.pt
```

Checkpoint metadata stores the train/validation split indices and architecture configuration, so model loading does not require manually matching command flags. Test should not be repeatedly inspected to choose which gate or time-decay setting to use.

## What the code does NOT prove

- It does not identify causal emotional influence, emotional contagion or the impact of a particular other speaker.
- The learnable gate is a predictive feature mixing weight, **not** a causal influence coefficient.
- Existing 2026 work including PERC and EICA already explores intra-/inter-speaker context and emotion inertia/contagion. The split alone is not a novel research contribution.
- The original PGM gradient-balancing weights are computed over the same original shared parameter groups; the new module is updated through final loss backprop but is not itself in the PGM shared-gradient set. This is deliberate for v0 and requires an ablation if later changed.
- For fully past-only experiments, the original CSS encoders would need causal masking and the baseline must be retrained under the same restriction.

## Next research decisions

Compare the ablations on validation first, paying particular attention to emotionally changing turns, minority-class Macro F1, GPU memory and inference latency. Investigate whether a real per-speaker, relation-conditioned dynamic influence representation adds value beyond this simple time-gated separation.

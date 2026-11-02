# MELD Speaker-History V0 — First-Seed Ablation Log

Date: 2026-11-02 (reported by researcher).

## Experimental protocol

- Base: CSS (AAAI 2026).
- Dataset: MELD, **strict 10% dialogue-level split from original training dialogues**.
- Seed: 10068.
- Training dialogues: 1037; validation dialogues: 115.
- All variants trained 100 epochs; select checkpoint by **validation weighted F1**.
- Original MELD official test set **not inspected** during training or model selection.
- No tests or GPU training were run by the assistant; numbers below were supplied as stdout from the researcher's server.

| Variant | Feature branch | Temporal decay | Best valid weighted F1 (%) | First best epoch | Delta from CSS (percentage points) |
| --- | --- | --- | ---: | ---: | ---: |
| A | CSS, none | — | 73.59 | 52 | 0.00 |
| B | Both self and other | enabled | **73.91** | 81 | **+0.32** |
| C | Self only | enabled | 73.71 | 59 | +0.12 |
| D | Other only | enabled | 73.21 | 57 | -0.38 |
| E | Both self and other | disabled | 73.77 | 84 | +0.18 |

B relative to E: +0.14 percentage points. The difference between B
and CSS is **small**, from **one seed on one validation split**.

## What the data support

- In this one run, combined self-/other-speaker history performs best.
- Other-only context underperformed CSS, and self-only showed a small
  positive delta. This does **not** imply that other speakers are irrelevant
  or harmful; CSS itself still uses bidirectional inter-turn context.
- A small B–E gap tentatively motivates analyzing temporal decay, but
  could be entirely sampling or optimizer noise.
- These results should NOT be described as a breakthrough or a new
  state-of-the-art result.
- Original CSS public-protocol MELD test F1 (paper 67.41% and original
  reproduction 66.95%) is NOT comparable to these validation scores.

## Next: analyze the VALIDATION partition only

Run after fetching `experiment/speaker-gate-v0`:

```bash
python -m unittest discover -s tests -v
python -m scripts.compare_validation --checkpoints \
  checkpoints/meld_strict_base_valid_best.pt \
  checkpoints/meld_both_decay_valid_best.pt \
  checkpoints/meld_self_valid_best.pt \
  checkpoints/meld_other_valid_best.pt \
  checkpoints/meld_both_no_decay_valid_best.pt \
  --output results/meld_validation_ablation.json
```

The report includes class-specific F1 and subsets where the speaker's
emotion has changed versus their previous turn. The grouping uses
validation true labels for **post-hoc analysis only**, never as model inputs.

The strict CSS backbone remains bidirectional. An attention weight or
validation subgroup association does NOT identify causal emotional
influence.

## After diagnostics: multi-seed paired runs

Repeat CSS/off and B/both with at least two additional seeds, e.g.
10069 and 10070. Compare each pair under the **same** seed and split.
Keep official test evaluation held out until the method is frozen.
Also consider IEMOCAP for cross-dataset generalization.

For fair convergence comparisons, record both best selected epoch and
full learning curves rather than relying on the last-epoch accuracy.

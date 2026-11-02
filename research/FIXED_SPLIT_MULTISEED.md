# Step 1 — Fixed-split, multi-seed reproducibility (MELD, V0)

This branch changes the **experiment protocol**, not the CSS or speaker-fusion
architecture. It builds on `experiment/speaker-gate-v0`. None of the user's
original 5 checkpoints are overwritten by the commands below.

## Why this change?

Earlier `--seed` simultaneously determined:
1. which training dialogues went to the validation set;
2. model initialization, dropout and training minibatch order.

To test whether a small ~0.32 percentage point gain persists, we want the
**same** 115 MELD validation dialogues across seeds, changing only stochastic
training factors. The new arguments are:

- `--split-seed 10068`: fixed dialogue split for every run;
- `--seed 10068|10069|10070`: random model initialization/training.

With no `--split-seed`, the script defaults to the old behavior
(`split_seed=seed`) for backward compatibility. When evaluating old
checkpoints, a missing `split_seed` key also falls back to `seed`.

We intentionally reseed the training RNG **after model construction**:
adding optional module parameters must not consume dropout's initial random
draws differently between CSS-off and CSS-on runs. This is a minor
**protocol change**; for an exact matched 3-seed study, rerun seed 10068
instead of merging its previous metrics into a new-protocol table.

## Step 1: Verify the new protocol on the user's server

```bash
cd /code/CSS
git fetch origin
git switch experiment/v0-fixed-split-multiseed
python -m unittest discover -s tests -v
python -m scripts.train_research --help
```

Check validation indices equal the old seed=10068 checkpoint
(only load checkpoints you generated yourself):

```bash
python - <<'PY'
import torch
from scripts.train_research import dialogue_split_indices
ckpt = torch.load('checkpoints/meld_strict_base_valid_best.pt',
                  map_location='cpu', weights_only=False)
_, valid = dialogue_split_indices(1152, 0.1, 10068)
assert valid == ckpt['valid_dialogue_indices']
print('PASS: same 115 validation dialogue indices as original seed=10068')
PY
```

One-epoch smoke test with deliberately NEW output names:

```bash
mkdir -p logs
python -u -m scripts.train_research --dataset MELD --speaker-mode off \
 --split-seed 10068 --seed 10069 --epochs 1 \
 --run-name protocol_smoke_css_s10069
python -u -m scripts.train_research --dataset MELD --speaker-mode both \
 --split-seed 10068 --seed 10069 --epochs 1 \
 --run-name protocol_smoke_both_s10069
```

For both stdout headers, confirm `split_seed=10068`,
`seed=10069`, `train_dialogues=1037`, `valid_dialogues=115`,
and `test_selection=never`. Smoke checkpoints must NOT be
reported as research results.

## Step 2 (only after Step 1 passes): paired full runs

Repeat both CSS-off and speaker-history both+decay for the *same*
training seeds 10068, 10069, 10070 and fixed split 10068.
Use names `fixed_css_s{SEED}` and `fixed_both_s{SEED}`;
`--epochs 100`, no test evaluation, and save logs.

Example for a single seed:

```bash
python -u -m scripts.train_research --dataset MELD --speaker-mode off \
 --split-seed 10068 --seed 10069 --epochs 100 \
 --run-name fixed_css_s10069 2>&1 | tee logs/fixed_css_s10069.log

python -u -m scripts.train_research --dataset MELD --speaker-mode both \
 --split-seed 10068 --seed 10069 --epochs 100 \
 --run-name fixed_both_s10069 2>&1 | tee logs/fixed_both_s10069.log
```

Compare paired **validation-weighted-F1** differences. Report all seeds,
mean, standard deviation, and each seed's best validation-selected epoch,
rather than only the highest run. Keep validation subgroup analyses
(`changed`, `changed_after_other`) after prediction. Do not use
official test scores to select the architecture.

After fixed-split reliability checks, consider distinct validation
splits, different datasets, and finally a single frozen test evaluation.

## Interpretation

- This only establishes how **robust** V0's effect is, not novel
  emotional causation or proof of speaker-to-speaker contagion.
- Original CSS uses bidirectional dialogue context.
- Even three seeds are limited evidence; a <1 pp change can be noise.
- `torch.load` is safe only for trusted checkpoint files.

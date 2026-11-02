# MELD overnight run — author's TEST-selected protocol

This branch is based on `experiment/speaker-gate-v0`, not on main.
It changes only the MELD training script to make its existing V0 speaker
history fusion optional, plus log/summary helpers and a summary unit test.

**WARNING:** Under the *original CSS author's code*, `valid=0.0`,
all original 1152 MELD training dialogues are used for training, and the
best epoch is selected by looking at the **TEST** weighted F1.
This matches the public training script's evaluation logic, but its
best TEST F1 is **test-selected**, not a clean held-out estimate.
Keep this result separate from previously reported validation scores.

## Overnight tmux command (after checking out this branch)

```bash
tmux new-session -d -s css_meld_author 'cd /code/CSS && bash scripts/run_author_meld_overnight.sh'
```

A single tmux session runs:
1. original CSS with `--speaker-history-mode off`, seed=10068,
   epochs=100, original learning rate, full train split, selecting best
   TEST F1 each epoch;
2. CSS+V0 self/other with time decay enabled, otherwise same
   experimental hyperparameters and TEST-best selection;
3. writes `summary.txt` and `summary.json`.

Both models are trained sequentially; they never compete for the GPU.
Python is set explicitly to `/usr/miniconda3/envs/cmdpad/bin/python`,
which can be overridden with `CSS_PYTHON`. The script fails early if
CUDA or the expected MELD data file is missing.

## Results location

The path to the latest run lives in:

```text
results/author_meld_last_run.txt
```

To read results the following day:

```bash
cd /code/CSS
RUN_DIR=$(cat results/author_meld_last_run.txt)
cat "$RUN_DIR/status.txt"
cat "$RUN_DIR/summary.txt"
tail -n 50 "$RUN_DIR/css.log"
tail -n 50 "$RUN_DIR/v0.log"
```

- `status.txt`: RUNNING / FAILED / COMPLETE
- `summary.txt`: best TEST-selected ACC, weighted F1, epoch, paired delta,
  checkpoint names and log locations
- `summary.json`: structured values suitable for parsing
- `css.log`, `v0.log`: all 100 epoch prints, classification reports,
  confusion matrices
- `checkpoints/`: each model's saved TEST-best checkpoint
- `protocol.txt`: Python executable, seed, git commit, evaluation warning

The original training script also writes or appends to
`record_<year>_<month>_<day>.pk` in the repository root.

Do NOT compare these TEST-selected metrics directly to the earlier
`1037/115` validation-only weighted F1 scores.

Run `python -m unittest discover -s tests -v` after switching branches.
To inspect the detached tmux session while it runs:

```bash
tmux attach -t css_meld_author
```

Detach with `Ctrl+b` then `d`. When the session exits on completion,
read the summary file above.

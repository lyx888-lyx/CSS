#!/usr/bin/env bash
# Use: bash scripts/run_author_meld_overnight.sh
# In tmux: tmux new-session -d -s css_meld_author 'cd /code/CSS && bash scripts/run_author_meld_overnight.sh'
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${CSS_PYTHON:-/usr/miniconda3/envs/cmdpad/bin/python}"
SEED="${CSS_AUTHOR_SEED:-10068}"
EPOCHS="${CSS_AUTHOR_EPOCHS:-100}"
RUN_DIR="results/author_meld/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/checkpoints"
printf '%s\n' "$RUN_DIR" > results/author_meld_last_run.txt

on_failure() {
    code=$?
    printf 'FAILED at %s (exit code %s)\n' "$(date -Iseconds)" "$code" | tee "$RUN_DIR/status.txt"
    printf '\nCheck logs: %s/css.log and %s/v0.log\n' "$RUN_DIR" "$RUN_DIR"
    exit "$code"
}
trap on_failure ERR

printf 'RUNNING since %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN (override CSS_PYTHON)" >&2
    false
fi
if [[ ! -f data/meld_multimodal_features.pkl ]]; then
    echo "Official MELD features not found: data/meld_multimodal_features.pkl" >&2
    false
fi
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; stopping before an unintended overnight CPU run")
print("CUDA ready:", torch.cuda.get_device_name(0), flush=True)
PY

{
  echo "Author's MELD protocol: full 1152 original training dialogues, valid=0.0,"
  echo "select best epoch by TEST weighted F1; original CSS versus CSS+V0."
  echo "This is test-selected, NOT a clean held-out generalization estimate."
  echo "Run: $RUN_DIR"
  echo "Date: $(date -Iseconds)"
  echo "Python: $PYTHON_BIN"
  echo "Seed: $SEED"
  echo "Epochs: $EPOCHS"
  echo "Git HEAD: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Original train_MELD.py defaults preserved (learning rate, SPF, PGM)."
  echo "CSS and V0 are run sequentially; both checkpoints saved in this run dir."
} > "$RUN_DIR/protocol.txt"

echo "===== [1/2] original CSS training started $(date -Iseconds) ====="
"$PYTHON_BIN" -u train_MELD.py \
    --Dataset MELD --seed "$SEED" --epochs "$EPOCHS" \
    --speaker-history-mode off \
    --save-best-dir "$RUN_DIR/checkpoints" \
    --run-name author_css_s"${SEED}" \
    2>&1 | tee "$RUN_DIR/css.log"
echo "===== [1/2] original CSS complete $(date -Iseconds) ====="

echo "===== [2/2] CSS + V0 history gate started $(date -Iseconds) ====="
"$PYTHON_BIN" -u train_MELD.py \
    --Dataset MELD --seed "$SEED" --epochs "$EPOCHS" \
    --speaker-history-mode both \
    --save-best-dir "$RUN_DIR/checkpoints" \
    --run-name author_v0_both_s"${SEED}" \
    2>&1 | tee "$RUN_DIR/v0.log"
echo "===== [2/2] CSS + V0 complete $(date -Iseconds) ====="

"$PYTHON_BIN" -m scripts.summarize_author_meld --run-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/summary.txt"
printf 'COMPLETE at %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"
echo "RESULTS: $RUN_DIR/summary.txt"
echo "STRUCTURED METRICS: $RUN_DIR/summary.json"
echo "FULL LOGS: $RUN_DIR/css.log ; $RUN_DIR/v0.log"
echo "LAST RUN LOCATION: results/author_meld_last_run.txt"

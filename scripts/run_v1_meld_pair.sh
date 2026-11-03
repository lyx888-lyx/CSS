#!/usr/bin/env bash
# Execute from repo root: bash scripts/run_v1_meld_pair.sh
# One tmux session, sequential CSS baseline and V1, 100 epochs each by default.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${CSS_PYTHON:-/usr/miniconda3/envs/cmdpad/bin/python}"
TRAIN_SEED="${V1_TRAIN_SEED:-10068}"
SPLIT_SEED="${V1_SPLIT_SEED:-10068}"
EPOCHS="${V1_EPOCHS:-100}"
RUN_DIR="results/v1_meld_pair/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/checkpoints"
printf '%s\n' "$RUN_DIR" > results/v1_meld_pair_last_run.txt

on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    printf 'FAILED at %s (exit code %s)\n' "$(date -Iseconds)" "$code" | tee "$RUN_DIR/status.txt"
    printf 'See logs under %s\n' "$RUN_DIR" >&2
  fi
}
trap on_exit EXIT
printf 'RUNNING since %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Cannot execute Python: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo 'Missing MELD data/meld_multimodal_features.pkl' >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: aborting rather than running 200 epochs on CPU")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

{
  echo "MELD paired, dialogue-validation-selected comparison"
  echo "Model A=original CSS (--precontext-mode off); B=V1 (--precontext-mode both)."
  echo "No test evaluation during training or checkpoint selection."
  echo "Train dialogue split seed: $SPLIT_SEED"
  echo "Model/train stochastic seed: $TRAIN_SEED"
  echo "Epochs per model: $EPOCHS"
  echo "Date: $(date -Iseconds)"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Python: $PYTHON_BIN"
} > "$RUN_DIR/protocol.txt"

echo "== [1/2] CSS baseline | $(date -Iseconds) =="
"$PYTHON_BIN" -u -m scripts.train_research \
  --dataset MELD --speaker-mode off --precontext-mode off \
  --seed "$TRAIN_SEED" --split-seed "$SPLIT_SEED" --epochs "$EPOCHS" \
  --checkpoint-dir "$RUN_DIR/checkpoints" --run-name css \
  2>&1 | tee "$RUN_DIR/css.log"

echo "== [2/2] CSS + V1 | $(date -Iseconds) =="
"$PYTHON_BIN" -u -m scripts.train_research \
  --dataset MELD --speaker-mode off --precontext-mode both \
  --seed "$TRAIN_SEED" --split-seed "$SPLIT_SEED" --epochs "$EPOCHS" \
  --checkpoint-dir "$RUN_DIR/checkpoints" --run-name v1_precontext \
  2>&1 | tee "$RUN_DIR/v1.log"

"$PYTHON_BIN" -m scripts.summarize_v1_pair --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/summary.txt"
printf 'COMPLETE at %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"
echo "RESULTS: $RUN_DIR/summary.txt"

#!/usr/bin/env bash
# 5-arm matched MELD V2 pilot. Default: 12 epochs per arm (not 100).
# Set CSS_PYTHON to your cmdpad env Python path if launching in tmux.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${CSS_PYTHON:-python}"
EPOCHS="${V2_EPOCHS:-12}"
SPLIT_SEED=10068
TRAIN_SEED=10068
BASE_DIR="results/v2_speaker_state_suite"
RUN_DIR="$BASE_DIR/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > results/v2_speaker_state_suite_last_run.txt

on_exit() {
  local code=$?
  if ((code != 0)); then
    printf 'FAILED %s exit=%s\n' "$(date -Iseconds)" "$code" \
      | tee "$RUN_DIR/status.txt"
    echo "Training/checkpoint logs preserved under $RUN_DIR" >&2
  fi
}
trap on_exit EXIT
echo "RUNNING $(date -Iseconds)" > "$RUN_DIR/status.txt"

if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "Missing MELD train features" >&2
  exit 1
fi
if [[ "$EPOCHS" -lt 1 ]]; then
  echo "V2_EPOCHS must be positive" >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import torch, sklearn; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)' \
  | tee "$RUN_DIR/environment.txt"

{
  echo "Five-arm V2 exploratory PILOT: current, last, ema, adaptive, adaptive_other"
  echo "Seed $TRAIN_SEED | split $SPLIT_SEED | epochs $EPOCHS per mode"
  echo "Same current-only text/audio/visual encoder / classifier topology"
  echo "Identical batch size, optimizer, validation-only checkpoint selection"
  echo "No official TEST and no ground-truth history labels provided to model"
  echo "Independent model, not direct CSS reproduction"
  echo "Git HEAD $(git rev-parse HEAD)"
  echo "Python $PYTHON_BIN"
} > "$RUN_DIR/protocol.txt"

for MODE in current last ema adaptive adaptive_other; do
  mkdir -p "$RUN_DIR/$MODE"
  echo "===== Starting $MODE at $(date -Iseconds) =====" \
    | tee -a "$RUN_DIR/progress.txt"
  "$PYTHON_BIN" -u -m scripts.train_v2_state_probe \
    --mode "$MODE" \
    --seed "$TRAIN_SEED" --split-seed "$SPLIT_SEED" \
    --epochs "$EPOCHS" --batch-size 32 \
    --hidden-dim 64 --dropout 0.15 \
    --lr 0.001 --weight-decay 0.0001 \
    --out-dir "$RUN_DIR/$MODE" \
    2>&1 | tee "$RUN_DIR/$MODE/train.log"
  echo "===== Finished $MODE =====" | tee -a "$RUN_DIR/progress.txt"
done

"$PYTHON_BIN" -u -m scripts.summarize_v2_state_suite \
  --suite "$RUN_DIR" --output "$RUN_DIR/summary.json" \
  2>&1 | tee "$RUN_DIR/comparison.txt"

echo "COMPLETE $(date -Iseconds)" | tee "$RUN_DIR/status.txt"
echo "Summary: $RUN_DIR/comparison.txt"
echo "Full result: $RUN_DIR/summary.json"

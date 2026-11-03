#!/usr/bin/env bash
# One prespecified V1.1 PILOT, lambda=2.5, train seed/split seed=10068.
# Run in a detached tmux: cd /code/CSS && bash scripts/run_v11_lambda2p5_pilot.sh
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${CSS_PYTHON:-/usr/miniconda3/envs/cmdpad/bin/python}"
SEED=10068
SPLIT_SEED=10068
EPOCHS=100
LAMBDA=2.5
RUN_DIR="results/v11_lambda2p5_pilot/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/checkpoints"
echo "$RUN_DIR" > results/v11_lambda2p5_pilot_last_run.txt

on_exit() {
  code=$?
  if [[ "$code" != 0 ]]; then
    printf 'FAILED at %s (exit code %s)\n' "$(date -Iseconds)" "$code" \
      | tee "$RUN_DIR/status.txt"
    echo "Full logs under $RUN_DIR (no previous checkpoints changed)" >&2
  fi
}
trap on_exit EXIT
printf 'RUNNING since %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python missing: $PYTHON_BIN (set CSS_PYTHON to override)" >&2
  exit 1
fi
if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "MELD features missing: data/meld_multimodal_features.pkl" >&2
  exit 1
fi
if [[ ! -f results/v1_meld_pair_last_run.txt ]]; then
  echo "Missing last V1 paired run path: results/v1_meld_pair_last_run.txt" >&2
  exit 1
fi
BASE_RUN="$(cat results/v1_meld_pair_last_run.txt)"
BASE_CSS="$BASE_RUN/checkpoints/css_valid_best.pt"
BASE_OTHER="$BASE_RUN/ablations/v1_other_retrained_s10068_valid_best.pt"
for CKPT in "$BASE_CSS" "$BASE_OTHER"; do
  if [[ ! -f "$CKPT" ]]; then
    echo "Baseline checkpoint missing: $CKPT" >&2
    exit 1
  fi
done
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available: aborting overnight run")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

{
  echo "V1.1 single prespecified lambda pilot, not a hyperparameter sweep."
  echo "Dataset MELD | train_seed=$SEED split_seed=$SPLIT_SEED"
  echo "Epochs=$EPOCHS lambda=$LAMBDA mode=V1-other"
  echo "Checkpoint selected by independent dialogue VALIDATION weighted F1."
  echo "Official MELD TEST set will NOT be evaluated."
  echo "Lambda rationale: single first-batch other_diff gradient norm ratio"
  echo "was 1.87208686e-4 at 0.05; lambda 2.5 targets ~0.94% at SAME batch."
  echo "The ratio may change during training. Gradient norms are not efficacy."
  echo "Existing CSS and V1-other checkpoints are comparison controls."
  echo "Baseline CSS: $BASE_CSS"
  echo "Baseline V1-other: $BASE_OTHER"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Python: $PYTHON_BIN"
  echo "Start: $(date -Iseconds)"
} > "$RUN_DIR/protocol.txt"

echo "== Train V1.1 (100 epochs), validation-only selection =="
"$PYTHON_BIN" -u -m scripts.train_research \
  --dataset MELD \
  --seed "$SEED" --split-seed "$SPLIT_SEED" \
  --speaker-mode off --precontext-mode other \
  --v11-consistency-lambda "$LAMBDA" --v11-probe-gradient \
  --epochs "$EPOCHS" \
  --run-name v11_lambda2p5_s10068 \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  2>&1 | tee "$RUN_DIR/train.log"

NEW_CKPT="$RUN_DIR/checkpoints/v11_lambda2p5_s10068_valid_best.pt"
if [[ ! -f "$NEW_CKPT" ]]; then
  echo "No validation-selected V1.1 checkpoint: $NEW_CKPT" >&2
  exit 1
fi

echo "== Re-evaluate three checkpoints on identical VALIDATION dialogues =="
"$PYTHON_BIN" -u -m scripts.compare_validation \
  --checkpoints "$BASE_CSS" "$BASE_OTHER" "$NEW_CKPT" \
  --output "$RUN_DIR/validation_comparison.json" \
  2>&1 | tee "$RUN_DIR/validation_comparison.log"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/summary.txt"
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
report = json.loads((run / "validation_comparison.json").read_text(encoding="utf-8"))
models = report["models"]
if len(models) != 3:
    raise RuntimeError("expected exactly three validation models")
names = ["Original CSS", "V1 other-only, retrained", "V1.1 lambda=2.5"]
scores = [m["scores"] for m in models]
print("MELD | V1.1 ONE-COEFFICIENT PILOT | VALIDATION ONLY")
print("=" * 64)
print("115 validation dialogues; TEST untouched; train/split seed 10068.")
print("Model                        Overall F1   Stable F1  Changed F1  Changed after other")
for name, s in zip(names, scores):
    print(
        f"{name:<28} {s['all']['weighted_f1']:>8.2f}% "
        f"{s['stable']['weighted_f1']:>11.2f}% "
        f"{s['changed']['weighted_f1']:>10.2f}% "
        f"{s['changed_after_other']['weighted_f1']:>16.2f}%"
    )
for control_index in (0, 1):
    control = scores[control_index]
    new = scores[2]
    print(f"V1.1 minus {names[control_index]}:")
    for group in ("all", "stable", "changed", "changed_after_other"):
        delta = (new[group]["weighted_f1"] -
                 control[group]["weighted_f1"])
        print(f"  {group}: {delta:+.2f} percentage points")
print("Every-epoch PGM / KL gradient observations: " + str(run / "train.log"))
print("Full score detail: " + str(run / "validation_comparison.json"))
print("Model checkpoints:")
for name, m in zip(names, models):
    print(f"  {name}: {m['checkpoint']}")
print("Caution: one training seed and one validation split; no novelty or")
print("generalization guarantee. Retrained baseline, not inference ablation.")
PY

printf 'COMPLETE at %s\n' "$(date -Iseconds)" | tee "$RUN_DIR/status.txt"
echo "RESULTS: $RUN_DIR/summary.txt"
echo "FULL LOG: $RUN_DIR/train.log"
echo "STRUCTURED: $RUN_DIR/validation_comparison.json"

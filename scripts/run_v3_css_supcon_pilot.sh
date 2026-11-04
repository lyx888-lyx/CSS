#!/usr/bin/env bash
# One fixed 2026-inspired CSS representation-learning pilot.
# CSS baseline is the PREVIOUS 100-epoch valid-selected checkpoint.
# New model trains 60 epochs, coefficient=0.01 and temperature=0.2.
# Never reads/evaluates official MELD TEST. Do not sweep hyperparameters.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${CSS_PYTHON:-python}"
BASE="results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt"
ROOT="results/v3_css_supcon_pilot"
R="$ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$R/checkpoints"
echo "$R" > results/v3_css_supcon_pilot_last_run.txt

on_exit() {
  code=$?
  if (( code != 0 )); then
    echo "FAILED $(date -Iseconds) exit=$code" | tee "$R/status.txt"
    echo "See: $R/train.log or $R/comparison.log" >&2
  fi
}
trap on_exit EXIT
echo "RUNNING $(date -Iseconds)" > "$R/status.txt"

if [[ ! -f "$BASE" ]]; then
  echo "Baseline missing: $BASE" >&2
  exit 1
fi
if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "Missing MELD features under data/" >&2
  exit 1
fi
{
  echo "V3 one-coefficient train-only class-balanced SupCon feasibility test."
  echo "Base CSS: $BASE"
  echo "New: CSS + train-only class-balanced supervised contrastive loss"
  echo "Seed 10068 / dialogue split seed 10068 / 60 epochs / weight 0.01 / tau 0.2"
  echo "The original CSS model is not retrained; new and existing checkpoints"
  echo "have different maximum training budgets (60 vs original 100 epochs),"
  echo "but both are selected using the same 115-dialogue VALID split."
  echo "An improvement must later be repeated with matched 100 epochs and 3 seeds."
  echo "Classification on held-out official TEST is never executed."
  echo "Git commit: $(git rev-parse HEAD)"
  echo "Python: $PYTHON_BIN"
} > "$R/protocol.txt"

"$PYTHON_BIN" -u -m scripts.train_research \
  --dataset MELD \
  --seed 10068 --split-seed 10068 \
  --speaker-mode off --precontext-mode off \
  --v3-supcon-weight 0.01 --v3-supcon-temp 0.2 \
  --epochs 60 --run-name v3_supcon_w001_s10068 \
  --checkpoint-dir "$R/checkpoints" \
  2>&1 | tee "$R/train.log"

CKPT="$R/checkpoints/v3_supcon_w001_s10068_valid_best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing new validation-selected checkpoint: $CKPT" >&2
  exit 1
fi
"$PYTHON_BIN" -u -m scripts.compare_validation \
  --checkpoints "$BASE" "$CKPT" \
  --output "$R/validation_comparison.json" \
  2>&1 | tee "$R/comparison.log"

"$PYTHON_BIN" - "$R/validation_comparison.json" <<'PY' | tee "$R/summary.txt"
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
items = d["models"]
assert len(items) == 2, "Expected baseline and V3 checkpoint"
for name, m in zip(("Original CSS", "CSS + SupCon"), items):
    s = m["scores"]
    print(f"{name}: epoch={m['selected_epoch']} "
          f"overall={s['all']['weighted_f1']:.2f}% "
          f"stable={s['stable']['weighted_f1']:.2f}% "
          f"changed={s['changed']['weighted_f1']:.2f}% "
          f"changed_after_other={s['changed_after_other']['weighted_f1']:.2f}%")
a, b = items[0]["scores"], items[1]["scores"]
print("V3 minus original CSS, percentage points:")
for k in ("all", "first", "stable", "changed",
          "stable_after_other", "changed_after_other"):
    print(f"  {k}: {b[k]['weighted_f1']-a[k]['weighted_f1']:+.2f}")
print("Only VALIDATION was checked. No claim of novelty or reproducible gain.")
PY

echo "COMPLETE $(date -Iseconds)" | tee "$R/status.txt"
echo "Summary: $R/summary.txt"

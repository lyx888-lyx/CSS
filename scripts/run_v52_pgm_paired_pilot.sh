#!/usr/bin/env bash
# V5.2 paired training pilot on a FRESH MELD TRAIN/VALID dialogue split.
# No official TEST scoring. Compare matched original CSS PGM versus
# direction-aligned PGM at same split, seed, epochs and hyperparameters.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${CSS_PYTHON:-python}"
SPLIT_SEED=24117
TRAIN_SEED=10068
EPOCHS=60
R="results/v52_pgm_paired_split${SPLIT_SEED}_trainseed${TRAIN_SEED}"

if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "Missing data/meld_multimodal_features.pkl" >&2
  exit 1
fi
if [[ -e "${R}/checkpoints/v52_original_valid_best.pt" ||
      -e "${R}/checkpoints/v52_direction_aligned_valid_best.pt" ]]; then
  echo "A paired run already exists in ${R}. Refusing to overwrite checkpoints." >&2
  exit 1
fi
mkdir -p "${R}/checkpoints"
echo "${R}" > results/v52_pgm_last_run.txt
echo "V5.2 fixed preregistered pilot: MELD train split seed ${SPLIT_SEED}, train seed ${TRAIN_SEED}, epochs ${EPOCHS}"
echo "Original and modified model are freshly initialized with the SAME seed."
echo "No --evaluate-test and no --eval-only options used."
echo "Original baseline started"

"${PY}" -u -m scripts.train_research \
  --dataset MELD --seed "${TRAIN_SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --speaker-mode off --precontext-mode off \
  --v52-pgm-mode original \
  --epochs "${EPOCHS}" \
  --run-name v52_original \
  --checkpoint-dir "${R}/checkpoints" \
  2>&1 | tee "${R}/original_train.log"

echo "Direction-aligned PGM training started"
"${PY}" -u -m scripts.train_research \
  --dataset MELD --seed "${TRAIN_SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --speaker-mode off --precontext-mode off \
  --v52-pgm-mode direction_aligned \
  --epochs "${EPOCHS}" \
  --run-name v52_direction_aligned \
  --checkpoint-dir "${R}/checkpoints" \
  2>&1 | tee "${R}/direction_aligned_train.log"

ORIGINAL="${R}/checkpoints/v52_original_valid_best.pt"
ALIGNED="${R}/checkpoints/v52_direction_aligned_valid_best.pt"
"${PY}" -m scripts.compare_validation \
  --checkpoints "${ORIGINAL}" "${ALIGNED}" \
  --output "${R}/paired_validation.json"

"${PY}" - "${R}/paired_validation.json" <<'PY'
import json
import sys

p = sys.argv[1]
report = json.load(open(p, encoding="utf-8"))
if report["split"] != "validation_only" or len(report["models"]) != 2:
    raise RuntimeError("unexpected validation comparison schema")
base, aligned = report["models"]
print("\nV5.2 PILOT — matched original CSS vs normalized-direction PGM")
print("Both runs: MELD seed=10068, fresh split=24117, max epochs=60, no TEST")
for name in ("all", "first", "stable", "changed",
             "stable_after_other", "changed_after_other"):
    b = base["scores"][name]["weighted_f1"]
    a = aligned["scores"][name]["weighted_f1"]
    if a is None or b is None:
        print(f"{name}: unavailable")
    else:
        print(f"{name:<22} CSS={b:6.2f}% aligned={a:6.2f}% delta={a-b:+.2f}pp")
print(
    "Selected epochs:",
    base["selected_epoch"], aligned["selected_epoch"],
)
print("No claim of improvement or originality until independent confirmations.")
PY

echo "V5.2 results saved in ${R}"

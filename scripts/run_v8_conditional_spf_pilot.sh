#!/usr/bin/env bash
# V8 single-seed, fixed-hypothesis pilot: compare three modifications
# to the SAME pre-existing V6.1 full/residual CSS checkpoints.
#
# Arms:
# 1. gain4 = constant high-order multiplier of four.
# 2. current = same gain and matched trainable gate using current
#    modality disagreements only.
# 3. other = same gate also sees PRECEDING other-speaker comparisons.
#
# This IS a model-training experiment, NOT an independent holdout:
# 846 TRAIN / 87 ALREADY-VIEWED VALID MELD dialogues.
# Official TEST never scored. Never tune x4 based on this cohort.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${CSS_PYTHON:-python}"
SEED=10068
SPLIT=36427
EPOCHS=100
BASE="results/v61_spf_paired_split36427_trainseed10068/checkpoints"
ROOT="results/v8_conditional_spf_seed${SEED}_split${SPLIT}"
FULL="${BASE}/v61_full_valid_best.pt"
RESID="${BASE}/v61_residual_only_valid_best.pt"

for f in "${FULL}" "${RESID}" data/meld_multimodal_features.pkl; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing matched baseline checkpoint or data: ${f}" >&2
    exit 1
  fi
done
for mode in gain4 current other; do
  if [[ -e "${ROOT}/checkpoints/v8_${mode}_valid_best.pt" ]]; then
    echo "Refusing to overwrite V8 checkpoint in ${ROOT}" >&2
    exit 1
  fi
done
mkdir -p "${ROOT}/checkpoints"

echo "V8 hypothesis: what matters for SPF, x4 strength vs conditional"
echo "modality information vs preceding-other-speaker context?"
echo "Fixed: seed=${SEED}, split=${SPLIT}, 846 TRAIN / 87 viewed VALID,"
echo "100 epochs, original PGM, same CSS init/features/optimizer."
echo "Precommitted gain4 constant, current vs other; no hyperparameter sweep."
echo "All VALID subgroup labels used only for AFTER-prediction reporting."
echo "NO OFFICIAL TEST EVALUATION."

for mode in gain4 current other; do
  echo "=== V8 training: ${mode} ==="
  "${PY}" -u -m scripts.train_research \
    --dataset MELD --seed "${SEED}" --split-seed "${SPLIT}" \
    --v61-clean-split --v61-spf-mode full \
    --v52-pgm-mode original --v8-spf-mode "${mode}" \
    --speaker-mode off --precontext-mode off \
    --epochs "${EPOCHS}" --run-name "v8_${mode}" \
    --checkpoint-dir "${ROOT}/checkpoints" \
    2>&1 | tee "${ROOT}/v8_${mode}.log"
done

"${PY}" -m scripts.compare_validation \
  --checkpoints \
    "${FULL}" \
    "${RESID}" \
    "${ROOT}/checkpoints/v8_gain4_valid_best.pt" \
    "${ROOT}/checkpoints/v8_current_valid_best.pt" \
    "${ROOT}/checkpoints/v8_other_valid_best.pt" \
  --output "${ROOT}/five_arm_validation.json"

"${PY}" - "${ROOT}/five_arm_validation.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["split"] == "validation_only" and len(r["models"]) == 5
names = ["CSS full", "CSS residual", "x4 polynomial",
         "Current-modality gate", "Prior-other-speaker gate"]
for group in ("all", "first", "stable", "changed",
              "stable_after_other", "changed_after_other"):
    scores = [m["scores"][group]["weighted_f1"] for m in r["models"]]
    n = r["models"][0]["scores"][group]["n"]
    print(f"\nV8 group: {group} | n={n} utterances")
    for name, score in zip(names, scores):
        if score is None:
            print(f"  {name:<26} unavailable")
        else:
            print(f"  {name:<26} W-F1={score:.2f}% "
                  f"delta-vs-original={score-scores[0]:+.2f}pp")
print("\nSelected checkpoints:")
for name, model in zip(names, r["models"]):
    print(f"  {name:<26} epoch={model['selected_epoch']}")
print("V8 single seed, previously viewed VALID: exploratory only.")
print("No evidence of independent generalization / novelty from this pilot.")
PY

echo "V8 pilot result saved: ${ROOT}/five_arm_validation.json"

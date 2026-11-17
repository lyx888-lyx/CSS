#!/usr/bin/env bash
# V7: two FIXED training-seed replications for the from-scratch SPF
# full-vs-residual ablation. Uses SAME V6.1 846 TRAIN/87 VALID cohort.
# NEVER tests or experiments on new val split; no official TEST.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${CSS_PYTHON:-python}"
EPOCHS=100
SPLIT_SEED=36427

if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "Missing original MELD feature pickle." >&2
  exit 1
fi

for SEED in 21431 31907; do
  ROOT="results/v7_spf_seed${SEED}_split${SPLIT_SEED}"
  FULL="${ROOT}/checkpoints/v7_full_valid_best.pt"
  RESIDUAL="${ROOT}/checkpoints/v7_residual_valid_best.pt"
  if [[ -e "${FULL}" || -e "${RESIDUAL}" ]]; then
    echo "Refusing to overwrite existing checkpoint in ${ROOT}." >&2
    echo "If the previous run was interrupted, inspect logs/checkpoints." >&2
    exit 1
  fi
  mkdir -p "${ROOT}/checkpoints"
  echo "=== V7 fixed seed replication ${SEED}: original full SPF ==="
  "${PY}" -u -m scripts.train_research \
    --dataset MELD --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
    --v61-clean-split --v61-spf-mode full \
    --v52-pgm-mode original \
    --speaker-mode off --precontext-mode off \
    --epochs "${EPOCHS}" --run-name v7_full \
    --checkpoint-dir "${ROOT}/checkpoints" \
    2>&1 | tee "${ROOT}/full_train.log"

  echo "=== V7 fixed seed replication ${SEED}: residual-only SPF ==="
  "${PY}" -u -m scripts.train_research \
    --dataset MELD --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
    --v61-clean-split --v61-spf-mode residual_only \
    --v52-pgm-mode original \
    --speaker-mode off --precontext-mode off \
    --epochs "${EPOCHS}" --run-name v7_residual \
    --checkpoint-dir "${ROOT}/checkpoints" \
    2>&1 | tee "${ROOT}/residual_train.log"

  "${PY}" -m scripts.compare_validation \
    --checkpoints "${FULL}" "${RESIDUAL}" \
    --output "${ROOT}/paired_validation.json"
  echo "Saved paired result for seed ${SEED} in ${ROOT}"
done

"${PY}" -u -m scripts.summarize_v7_seed_controls

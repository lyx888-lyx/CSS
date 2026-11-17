#!/usr/bin/env bash
# V6.1 STRICT SPF ABLATION — full CSS SPF vs residual-only SPF trained
# FROM SCRATCH, equal seed/optimizer/epochs, clean disjoint validation.
# No official TEST scoring. Never reuse the 115 or 104 old validation
# dialogues for new training / model selection.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${CSS_PYTHON:-python}"
TRAIN_SEED=10068
SPLIT_SEED=36427
EPOCHS=100
R="results/v61_spf_paired_split${SPLIT_SEED}_trainseed${TRAIN_SEED}"

if [[ ! -f data/meld_multimodal_features.pkl ]]; then
  echo "Missing MELD feature pickle" >&2
  exit 1
fi
if [[ -e "${R}/checkpoints/v61_full_valid_best.pt" ||
      -e "${R}/checkpoints/v61_residual_only_valid_best.pt" ]]; then
  echo "Experiment already has at least one checkpoint in ${R}." >&2
  echo "Refusing to overwrite or auto-resume to protect results." >&2
  exit 1
fi
mkdir -p "${R}/checkpoints"
printf '%s\n' "${R}" > results/v61_spf_last_run.txt

echo "V6.1 paired 100-epoch original CSS SPF vs residual-only SPF"
echo "846 TRAIN dialogues / 87 NEW VALID dialogues"
echo "115 old VALID +104 V5.2 VALID excluded entirely"
echo "V6 previously sampled 65 TRAIN dialogues excluded from NEW VALID"
echo "Same seeds, hyperparams, full initial weights, optimizer and training budget."
echo "Original PGM always used. No held-out official TEST scoring."

"${PY}" -u -m scripts.train_research \
  --dataset MELD --seed "${TRAIN_SEED}" --split-seed "${SPLIT_SEED}" \
  --v61-clean-split --v61-spf-mode full \
  --v52-pgm-mode original \
  --speaker-mode off --precontext-mode off \
  --epochs "${EPOCHS}" --run-name v61_full \
  --checkpoint-dir "${R}/checkpoints" \
  2>&1 | tee "${R}/full_train.log"

"${PY}" -u -m scripts.train_research \
  --dataset MELD --seed "${TRAIN_SEED}" --split-seed "${SPLIT_SEED}" \
  --v61-clean-split --v61-spf-mode residual_only \
  --v52-pgm-mode original \
  --speaker-mode off --precontext-mode off \
  --epochs "${EPOCHS}" --run-name v61_residual_only \
  --checkpoint-dir "${R}/checkpoints" \
  2>&1 | tee "${R}/residual_only_train.log"

"${PY}" -m scripts.compare_validation \
  --checkpoints \
    "${R}/checkpoints/v61_full_valid_best.pt" \
    "${R}/checkpoints/v61_residual_only_valid_best.pt" \
  --output "${R}/paired_validation.json"

"${PY}" - "${R}/paired_validation.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("split") != "validation_only" or len(result["models"]) != 2:
    raise RuntimeError("invalid paired validation result")
full, residual = result["models"]
print("\nV6.1 RETRAINED SPF ABLATION — 846 train / 87 new VALID, no TEST")
for group in ("all", "first", "stable", "changed",
              "stable_after_other", "changed_after_other"):
    a = full["scores"][group]["weighted_f1"]
    b = residual["scores"][group]["weighted_f1"]
    n = full["scores"][group]["n"]
    if a is None or b is None:
        print(f"{group}: n={n} unavailable")
    else:
        print(
            f"{group:<22} n={n:>3} "
            f"CSS SPF={a:6.2f}% residual-only={b:6.2f}% "
            f"delta(residual-full)={b-a:+.2f}pp"
        )
print("Selected epochs (full / residual):",
      full["selected_epoch"], residual["selected_epoch"])
print("One split/one seed; descriptive evidence, not publication proof.")
PY
echo "Saved all V6.1 outputs in ${R}"

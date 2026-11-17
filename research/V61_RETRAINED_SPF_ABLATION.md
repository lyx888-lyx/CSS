# V6.1: Does SPF high-order fusion add anything beyond its residual?
## Source-mechanism finding -> controlled retraining, NOT a new module

CSS (AAAI2026) addresses multimodal representation synergy with
a factorized high-order polynomial plus a linear residual:

    SPF(T,A,V) = signed_sqrt(prod_k z_k) @ W_out + b_out
                 + residual_proj(cat(T,A,V))

Original CSS still has sample-dependent gates *upstream* of SPF.
SPF's internal modality gates are globally learned per-order params.

### User-run V6: facts, 668 sampled TRAIN utterances

- Exact SPF decomposed reconstruction error **0**.
- Polynomial norm ≈ 3.0–3.6, residual norm ≈ 13.9–15.3,
  mean within-batch per-turn norm ratio ≈ **0.21–0.23**.
- Frozen CSS classifier argmax unchanged on **668/668**
  TRAIN utterances when removing polynomial branch.
- Frozen classifier, polynomial-only feature accuracy
  **16.32%**, far below original full **72.75%**.
- All four polynomial per-order text/audio/vision gates ≈
  **0.7307–0.7311**, close to `sigmoid(1)≈0.7311` initialization.
- Stable samples original full train accuracy 77.10%,
  Changed samples 66.18%.
- Fear 12 sampled turns; Disgust 19 sampled turns,
  insufficient for rare-class population claims.

**What this does not prove:** 
The polynomial may affect probabilities, margins,
gradients and learning trajectories even when argmax does
not change in one frozen checkpoint. Removing one part
without retraining produces distribution shift. Original
SPF *could* outperform a retrained residual-only model.
That is what V6.1 tests.

## Single preregistered H0/H1

- H0: retrained original full SPF has no reliable advantage
  over a from-scratch residual-only alternative on the same
  new held-out MELD dialogues.
- H1: high-order SPF offers incremental discrimination beyond
  the residual-only linear projection under the same conditions.

This is a **baseline ablation**, not an original research contribution.
A result favoring residual-only does NOT automatically justify adding
novel gates. It suggests checking why high-order features are
unused or whether the residual shortcut dominates.

## The ONE model difference

`model.py::SPFusion` gains non-parameter attribute
`ablation_mode`:

- `full` is the **unchanged original forward path**:
  polynomial + residual.
- `residual_only` returns `residual_proj(cat(T,A,V))`
  directly and never computes polynomial components. **It is
  trained from scratch**, not inference-only removal.

All module state_dict keys are identical; both initialize
all weights in the same order with same seed10068. Full
and residual-only both use original CSS PGM, loss definitions,
train loader shuffle, optimizer Adam/lr, batch size,
and **100 epoch budget**. Output architecture difference
is only the retained polynomial branch; residual-only model
has fewer active parameters and less compute. That difference
is the intended architectural ablation and should be reported
as such, not interpreted as matched-capacity comparison.

No temperature sweep, no hyperparameter selection after
viewing outcome, no TEST.

## Method-development data isolation

Prior data have been inspected across V0–V6:

1. Original MELD validation selected by seed **10068**:
   115 dialogues — exclude from both new TRAIN and VALID.
2. V5.2 clean validation selected by split seed **24117**:
   104 dialogues — exclude from both new TRAIN and VALID.
3. V6 SPF sampled TRAIN diagnostics from 933 remaining:
   64 dialogues (seed **73117**, size64).
   Plus V6 1-batch smoke sampled 2 dialogues with same
   seed but different size. Union = **65** unique dialogues.
   Keep them **only on new TRAIN** (they were already used
   for descriptive analysis); never select as new VALID.

Remaining 868 unseen-by-V6 eligible dialogues are split
with preselected seed **36427** into 781 new training and
87 new validation dialogues. Add the 65 V6 seen dialogues
back to TRAIN. Final:

- **846 TRAIN dialogues**
- **87 untouched new VALID dialogues**
- **115+104 old validation dialogues excluded from both**
- All sets pairwise disjoint, jointly cover original 1152
  MELD TRAIN dialogues.
- No use of official TEST labels or predictions.
- Split is generated BEFORE seeing V6.1 outcomes and
  locked for the paired run.

Caution: MELD's per-class rare support and
`changed_after_other` group may be tiny in 87 dialogues.
This pilot cannot support a publication-level generalization
claim without new independent replications/dataset.

## Pull and run (GPU user server)

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v61-spf-retrained-ablation

python -m unittest discover -s tests \
  -p 'test_v61_spf_retrained_ablation.py' -v

# One-epoch RESIDUAL-ONLY smoke (use separate output):
python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 36427 \
  --v61-clean-split --v61-spf-mode residual_only \
  --v52-pgm-mode original --speaker-mode off --precontext-mode off \
  --epochs 1 --run-name v61_residual_smoke \
  --checkpoint-dir results/v61_residual_smoke
```

If tests and smoke succeed:

```bash
CSS_PYTHON=python bash scripts/run_v61_spf_retrained_ablation.sh
```

Saves original full vs residual-only train logs,
checkpoints, and `paired_validation.json` in:
`results/v61_spf_paired_split36427_trainseed10068/`.

The shell script will REFUSE to overwrite checkpoints from a
preexisting run. After both train, it compares exact new 87
dialogue IDs with `scripts.compare_validation` and prints
weighted-F1 for all, stable, changed and after-other.

### What to interpret

- Full SPF **higher**: polynomial branch adds useful
  learned information in a fair retraining setting,
  even if V6 argmax deletion gave 0 flips. Then study
  when higher-order interaction matters, not "SPF useless".
- Residual-only **comparable/better**: under this setting,
  high-order SPF may have limited incremental usefulness.
  That is one useful bottleneck, not proof it is redundant
  across seeds or datasets. Matched early convergence and
  gradients/feature margin diagnostics may clarify mechanism.
- Changed and changed-after-other scores should be analyzed
  **with group N and paired error migration**, not as
  high-confidence group differences in this small VALID cohort.
- Do NOT compare these W-F1s directly against V5.2's 71.33
  or earlier original VALID CSS73.59, because neither uses
  the same 87-dialogue validation cohort.
- This ablation should NOT be branded as the final
  paper innovation. Dynamic polynomial gating is prior art;
  after diagnosis, additional novelty/references and
  multi-seed + IEMOCAP/MELD replication are needed.

No assistant-side GPU training or unit tests have been run.

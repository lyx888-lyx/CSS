# V5.2: test PGM optimization-direction mismatch (paired CONTROL experiment)

## Source-grounded evidence established before designing method

User reproduced original CSS AAAI2026 and ran observational V5/V5.1:
- PGM gradient cosine on fixed 8 sampled **TRAIN** batches:
  L1/L2 +0.2979 (0/8 negative),
  L1/L3 -0.0322 (4/8 negative),
  L2/L3 +0.0944 (0/8 negative).
- Mean PGM simplex alphas (L1/L2/L3):
  [0.3398, 0.2659, 0.3942].
- Pre-directional-cancellation alpha*||g|| magnitude shares:
  [0.3367, 0.6411, 0.0222].
- **V5.1 reconstruction from saved JSON**:
  mean angle between source PGM normalized-gradient target and
  raw-weighted loss gradient in selected PGM shared parameters
  **38.42 degrees**, 8/8 over 30 degrees
  (range 37.47–39.36 deg).
- L3 alignment with target +0.6362 but with actual CSS
  raw update only +0.0816.
- L2 alignment with target +0.6362 but with actual CSS
  raw update +0.9163.

**These results demonstrate a mathematical difference, not that
the original optimization is wrong or causes poor F1.**
A single selected checkpoint and 8 sampled batches are
not a whole-training-trajectory or multi-seed analysis.

## Research hypothesis

Original PGM computes alpha via normalized task gradients,
but the final weighted-loss backward uses unnormalized task
gradients. The direction discrepancy is especially strong when
the raw per-task norms differ greatly.

H1: Holding CSS architecture, its original PGM simplex weights,
training split, initialization, optimizer and shared-gradient
*norm* equal, using the normalized PGM target direction may
produce superior model learning and held-out generalization.

H0: The difference is mathematically measurable but is not a
performance bottleneck, or it may harm useful L2 supervision.

### The one experimental change

Original CSS:

```
g_i = gradient(L_i, original_PGM_shared_parameters)
alpha = original_PGM._project(g_i) # original SLSQP on unit gradients
L_original = sum_i alpha_i * L_i
d_original = sum_i alpha_i * g_i
```

Experimental V5.2:

```
u_i = g_i / (||g_i|| + 1e-8)
d_unit = sum_i alpha_i * u_i
scale = ||d_original|| / ||d_unit||
c_i = detach(scale * alpha_i / (||g_i|| + 1e-8))
L_aligned = sum_i c_i * L_i
d_aligned = scale * d_unit
```

**Exactly in the original PGM-selected shared parameter subset
and at each same parameter state/minibatch, this makes
||d_aligned|| = ||d_original||, with direction aligned to
PGM's original normalized gradient projection.**

This distinguishes *direction* from overall shared-gradient
norm. Critically:
- It does NOT normalize or norm-match gradients in parameters
  excluded from the PGM selection (SPF, speaker embeddings,
  classifier heads). The actual optimizer is Adam, so
  equality of raw shared-gradient norms does NOT guarantee
  equality of Adam parameter-step norms or trajectories.
- Original CSS alpha still comes from the original source PGM
  `_project`, unchanged. The original three task losses
  `_compute_task_losses` and their target propagation remain
  unchanged, including **non-detached fused KL target**.
- This is a **loss combination variant** of established
  gradient normalization, NOT a novel contribution by itself.
- The raw scalar training loss changes its scale, so **do not
  compare train_loss magnitudes** between baseline and variant.
  Compare held-out W-F1, learning stability, gradient geometry
  and clean ablations.

Source:
- `research/pgm_direction_consistent.py`: opt-in subclass.
- `train_MELD.py`: default original PGM unchanged;
  option `--v52-pgm-mode direction_aligned` chooses subclass.
- `scripts/train_research.py`: passes opt-in flag, saves config.
- `tests/test_v52_pgm_direction_consistent.py`: CPU tests
  verifying exact shared-gradient direction & magnitude.
- `tests/test_v52_disjoint_split.py`: no legacy validation overlap.
- `scripts/run_v52_pgm_paired_pilot.sh`: matched paired run.

## Disjoint development evaluation: DO NOT reuse old 115 VALID

Past V0/V1/V1.1/V2/V3/V4 repeatedly inspected 115 MELD
VALID dialogues defined with split seed 10068.

A naively different random split would overlap those dialogues.
V5.2 therefore first permanently excludes the **original 115
VALID dialogues** (seed10068) from BOTH V5.2 training and
V5.2 model selection, then uses a second fixed seed 24117
to split the remaining original 1037 TRAIN dialogues:
- 933 train dialogues
- 104 fresh validation dialogues
- 115 OLD VALID dialogues **not touched** by either arm

Both experimental arms use EXACT same train/validation dialogue
indices, train RNG seed10068, Adam/lr, dropout, batch size,
schedule, 60 maximum epochs, identical CSS architecture.
The source classifier head and all other modules are unchanged.
No official TEST scoring.

The new validation is independent of the old validation
*dialogue IDs*, but this single 104-dialogue group is still
exploratory because researchers know the hypothesis and are
allowed one prespecified check. The 60-epoch best-validation
selection is identical in both arms. Multiple seeds/other
splits and a second dataset would be required for publication.

## Run instructions

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v52-pgm-direction-consistency-pilot

# New tests, also ensure existing tests remain green:
python -m unittest discover -s tests -p 'test_v52*.py' -v

# Fail fast if gradients cannot pass through the new loss.
# This smoke's validation F1 is NOT evidence.
python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 24117 \
  --v52-clean-split --v52-pgm-mode direction_aligned \
  --speaker-mode off --precontext-mode off \
  --epochs 1 --run-name v52_aligned_smoke \
  --checkpoint-dir results/v52_aligned_smoke
```

If the 1-epoch smoke passes:

```bash
CSS_PYTHON=python bash scripts/run_v52_pgm_paired_pilot.sh
```

Or detached with tmux after tests:

```bash
tmux new-session -d -s css_v52_pgm \
  "cd /code/CSS && CSS_PYTHON=python bash scripts/run_v52_pgm_paired_pilot.sh"
```

The script REFUSES to overwrite existing paired run checkpoints.
It saves logs/checkpoints under
`results/v52_pgm_paired_split24117_trainseed10068/`
and a pointer `results/v52_pgm_last_run.txt`.

After both arms finish, `scripts.compare_validation` validates
matched dialogue indices, model train seed/classes, and reports
all/stable/changed/after-other Weighted-F1 and selected epoch.
The old checkpoint's 73.59% on DIFFERENT validation dialogues
must NOT be used as the control in this V5.2 experiment.

## GO / NO-GO

1. Smoke must complete without NaN, invalid task-gradient norm
   or solver failure. Large effective L3 loss coefficients
   are expected mathematically; they must be reported.
2. If matched paired results show no whole-WF1 improvement
   or significant stable/emotion-change harm: **stop** this
   direct correction; do not tune scale, KL temperature, loss
   coefficients or random seeds on the new validation group.
3. If pilot improves: it is just **encouraging exploratory
   evidence**, not proof of generalization or novelty.
   Preregister new independent splits and initialization seeds,
   compare against established gradient normalization,
   equal-loss gradient weighting and original CSS with equal
   budgets, then validate on IEMOCAP and MELD before finally
   touching TEST once.
4. PGM/SLSQP may already use known multi-task optimization
   strategies; loss-gradient normalization itself is prior art.
   Novelty would require a principled newly demonstrated
   method or explanatory theory beyond this control check.

No assistant-side GPU training was run. All model performance
claims must originate from user-run logs.


## User-run V5.2 result — fixed protocol, **NO-GO** (2026-11-17)

The user ran the full paired 60-epoch MELD pilot on the preregistered
new disjoint validation cohort (933 new TRAIN / 104 NEW VALID;
115 repeatedly inspected OLD VALID excluded from both arms).
Both arms used train seed10068, split seed24117, and did not touch TEST.

| Group | Original CSS W-F1 % | Direction-aligned PGM W-F1 % | Difference pp |
|---|---:|---:|---:|
| all | **71.33** | 68.39 | **-2.94** |
| first | 75.37 | 74.49 | -0.88 |
| stable | 80.12 | 79.41 | -0.71 |
| changed | **62.15** | 56.52 | **-5.63** |
| stable_after_other | 80.09 | 78.68 | -1.41 |
| changed_after_other | **64.58** | 59.07 | **-5.51** |

Best epoch (within fixed 60-epoch budget): original 58,
experimental direction-aligned 54.
User run output location:
`results/v52_pgm_paired_split24117_trainseed10068/`.

**Decision:** STOP V5.2 direct normalized-direction loss correction.
Do not sweep loss scales, gradient-normalization coefficients,
temperature, training seeds or splits *based on this new validation
cohort* to look for a positive result. Keep the negative finding.

**Precisely what is and is not established:**
- V5.1 established a reproducible ~38.42° mismatch between the
  normalized PGM target direction and original raw-weighted gradient,
  in 8 sampled TRAIN batches for one fixed checkpoint.
- This V5.2 paired pilot offers **no support** for the hypothesis
  that forcing direction agreement improves MERC on this MELD setup;
  the opposite was observed (-2.94 pp overall).
- The 5.63 pp decline for changed turns is descriptive. It does
  **not** prove particular cross-speaker interactions were harmed;
  `changed` labels are determined only for retrospective reporting.
- This experiment changes loss coefficients over **all** parameters,
  and matches gradient *norms only in PGM-selected shared parameters*.
  It does NOT hold SPF or classification-head gradient norms fixed,
  and Adam's adaptive steps also differ. Thus results **cannot
  causally isolate** a direction-only effect across the full network.
- 1 training seed, 1 fresh validation split, no confidence intervals
  or cross-dataset replication: the negative result is local, not a
  universal theorem about normalized gradients.

**Next research choice:** return to CSS's original high-order
multimodal representation rationale (SPF) and speaker/context
error diagnostics. A new lightweight **read-only TRAIN-only**
SPF contribution audit (polynomial component vs residual and
per-order gate behavior) is a better next gate than adding
another loss without causal evidence. Any new method must be
connected to a concrete original-CSS limitation.

# V7 Research selection: from CSS diagnosis to reproducible MERC gains
Date: 2026-11-17. All methods explicitly 2026; provenance is official paper venue and repositories, **not** unverified leaderboard numbers.

## Goal and status

Goal: find a defensible, **repeatable improvement** in MERC overall
Weighted-F1 under matched training/validation/feature protocols,
while tracking emotion changes and cross-speaker-context subgroups.
Working aspiration: **>= +1.0 percentage point mean overall W-F1**
versus a SAME-SPLIT, SAME-BUDGET original CSS control, without
material decline in Changed and Changed-after-other subgroup.
This is a team-defined threshold, NOT proof of significance
and not a promise of improvement.

**Original CSS AAAI2026 has already been reproduced** on
official IEMOCAP/MELD testing in earlier work, separately from
strict new development validation. The official TEST stays
untouched in ALL future research development, to be evaluated
once after the algorithm and operating point are frozen.

Known 2026 CSS source facts:
- SPF = factorized high-order polynomial + learned linear
  residual. Its modality_gates are global learned per order.
  Other CSS layers do have sample-dependent gates.
- PGM original `_project` computes weights on normalized
  gradients in a selected shared parameter subspace, then
  weights raw losses in backprop. V5.1 found a ~38.42 degree
  direction difference in 8 TRAIN minibatches. Normalizing
  the actual update (V5.2) **decreased** overall held-out
  W-F1 by 2.94 points and Changed by 5.63 points in
  one same-split 60-epoch pair. STOP this correction.
- V6 TRAIN SPF reconstruction exact; polynomial/residual
  feature norm ratio ~0.22; frozen-head component deletion
  0/668 top1 changes at V5.2 baseline checkpoint.
- V6.1 true fresh retraining (100 epochs, 846 TRAIN /
  87 clean VALID, original CSS/PGM) full SPF vs residual:
  76.14 vs 76.43 W-F1, +0.29pp residual-only.
  Changed: 66.79 vs 67.85, +1.06 pp residual.
- V6.2 1500 whole-dialogue paired bootstrap overall
  CI of residual - full W-F1 [-0.34,+0.97]pp,
  containing zero. 4 repaired / 2 newly broken correct
  predictions; these are 6 **correctness changes**, NOT
  necessarily six differing raw class predictions.
  Changed subset nominal CI [+0.04,+2.39]pp is based
  on just 3 corrected labels and overlapping multiple
  exploratory subgroups; NOT a robust advantage.
- V6.3 (different full-SPF checkpoint from V6) sampled
  same 64 TRAIN dialogues/668 utterances: 8 top1 changes;
  mean class-centered polynomial/residual **logit norm**
  ratio 0.0549, mean absolute true-class margin delta
  0.106, mean total-variation of softmax 0.01185.
  Margin improves 50.1% / harms 49.9% of sampled turns.
  Thus SPF is **not zero** and influences margins, but
  lacks clear incremental top1 advantage in THIS setup.

The hypothesis is specifically:
> The complex SPF high-order representation can contribute
> small or unhelpful class-discriminative evidence relative
> to the residual shortcut; investigate whether the issue is
> information redundancy, optimization/shortcut bias, or
> contextual reliability, *before adding a new architecture*.

## Current 2026 literature map — verified URLs

| Paper | 2026 venue | Official publication | Public code | Relevant idea | Reproduction/blockers |
|---|---|---|---|---|---|
| CSS — Cross-Space Synergy | AAAI 2026 (top conference) | https://ojs.aaai.org/index.php/AAAI/article/view/39602 | this CSS repo | SPF high-order + PGM; primary reproducible baseline | already locally reproducible; beware original test-based checkpoint selection |
| EMART — Emotion-Wheel-Guided Audio-Referred Text Representation | ACL 2026 Long (top conference) | https://aclanthology.org/2026.acl-long.1875/ | https://github.com/DILAB-HYU/EMART | text-centered audio complement; emotion wheel SupCon and speaker-aware previous-utterance context | user had previously run code; text/audio + different features; **not** a matched CSS head-to-head until protocols align |
| DnR — Divide and Refine | WACV 2026 | https://openaccess.thecvf.com/content/WACV2026/html/Mai_Divide_and_Refine_Enhancing_Multimodal_Representation_and_Explainability_for_Emotion_WACV_2026_paper.html | https://github.com/mattam301/DnR-WACV2026 | explicit modality unique/redundant/synergy decomposition + contrastive refinement | repo executable paths presently cover late_fusion/MMGCN/DialogueGCN/MM-DFN, **not CSS**; original IEMOCAP/MELD pickles are required; WACV is a reference, do not silently label it AAAI/ACL tier |
| DF-GCN — Dynamic Fusion-Aware Graph Convolutional Network | Knowledge-Based Systems 2026 | https://www.sciencedirect.com/science/article/pii/S0950705126005721 | https://github.com/yuntaoshou/DFGCN | speaker-interaction graph dynamics + utterance-conditioned fusion | older torch/fairseq/CUDA deps; README provides some pre-extracted features but **do not presume matched CSS features**; dynamic gate alone clearly prior art |
| MiSTER-E — Mixture of Experts for MERC | Computer Speech & Language 2026 | https://www.sciencedirect.com/science/article/pii/S0885230826000288 | https://github.com/iiscleap/MiSTER-E | dynamic speech/text/cross-modal expert selection; contrastive/distillation; **does not use speaker identity** | different modality and pretrained-model cost, 69.5 MELD weighted-F1 reported on author's own protocol (not directly CSS-comparable) |
| CDCP — Conflict-Aware Dual-Path Context Propagation | repository says ACM MM 2026; **paper identity unverified** | no independently verified ACM DOI/paper in current scan | https://github.com/Tianba0116/CDCP | dual-path speaker context and conflict-aware expert reconstruction; closely related to advisor's interspeaker goal | code files/checkpoint and preprocessing instructions exist; **README selects best checkpoint on TEST F1**, leakage in reproduction/selection. Another repo https://github.com/MKMaS-GUET/CDCP claims to be official but says code coming soon. Verify author, DOI, code provenance before relying on it as 2026 top-venue result |
| SFAC — Semantic-Guided Fine Alignment and Asymmetric Contrast | CIKM 2026 program | https://cikm2026.diag.uniroma1.it/cikm-program/program.html | https://github.com/cgao-comp/SFAC (link appears in program; live repo accessibility unverified) | token/frame alignment and asymmetric contrast, preserving complementary audio/vision evidence | requires token/frame-level features, **not guaranteed available in CSS pickles** |
| MER-Cross | ACM MM 2026 workshop challenge, **not same target as CSS** | https://huggingface.co/datasets/MERChallenge/MER2026 | https://github.com/zeroQiaoba/MERTools | listener's emotion during OTHER speaker's turn | task predicts listener emotion, not target-utterance speaker emotion; **do not compare its reported F1 directly to CSS MERC** |
| Satisfactory? | Pattern Recognition 2026 | https://www.sciencedirect.com/science/article/pii/S0031320326000506 | unknown in this scan | evaluation protocol, out-of-domain robustness, calibration | reinforces caution on test leakage and heterogeneous feature comparisons |

**Avoid false novelty:** dynamic modality gates, class-specific
modality weights, expert selection, multimodal disentanglement,
speaker-aware context and valence/arousal priors have all been
investigated already. An independent contribution must address
a specific limitation they do not solve, provide causal/controlled
mechanistic evidence and surpass appropriate strong baselines.

**Avoid misreading another task:** MER-Cross recognizes the
listener's emotion whereas CSS recognizes the emotionally
labeled target utterance; different annotations and metrics.

## Ranked action plan for actual metric improvement

### Gate 1 — training-seed robustness of current SPF finding

Before inventing the seventh architecture, run exactly two more
independent initialization/shuffle seeds on the **same fixed**
V6.1 846 TRAIN / 87 VALID split, with from-scratch
original SPF and residual-only CSS, same optimizer, original
PGM and same 100-epoch budget.
We already have seed10068; preregister additional seeds
**21431, 31907** before viewing any further outcomes.
Existing result and these two together yield three matched
seed pairs; this tests initialization robustness **on the
same already inspected validation cohort only**.
Do NOT call these independent validation confirmations.

Prespecified readout: mean and individual paired delta
(residual minus full) overall W-F1, changed and
changed-after-other. Count signs and spread, do not
tune learning rate/loss/gate parameters to rescue a result.

If residual-only gains persist across seeds, the next
hypothesis should focus on whether residual shortcut dominates
high-order discriminative directions, NOT plain addition of
a dynamic gate. If not persistent, SPF redundancy is
seed-specific and we should change research direction.

### Gate 2 — choose novel *mechanism*, not recycled module

Candidate research questions (not implemented yet):
1. **Complementarity-selective high-order residual correction**:
   can multimodal high-order interaction provide *incremental*
   evidence when unimodal cues conflict or when conversational
   context indicates uncertainty, but remain dormant otherwise?
   Must distinguish from DnR decomposition and DF-GCN/MiSTER-E
   expert gating; no ground-truth labels or future true
   emotion transitions are permitted for routing at inference.
2. **Speaker-mediated context-conditioned evidence**:
   can the *preceding other-speaker utterance* modify
   cross-modal complementarity in the target utterance,
   beyond CSS's existing bidirectional speaker-embedding
   context and beyond prior temporal/graph GCNs?
   Need pre-defined causal inputs vs full-dialogue inference,
   careful comparison with CDCP/SFAC and no leakage.
3. **Classifying when interaction is harmful**:
   compare polynomial class-centered logit margin, local
   modality evidence, and model disagreement on TRAIN only.
   Baseline distinctions should not use true labels as
   an inference-time feature.

Choose **one** after clarifying novelty, feature needs,
and a falsifiable expected advantage.

### Gate 3 — matching protocols, frozen method, held-out TEST

- Use CSS original MELD features and exact split for all
  local CSS ablations. No comparing results from different
  text/audio/visual feature extractors as though architecture-only.
- Run matched original CSS and chosen method under >=3
  initialization seeds with identical budget; report mean,
  SD and per-seed differences in overall W-F1 and shift
  metrics; actual goal ~+1 pp overall, not guaranteed.
- Because V6.1's 87-dialogue VALID was already inspected,
  results on it are DEVELOPMENT only, not independent proof.
  For a new independent estimate, preregister a distinct
  clean partition before seeing results or use a published
  independent dataset (IEMOCAP), with non-overlap and clear
  training/selection policies. Do not exhaust new data
  through post-hoc search.
- After algorithm and hyperparameters are frozen,
  retrain with the allowed full training data under the
  official protocol and evaluate the official TEST **once**.
  Compare with CSS official numbers only when dataset,
  class mapping, features and protocol match.
- Do not use repeated official TEST scores for checkpoint
  selection; explicitly avoid CDCP's README-described
  test-F1-best checkpoint rule.

## Rules of interpretation

- A point estimate improvement without stable seed-wise
  gains is exploratory.
- A bootstrap CI from the VALID cohort used in
  checkpoint selection does not replace cross-seed and
  independent holdout evidence.
- Never create a novelty claim by rebranding
  well-known gating, attention, contrastive loss or
  gradient normalization.
- Negative/neutral outcomes are preserved in the
  versioned research log.
- No assistant-side code execution with the server
  GPU; scripts and claims require user run.


## 2026-11-17 — User-run V7 fixed three-seed result (LOCKED)

Train/validation: MELD 846 TRAIN / 87 already-inspected VALID,
100 original epochs, identical CSS features/PGM and matched
full-SPF-vs-residual arms. TRAIN seeds **10068, 21431, 31907**.
The first seed is from V6.1; the other two were locked before
being run. All results are **residual-only minus full SPF W-F1**:

| Group | seed10068 | seed21431 | seed31907 | Mean delta pp | SD (seed) | Positive |
|---|---:|---:|---:|---:|---:|---:|
| all | +0.29 | +0.11 | +0.13 | **+0.18** | 0.10 | 3/3 |
| first | -0.41 | +0.41 | -0.84 | -0.28 | 0.64 | 1/3 |
| stable | +0.00 | -0.45 | +0.36 | -0.03 | 0.41 | 1/3 |
| changed | +1.06 | +0.39 | +0.66 | **+0.70** | 0.34 | 3/3 |
| stable_after_other | +0.53 | +0.25 | +0.53 | +0.44 | 0.16 | 3/3 |
| changed_after_other | +0.98 | +0.82 | +1.17 | **+0.99** | 0.18 | 3/3 |

Source path: `results/v7_three_seed_spf_summary.json`
and matched per-seed `paired_validation.json` reports.

**Interpretation:**
- Full-SPF's incremental performance benefit beyond a retrained
  linear residual is NOT apparent in these three seed pairs.
- The *sign* of the slight residual-only overall gain is
  consistent, but the mean +0.18pp is small and cannot establish
  a universal no-benefit claim, especially on one previously
  viewed validation subset and with no independent split.
- The changed-after-other +0.99pp difference is a prioritization
  clue, NOT evidence that a mechanism really measures another
  speaker's emotional impact; labels are post-prediction only.
- No new method was tested, and the V7 run does NOT fulfill our
  self-chosen target of +1.0pp mean **over the original CSS**
  on matched protocol.

**Gate decision:** enough evidence to justify ONE prespecified
METHOD pilot with proper controls, not enough to claim that
"high-order feature is useless" or to choose a method by
tuning the same 87 VALID dialogues repeatedly.
The proposed next pilot V8 is a *fixed hypothesis test*
against original full-SPF, residual-only, fixed gain x4,
and matched conditional gates, with train seed10068 and
same fixed development cohort. Its outcomes are exploratory;
a genuine improvement must be confirmed with independent
datasets/splits and multiple seeds.

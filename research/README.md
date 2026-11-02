# CSS speaker-dynamics research: stage 1

## Hypothesis

Emotion classification may benefit from **separating the target speaker's
own past from other speakers' preceding utterances**, and distinguishing
recent from distant context. Do **not** claim that this decomposition is
new by itself: recent ERC work already uses intra/inter speaker contexts.

Research direction to investigate later:
- speaker-conditioned, *directed* and temporally decayed interaction strength;
- changes of a speaker's state between their own successive turns;
- comparing adaptive interaction against simple same/other attention;
- effects on minority classes and emotional transitions.

These are **hypotheses**, not demonstrated improvements.

## Preliminary data diagnostics (train split only)

Run in the `css` environment with both datasets linked under `data/`:

```bash
python -m unittest discover -s tests -v
python scripts/analyze_speaker_dynamics.py --dataset MELD --split train --out results/meld_train_speaker_stats.json
python scripts/analyze_speaker_dynamics.py --dataset IEMOCAP --split train --out results/iemocap_train_speaker_stats.json
```

This script reports the availability of previous same-speaker and
other-speaker turns, dialogue sizes, and within-speaker emotion changes
with/without intervening other speakers. These are **descriptive associations**,
not causal influence estimates; differences may result from many confounders
(e.g., dialogue length, emotion distribution, or turn-taking patterns).

Only use training labels for exploratory work. Do not inspect test labels to
choose architectures, tune hyperparameters, or select the best epoch.

## Mask module

`research/speaker_history.py` provides past-only same-speaker and
other-speaker [B, L, L] masks usable by future ablation modules.
They intentionally leave the original CSS model unchanged.

**Important:** CSS's original modality-aware and interaction-aware attention
branches are bidirectional. Adding a past-only mask to a downstream block
does not make the *entire model* causal. A strictly past-only comparison
would also need corresponding causal masking in the upstream encoders and
separate baseline retraining.

## Experimental protocol before claiming an improvement

1. Preserve upstream `main` and `repro/preflight-2026` baselines.
2. Define and freeze an independent validation split made from training dialogues.
3. Retrain original CSS using the same split/hardware/seeds as any modification.
4. Ablate: CSS / same-speaker only / other-speaker only / both / temporal decay.
5. Report weighted F1, macro F1, per-class F1, transition-focused results,
   mean and std over at least three seeds, and any inference-time overhead.
6. Never claim causal effect without extra supporting experimental design.

## Stage-1b: gap-stratified follow-up

The initial same-speaker change-rate comparison (with vs without
intervening other speakers) **cannot** separate speaker interaction
from utterance distance: without an intervening other speaker the two
consecutive turns are necessarily adjacent (gap=1), whereas with an
intervening speaker the return gap is necessarily >=2.

A second, still descriptive diagnostic reports differences *within*
interleaved turns, grouped by distance and by whether the immediately
preceding other speaker's annotated emotion equals the target
speaker's previous annotated emotion:

```bash
python -m unittest discover -s tests -v
python scripts/analyze_speaker_interactions.py --dataset MELD --out results/meld_train_return_gaps.json
python scripts/analyze_speaker_interactions.py --dataset IEMOCAP --out results/iemocap_train_return_gaps.json
```

These diagnostics use TRAIN annotated emotions only and must NEVER
be turned into inference-time features. Last-other agreement and
class-dependent transition frequencies are not proof of emotional
contagion. Stratification alone does not remove other confounders.

Do not promote new modules or paper novelty claims based solely on
these descriptive rates.

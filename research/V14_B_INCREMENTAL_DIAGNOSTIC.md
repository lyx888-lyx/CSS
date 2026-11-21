# V14 — TRAIN-only incremental information probe for the B speaker

Date: 2026-11-21. Project **ONLY /code/CSS**.
This is **NOT** another 200-epoch CSS modification. It is a locked,
low-cost, falsifiable question BEFORE choosing a new architecture:

> Given the current A utterance and A's last utterance, does the actual
> intervening B message contain *incremental predictive information*
> for A's current emotion, compared with unrelated B messages?

V12 reproduced ORIGINAL CSS source-faithful author benchmark
IEMOCAP official TEST W-F1 75.28%, near paper 75.66%.
V12 self/reaction acoustic modifications: 74.38 /74.06.
V13 training-only self/cross acoustic prediction: 74.75 /74.83.
All V12/13 author-style checkpoints were chosen by TEST every epoch,
the original paper source behavior; this is not untouched TEST
generalization, and IEMOCAP Session5 has already been exposed.

The V14 diagnosis **never evaluates TEST** and **never modifies the
CSS backbone, PGM, original baseline, checkpoint, or train partition**.

## Exact data definition

Read `IEMOCAPDataset(train=True)` only, and iterate ONLY
`ds.keys`, the official TRAIN keys Ses01–04, 120 dialogues,
5810 utterances. Actual observed session dialogue counts are
**Ses01=28, Ses02=30, Ses03=32, Ses04=30** (120 total);
the original pickle does NOT guarantee 30 dialogues in each session.
Preflight guards total=120, coverage=Ses01–04 and total
TRAIN utterances=5810 without imposing an equal-session split. The original feature pickle itself contains
TEST arrays as well, but the V14 code never accesses TEST keys,
labels, sample predictions, or model scoring, and it never
instantiates the `train=False` dataset.

Every eligible example must have:
- A's current utterance at index i;
- A's last previous utterance j, strictly j<i;
- >=1 intervening DIFFERENT-speaker B utterance k with j<k<i.

An adjacent same-speaker A-A without a B is not eligible.
First A turns without previous A are not eligible.
Dialogues are never mixed when constructing a genuine triple.

Inputs are text SDT 1024, acoustic 1582 and visual 342 feature
arrays from the original CSS pickle. A fixed random Rademacher
projection (single label-free seed 20261121) maps each modality
to dimensions 96/96/32 => 224 dimensions per utterance; the
identical maps are applied to A and B, independently of emotion
labels and fold. This is **lossy** and diagnostic only, not a
replacement for end-to-end CSS training.

Target label = A's current true emotion, used for supervised
probe training and fold scoring. A's previous true emotion
is consulted **ONLY after prediction** to label stable/changed
analysis groups. Gold emotions are not included in any input
feature, including the shuffled B donor selection.

## Four exact matched arms

Every arm uses exactly the same eligible A-B-A samples and
the same current-A classification targets.

A features: [A-current, A-previous, A-current minus A-previous].

B features: [B-intervening-mean, A-current minus B-mean,
A-previous minus B-mean]. This includes RELATIONAL
features but their semantic benefits are not guaranteed.
Every arm has the same number of linear model input columns:
1. **A-only** = A features, zero-filled B block.
2. **Real B** = A plus actual B mean/relations.
3. **Shuffled B** = A plus unrelated B from a DIFFERENT dialogue,
   SAME session and SAME B speaker role (M/F). Prefer the same
   count-bin of intervening B turns. Independent training
   and validation donors; 3 fixed negative-control seeds 31/47/73.
   Train a separate matched-capacity linear classifier each draw.
4. **Real B model, B replaced** = the SAME real-B-trained model,
   but swap out real B for unrelated donor B at inference.
   This tests whether the trained probe depends on the actual
   conversational association, without retraining.

This is evidence of predictive **association**, not evidence
B caused A's emotion change. Across IEMOCAP speakers, same-sex
M/F matching is coarse; donor matches are intentionally not
matched using true emotion labels.

The classifier is fixed:
`LogisticRegression(C=0.1, class_weight="balanced",
solver="lbfgs", max_iter=350, tol=0.001)`.
Two feature `StandardScaler` instances fit on each FOLD TRAIN
only (one A block, one B block); all arms reuse the same fitted
scalers. No model selection, parameter sweep, or TEST usage.

## Split and interpretation

Four leave-one-**official-TRAIN-session**-out folds:
- TRAIN Ses02–04, evaluate Ses01;
- TRAIN Ses01+03+04, evaluate Ses02;
- TRAIN Ses01+02+04, evaluate Ses03;
- TRAIN Ses01–03, evaluate Ses04.

Entire dialogues and speakers/sessions are grouped.
The held-out session becomes *development-only* evidence;
it is never official Session5 TEST. Each fold is trained
on a portion of the 5810 original TRAIN utterances,
but the FINAL CSS benchmark protocol (if later justified)
MUST use **all** Sessions1–4 TRAIN again. The diagnostic
cannot be compared numerically to paper CSS W-F1 75.66.

Evaluate accuracy, W-F1, macro F1 on each of four folds and
pool all out-of-fold A-B-A predictions. Print diagnostic
stable/changed subgroup metrics. Each repeat's held-out
shuffled samples remain in the SAME fold and never use a
training-fold donor.

## Precommitted next-step decision

A practical **positive screening signal** (not a statistical
significance claim) should meet ALL:
1. Pooled Real-B W-F1 >= A-only +0.50 pp.
2. Pooled Real-B W-F1 >= mean of 3 shuffled-trained B
   controls +0.50 pp.
3. Real B beats BOTH A-only and mean shuffled B in
   at least 3 of 4 independently held-out original TRAIN sessions.
4. Replacing Real B with unrelated B should measurably hurt
   the Real-B-trained probe, ideally pooled >=0.50 pp.

The 0.50 pp thresholds are conservative exploratory gates,
NOT a promise of better CSS official TEST results or a formal
hypothesis test. Track changed-turn performance as supporting
analysis, not as a tunable primary metric.

If the screen **fails**, STOP this particular low-cost
representation/classifier design and write the negative result.
Do NOT conclude B has no information in general: this linear,
lossy random-projection probe may be too weak to extract it.
Do not tune projection dimensions, shuffling seeds, C,
classifier architecture or target samples to chase held-session
scores. A next hypothesis requires new evidence or a distinct
protocol, and TEST must not be used to choose it.

If the screen **passes**, do NOT immediately proclaim a novel
method. Next assess research novelty vs existing literature
on emotion-cause modeling, cross-speaker attention, history
selection and conservative correction; then implement a
carefully controlled CSS-compatible method. Prior V11/V12/V13
TEST inspection prevents presenting future IEMOCAP Session5
as an untouched independent benchmark.

## Run in CSS only

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v14-b-incremental-diagnostic

python -m unittest discover -s tests \
  -p 'test_v14_b_incremental_probe.py' -v

# Only official TRAIN keys, no fitting; counts true A-B-A examples
python -u -m scripts.run_v14_b_incremental_probe --preflight

# Small synthetic-sized 1-fold sanity experiment;
# separate report, no TEST, no CSS training
python -u -m scripts.run_v14_b_incremental_probe --smoke

# Full TRAIN-only 4-fold diagnostic / 3 shuffled controls
python -u -m scripts.run_v14_b_incremental_probe
```

Outputs are:
- `results/v14_b_incremental_diagnostic/smoke.json`
- `results/v14_b_incremental_diagnostic/report.json`

The full report contains folds, donor-validity checks, overall
and stable/changed subgroup scores, and aligned per-example
predictions/labels. Results are descriptive and limited to
the original TRAIN cohort.

No new CSS network, no PGM changes, no 200-epoch runs, and
**NO official TEST scores** are generated by this branch.


## 2026-11-21 — Actual user server run: four-fold TRAIN-only V14 NO-GO

Preflight verified original TRAIN-only Ses01–04, 120 dialogues
with distribution 28/30/32/30, 5810 turns. Exactly 3751
eligible A(previous)–B(intervening)–A(current) turns;
1166 changed, 2585 stable. By session, eligible/changed:
Ses01 926/278; Ses02 878/264; Ses03 952/314;
Ses04 995/310. The official TEST loader was not created
or scored. The initial separate 1-fold SMOKE used only
192 A-B-A TRAIN / 64 A-B-A VALID examples and is a
functional test, NOT the final decision.

Complete four-fold leave-one-original-TRAIN-session-out
linear probe, all arms aligned on the same 3751 targets:

| Pooled arm | out-of-fold weighted F1 |
|---|---:|
| A-only | **59.02%** |
| Real B | 57.71% |
| Mean of three shuffled B-trained controls | 58.18% |
| Real-B-trained model, B replaced by shuffled donors | 55.14% |

- Real B minus A-only = **-1.31 pp**.
- Real B minus shuffled-B control mean = **-0.47 pp**.
- Real B minus the very same model after B substitution = +2.57 pp.
  This indicates predictions rely on B inputs and/or are sensitive
  to the input distribution shift. **It does NOT demonstrate
  incremental accuracy benefit**, because Real B is worse
  than both A-only and shuffled-trained controls.

| Held-out TRAIN session | A-only W-F1 | Real B W-F1 | Real-A pp | Real-mean-shuffled pp |
|---|---:|---:|---:|---:|
| Ses01 | 59.25 | 56.71 | -2.54 | -1.75 |
| Ses02 | 62.39 | 60.01 | -2.38 | +0.26 |
| Ses03 | 58.24 | 58.31 | +0.07 | -0.12 |
| Ses04 | 56.68 | 56.24 | -0.44 | -0.29 |

Real B beat BOTH A-only and mean shuffled control in **0/4**
held-out original TRAIN sessions.

Changed-turn only posthoc W-F1 (n=1166):
A-only 48.33%; Real B 47.34% (delta -0.99 pp).
Three shuffled-trained B arms: 49.69 / 48.21 / 47.60%.
These selected-subset metrics are descriptive; no guarantee
of emotion-change benefit in current probe.

**Prespecified SCREEN FAILS. Decision: NO-GO for THIS fixed
lossy-projection, linear-classifier diagnostic.** No V15
model training authorized on the assumption this evidence
confirms B's incremental value.

### What this does and does not say

1. Under the fixed probe, actual B information does **not**
   reliably improve A current emotion classification beyond
   current/previous A features. Nor does the real-B arm
   outperform the cross-dialogue shuffled B control.
2. Existing B sensitivity alone is insufficient evidence
   for **useful** predictive contribution, much less causality.
3. The code's linear feature block is
   `A=[a_i,a_j,a_i-a_j]` and
   `B=[b,a_i-b,a_j-b]`. Since logistic regression is LINEAR,
   these differences can be algebraically absorbed into
   linear coefficients on `a_i,a_j,b`. There are **no
   multiplicative or context-dependent interactions**.
   Thus this probe cannot learn an "if B says X AND A
   previously feels Y" effect beyond linear separability.
4. Fixed random 96/96/32 projection is lossy; the classifier
   is not CSS; only A-B-A turns are evaluated; not the
   original 1623 TEST utterances or CSS's full 5810-utterance
   TRAIN evaluation. Therefore probe W-F1 of ~59 is **NOT**
   a CSS baseline regression and must not be compared to
   the paper 75.66 / original CSS 75.28 directly.
5. These data alone cannot rule out **nonlinear and
   label-dependent semantic interactions** nor prove
   B has zero influence on human emotions.
6. Next study must start with an independently formulated
   testable hypothesis, not a hyperparameter sweep of V14
   in reaction to its held-session results. Any future CSS
   TEST benchmark is already exposed from V11–V13 and
   not independently confirmatory.

Raw user-generated local server artifact:
`results/v14_b_incremental_diagnostic/report.json`
(not uploaded into repo). Metrics above were user-reported,
not independently recomputed from their server file.

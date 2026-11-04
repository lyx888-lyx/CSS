# V4 pre-design gate — MELD minority emotion confusion analysis

## Why no immediate V4 classifier change?

User's fixed split MELD validation results, CSS vs class-balanced train-only
SupCon (single seed, repeated inspected 115 dialogues):

| MELD class | Official name | VALID support | Original CSS F1 | CSS+SupCon F1 |
| --- | --- | ---: | ---: | ---: |
| 0 | Neutral | 517 | 86.13 | 85.95 |
| 1 | Surprise | 110 | 69.68 | 70.32 |
| 2 | **Fear** | **29** | **34.15** | **29.27** |
| 3 | Sadness | 75 | 54.93 | 54.68 |
| 4 | Joy | 180 | 68.47 | 68.26 |
| 5 | **Disgust** | **31** | **32.43** | **32.43** |
| 6 | Anger | 102 | 60.78 | 60.58 |

Official class mapping:
https://github.com/declare-lab/MELD (README, Labelling).

V3 overall W-F1 73.59 -> 73.36 (-0.23 pp); stable 82.25 -> 81.43;
changed 63.69 -> 63.70. No meaningful changed improvement.
The two rare classes contain only 60 / 1,044 validation utterances
(5.75%), so high class-specific F1 gains may not transfer to overall
weighted F1, particularly if neutrality or joy worsens.
Not enough evidence to say "class imbalance" is the only cause:
need recall, precision and error DESTINATION first.

## Read-only diagnostic code

`scripts/audit_meld_class_confusions.py` loads the strict
VALID-selected original CSS checkpoint and optionally the new V3
checkpoint. It checks all 115 validation dialogue indices against
split seed 10068 and original MELD TRAIN partition. No official
TEST content. Reports:

- train-support and validation-support for each class;
- predicted counts, TP, precision, recall, F1 per class;
- 7x7 confusion matrix with TRUE rows, PREDICTED columns;
- top 3 mistaken destinations for Fear, Disgust, Sadness,
  Anger and Neutral;
- same metrics within stable, changed and changed-after-other
  validation subgroups;
- if V3 checkpoint provided, classwise differences and net
  changed-correct count on exactly the same utterances.

No model fitting, weight tuning, class-threshold search,
ground-truth label input at inference or official TEST reading.

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v4-meld-class-confusions
python -m unittest discover -s tests -v

R=$(cat results/v3_css_supcon_pilot_last_run.txt)
python -u -m scripts.audit_meld_class_confusions \
  --baseline results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt \
  --candidate "$R/checkpoints/v3_supcon_w001_s10068_valid_best.pt" \
  --output results/v4_meld_class_confusions.json
```

If V3 checkpoint is missing, omit `--candidate ...`:
the original CSS diagnosis works independently.

## Actionable choice AFTER reading the confusion destinations

1. If Fear/Disgust have **very low recall but decent precision**
   and most are predicted as Neutral, consider training-only
   class-aware logit correction / balanced classification
   with fixed priors and weak regularization; baseline against
   class-weighted CE / balanced softmax using MULTIPLE seeds.
2. If Fear/Disgust have **reasonable recall but poor precision**,
   increased recall pressure or naive class weights could be harmful:
   inspect class-pair confusion and modality evidence instead.
3. If Fear and Disgust systematically mix with specific emotions,
   consider a class-pair structured margin or prototype
   contrastive objective based on training labels and verified
   affect geometry, not another generic SupCon.
4. If errors mostly occur during emotion changes after other
   turns, only then evaluate interaction-conditioned
   learning; require subgroup benefit beyond simple EMA.
5. If class-specific mistakes are unstable and counts are tiny,
   make a new independent validation split and avoid overfitting
   these 29/31 samples.

**Constraint:** V0/V1/V1.1/V2/V3 have already explored the SAME 115
validation dialogues. Repeated choices based on them bias results.
Do not publicly claim any gain unless a design fixed in advance
survives fresh dialogue splits, multiple model initialization seeds
and an untouched TEST run only once the method is finalized.

Do not call class balancing or confusion-aware prototypes
"innovations" simply for being combined with CSS; relevant 2026
EMART (ACL), EII-SCL (Interspeech), IDCL (ICASSP) and DnR (WACV)
already explore structured supervision and representation learning.

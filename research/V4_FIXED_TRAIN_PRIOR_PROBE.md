# V4: class-prior bias probe before more CSS training

## Fixed evidence from user-run V4 confusion report

Strict 1037 original-MELD-TRAIN dialogues / 115 VALID dialogues,
train seed and split seed 10068, 1044 VALID utterances, CSS F1=73.59,
TEST never used.

| Class | Name | TRAIN count | VALID support | Predicted by CSS | TP | Precision % | Recall % | F1 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Neutral | 4663 | 517 | **593** | 478 | 80.61 | **92.46** | 86.13 |
| 1 | Surprise | 1245 | 110 | 111 | 77 | 69.37 | 70.00 | 69.68 |
| 2 | Fear | 279 | 29 | **12** | 7 | 58.33 | **24.14** | 34.15 |
| 3 | Sadness | 719 | 75 | 67 | 39 | 58.21 | 52.00 | 54.93 |
| 4 | Joy | 1726 | 180 | 153 | 114 | 74.51 | 63.33 | 68.47 |
| 5 | Disgust | 262 | 31 | **6** | 6 | **100.00** | **19.35** | 32.43 |
| 6 | Anger | 1160 | 102 | 102 | 62 | 60.78 | 60.78 | 60.78 |

True Fear incorrectly predicted Neutral (10), Sadness (5),
Surprise (3). True Disgust incorrectly predicted Neutral (9),
Surprise (6), Anger (6). Sadness incorrectly predicted Neutral 23.
Neutral is predicted 593 times vs just 517 true occurrences.

This is a *rare-class recall / predicted-frequency deficit*
pattern. A precise inference of class priors being the sole
cause is NOT justified; acoustically/textually ambiguous features,
short utterances, weak examples and checkpoint calibration may
contribute too. Disgust 100% precision is estimated from just
6 TP; it is highly unstable.

## Preregister ONE standard prior correction (NOT our innovation)

Paper: Menon et al., **Long-tail learning via logit adjustment**,
ICLR 2021; https://research.google/pubs/long-tail-learning-via-logit-adjustment/
https://mlanthology.org/iclr/2021/menon2021iclr-longtail/

For original CSS probability vector p(t,c) and train class count n(c):

```
pi(c) = n_train(c) / sum_k n_train(k)
adjusted_score(t,c) = log(clip(p_css(t,c),1e-15,1))
                      - 0.15 * log(pi(c))
pred(t) = argmax_c adjusted_score(t,c)
```

The **one fixed** coefficient tau=0.15 is used for every class.
It is a deliberately conservative experimental choice, not
claimed optimal, and does not depend on the observed validation
labels. No special handling / hard-coded boost for Fear, Disgust,
or Neutral. Validation labels are used only AFTER prediction for
scoring and error migration. No training or official TEST calls.

Note: logit adjustment is designed for long-tail / balanced
classification risk. Weighted F1 puts strong weight on Neutral.
Improving Macro F1 while hurting Weighted F1 is possible and
would fail the user's primary performance objective.

`scripts/audit_v4_fixed_prior_logit.py` collects original frozen
CSS validation probabilities, applies the fixed train-only prior
correction, computes per-class precision/recall/F1, predicted counts,
and whole/stable/changed/after-other fixed vs broken examples.
`tests/test_v4_fixed_prior_logit.py` checks analytic behavior,
normalization, and paired metrics.

## Exact minimal run

```bash
cd /code/CSS
conda activate cmdpad
git -c http.version=HTTP/1.1 fetch origin
git switch analysis/v4-prior-logit-adjustment
python -m unittest discover -s tests -v
python -u -m scripts.audit_v4_fixed_prior_logit \
  --checkpoint results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt \
  --output results/v4_fixed_prior_adjustment.json
```

Expected: baseline CSS VALID weighted F1 about 73.59%;
7 class precision/recall/F1 plus changed/stable subgroup score
differences, and paired fixed/broken counts.

**Do not tune tau from the 115-dialogue validation results.**
This cohort has been repeatedly inspected after V0,V1,V1.1,V2,V3.
Any improvement here is exploratory and may be selection bias.

### GO / NO-GO

- GO only as research *signal*: meaningful class-2/5 recall increase
  together with improved or at least non-degraded overall F1,
  changed F1 and Neutral F1. Then compare against no-correction,
  train-only calibration, standard weighted CE/balanced softmax,
  in matched multiple seeds and fresh dialogue split.
- NO-GO if overall/stable performance drops or few rare-class
  examples change correctly. Stop editing thresholds on this
  validation set; other work on representation/rare-class prototypes
  may still be warranted, with a preregistered independent validation
  scheme.

## 2026 novelty / prior art warning

**UniMER**, EMNLP 2026 main:
https://aclanthology.org/2026.emnlp-main.1611/
Qin & Anastasiu propose class-prototype attention pooling
specifically to handle minority emotion representation collapse;
tested MELD, M3ED, AkaCE. No directly reproducible code path
has been verified in this project. This paper proves that
"class prototype + minority emotion" is NOT a new
research contribution by itself.

Other references: EMART, ACL 2026, emotion wheel supervision;
EII-SCL, Interspeech 2026, emotional inertia contrastive loss;
DnR, WACV 2026, modality representation factorization.

For future actual CSS method development, the question is
whether **speaker-context-conditioned selective calibration**
can distinguish a genuinely rare emotion signal from neutral
false positives WITHOUT hand-tuned class thresholds, and whether
that mechanism works across validation splits and datasets.
The PR is ONLY a simple foundational baseline probe.

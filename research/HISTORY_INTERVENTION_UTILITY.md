# History intervention utility learning — frozen CSS exploratory pilot

## Motivation and limits

Existing TRAIN-only fixed-epoch 200 four-fold audit found:
previous-own-key mask on 1597 changed targets +25 net correct (+1.565 pp);
3905 stable targets -65 net correct (-1.665 pp);
5502 matched targets -40 net correct overall.
Near-other and earlier-own controls have much smaller changed-turn gains.
An oracle that masks only actual shifts would add just 25 correct predictions
on all 5810 turns; simply predicting shifts is therefore not the main proposal.
This supports studying whether the specific intervention will help or harm,
not assuming every emotional shift should trigger suppression.

Related work to distinguish: SCoPE (arXiv:2607.20445), EII-SCL
(arXiv:2607.17366), HiRoC (doi:10.1007/s44443-026-01163-x).
Novelty and improvement are NOT established.

## Method

Two frozen candidate outputs: original CSS and latest same-speaker key mask.
Use the same matched cohort as the saved history masking audit. For training
labels, utility = correct(masked) - correct(original), i.e. +1, -1, or 0.
On disagreements, fit regularized multinomial logistic regression to predict
utility categories; choose masked iff P(+1) > P(-1). This decision optimizes
expected accuracy utility, not F1 directly. No threshold tuning.
Gold previous and current emotion labels are never inference inputs.

Arms (all retain original prediction on unmatched turns):
- baseline;
- always_mask;
- max_confidence: choose masked if its max probability is higher;
- shift_only: binary logistic regression on the same full feature vector,
  trained on true shift labels, choose masked on disagreement if P(shift)>0.5;
- probability_utility: 18 features (two 6-way predictions and difference);
- history_utility: same 18 features plus predicted previous-own one-hot,
  log1p lag, two probabilities on predicted prior class, and two indicators
  of matching the predicted prior class.

Fixed StandardScaler + sklearn LogisticRegression(C=1,max_iter=1000).
No CSS training, no gold labels in inputs, no additional external data.
For each outer fold, generate training-side intervention results ONLY using
that fold's checkpoint and its three TRAIN sessions. Fit selector before
loading that fold's held-out mask rows. Never pool the other folds' OOF rows
for training: those backbones saw this held-out session.
The training-side CSS predictions are in-sample, so learning from their
error patterns may not transfer. The outer sessions were already examined
during method development. This remains exploratory.

## Frozen screen

On pooled 5810 official TRAIN out-of-fold utterances, history_utility must:
- improve W-F1 by >=0.50 pp over baseline;
- improve W-F1 by >=0.20 pp over each probability_utility and shift_only;
- outperform max_confidence in W-F1;
- improve W-F1 in >=3 of 4 held-out sessions;
- improve changed-turn ACC and lose <=0.25 pp stable-turn ACC.

All must pass to continue this candidate. This is an engineering screening
criterion, not a significance test. Do not tune on the already inspected
held-out outcomes. Even a pass is not a new-paper claim; next work would
require independent evaluation, novelty verification and a more efficient
trainable model. Inference variants can be expensive; count their cost.

## Commands on /code/CSS, cmdpad environment

    git fetch origin
    git switch experiment/history-intervention-utility
    python -m unittest discover -s tests -p 'test_history_intervention_utility.py' -v
    python -u -m scripts.run_history_intervention_utility --smoke
    for s in Ses01 Ses02 Ses03 Ses04; do
      python -u -m scripts.run_history_intervention_utility --fold "$s" || break
    done
    python -u -m scripts.run_history_intervention_utility --aggregate

Uses existing results/css_session_confusion_stability/seed61078/ four
checkpoints, four original reports, cross_session_summary.json and
history_masking/Ses01.json ... Ses04.json. No need to rerun those held-out
masks. Training-side frozen inference is new, not GPU backpropagation.
Lower --batch-size to 2 or 1 if needed.
Outputs history_utility/Ses01.json ... Ses04.json and summary.json, no
overwrites. No Session5 TEST instantiation or evaluation.

Status: implementation and synthetic unit tests submitted; real server
tests and pilot results have not been executed here.

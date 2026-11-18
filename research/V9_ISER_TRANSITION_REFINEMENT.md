# V9 — ISER: inter-speaker emotion transition refinement
Date: 2026-11-18. Status: code committed; performance untested.

## Why this experiment is different
V8 conditional SPF gain was a NO-GO. Stop changing SPF/PGM.
V9 changes the class-decision layer instead, with a frozen ORIGINAL
full-SPF CSS model. It predicts a 7x7 conditional emotion transition
from intervening other-speaker raw multimodal features and marginalizes
the prior same-speaker predicted emotion distribution. It tests
incremental predictive information, not psychological causality.

For the latest same-speaker turn j before target t, only the open
interval (j,t) enters the other-speaker feature mean. By definition
all intervening turns have a different speaker from the target.
First speaking turns and consecutive same-speaker turns are
unchanged by EVERY refinement arm. The corrections therefore
have identical eligible-turn support; full CSS and residual-only
CSS are distinct backbone baselines.

Let p_j be the FROZEN CSS predicted distribution at j (not a label).
Let h be the concatenation of mean-pooled raw normalized text,
audio and visual features in (j,t). Normalize each modality per
turn to equal L2 norm before pooling. No emotion labels determine
h, j, eligibility or prediction features.

    T_m = row_softmax(S)
    T_c(h) = row_softmax(S + MLP(h).reshape(7,7))
    q_m = p_j @ T_m
    q_c = p_j @ T_c(h)
    z_ISER = log(p_CSS(t)) + 0.5 * (log(q_c) - log(q_m))

S is 7x7, fitted on all repeated-speaker TRAIN turns using the
predicted prior distribution and current target label NLL, then
frozen. This is a **marginal reference transition** learned from
observed conversations, not an identified pure self-only
counterfactual: intervening speech may influence its estimate.

The MLP is Linear(raw feature dimension,32), GELU, Linear(32,49),
last layer zero-initialized. Thus T_c starts identical to T_m and
the incremental correction starts at zero. No gate magnitude
sweep, class-prior tuning or real labels at inference.

## Six arms
1. Original frozen CSS full-SPF checkpoint (76.14% W-F1).
2. Existing independently retrained residual-only CSS (76.43%).
3. self: add 0.5*(log q_m - log pi_train) on eligible turns.
4. conditional: add 0.5*(log q_c - log pi_train).
5. incremental (ISER): add 0.5*(log q_c - log q_m).
6. shuffled: same architecture, incremental loss, seed and
   budget as ISER but context from randomly assigned TRAIN
   donor turns of a different dialogue. At validation use ONLY
   the TRAIN donor bank, not other validation samples.
   This is a negative-control diagnostic, not a deployment model.

pi_train is the smoothed empirical current-class prior among
eligible TRAIN turns. All refinement arms use the SAME frozen
full CSS backbone and SAME eligibility mask. The conditional and
incremental heads are separately trained with the same init and
minibatch order; they differ in the correction formula. Self is
the fixed fitted transition, not a capacity-matched neural head.
Compare against all controls, not only the original CSS.

## Fixed lightweight training protocol
- Existing V6.1 MELD split: 846 TRAIN, 87 already inspected VALID,
  875 validation utterances; original 115 and following 104
  validation dialogues excluded. Train seed 10068, split 36427.
- Strictly load the ORIGINAL full-SPF and residual-only checkpoints
  from V6.1; verify config, split IDs and reproduce F1 before training.
- Freeze CSS (eval/no_grad); cache original CSS TRAIN and VALID
  outputs once, and residual-only VALID separately.
- Only TRAIN labels enter loss or class-prior estimates.
  S fit 50 full-batch Adam epochs, LR 0.02.
- Each interaction head: AdamW LR 0.001, wd 0.001, 40 epochs,
  batch 256, head init seed90421, shuffle seed90422, gradient clip 5.
  Optimize final classification CE + 0.5 transition NLL.
- All adapter epochs fixed BEFORE running; **do not select a best
  adapter epoch by VALID**. Evaluate only after the fixed budget.
- Use the same already-inspected VALID for exploration, no TEST.
  No independent holdout or statistical significance claim.
- Original backbone was trained on these TRAIN dialogues, so
  its predicted histories on TRAIN are IN-SAMPLE, potentially
  overconfident relative to held-out predictions. This is a
  realistic stacking/overfit risk, not label leakage at inference.
  If V9 succeeds, independent confirmation should use
  out-of-fold backbone training posteriors or joint end-to-end
  training, fixed before held-out evaluation.
- Raw CSS model's encoder is bidirectional. The history *indices*
  and explicit raw context are past-only, but CSS p_j and p_t
  may encode future utterances. No online/causal model claim.

## Run (user server)
```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v9-iser-transition-refinement
python -m unittest discover -s tests -p 'test_v9_iser_transition.py' -v

python -u -m scripts.run_v9_iser_pilot \
  --output results/v9_iser/six_arm_validation.json
```

Use existing V6.1 checkpoint paths by default:
`results/v61_spf_paired_split36427_trainseed10068/checkpoints/v61_full_valid_best.pt`
and `v61_residual_only_valid_best.pt` in the same directory.
If moved, specify `--full-checkpoint` and `--residual-checkpoint`.
Optional `--cpu` (CSS inference may be slower).
JSON contains overall/group Weighted F1, macro F1, per-class,
predictions and paired fix/break counts, head states in adjacent .pt.
Output path must not exist to prevent overwriting results.

## Decision rule (exploratory, not a significance test)
Target overall >=77.14% (original CSS +1.00pp), greater than
residual-only 76.43%, and Changed >=69.00%, with a positive
Changed-after-other gain vs both original and residual-only.
To motivate an interaction-specific claim, incremental must
outperform both conditional and shuffled controls, not merely
a static prior or extra class parameters.
If all arms fail, STOP tuning this architecture on the 87 VALID;
report all results and reconsider feature quality or baseline.
If it shows promising gains, lock the method and confirm on
independent data/splits and multiple seeds. Dynamic transitions
and decision correction are existing ideas; novelty is NOT
established by implementation nor any one-run improvement.

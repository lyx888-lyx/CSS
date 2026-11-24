# Frozen CSS history-mask audit

## Hypothesis
Previous emotion is associated with specific wrong predictions on true emotion
transitions in the existing 4-fold TRAIN-only analysis. This does not establish
that historical evidence caused these errors. Test model sensitivity by
excluding one historical attention key in the already-trained original CSS.

## Protocol
- Use original TRAIN only (Ses01–04), fixed 200-epoch held-out checkpoints and
  frozen predictions from session_confusion_stability/seed61078.
- Verify checkpoint SHA256, fold metadata, and exact baseline argmax for each
  dialogue. Abort if the original predictions do not reproduce.
- For every target with prior own AND prior other history, select latest own
  turn, and the other-speaker turn closest to its position (ties: later turn).
  Additionally mask second-latest own history when available. Only speaker
  identity and position choose masks. No gold label chooses a mask or enters
  inference. All mask positions precede the target.
- Keep the target features, speaker embeddings, lengths, network parameters
  and original bidirectional architecture unchanged. Set one history key's
  u_mask to zero, excluding it from original contextual attention; each mask
  is a separate intervention. Compare on exactly matched targets.
- Report paired fixes/breaks, W-F1, accuracy, change in true-vs-previous-class
  probability margin, and net reduction in erroneous predictions of previous
  emotion. Gold labels define changed/stable and the 3 pair subgroups AFTER
  inference. Report per-session effects and lag differences. No significance
  claims; masks can be out of distribution and controls differ in role and lag.
- No model training or official TEST. This is a diagnostic, not a deployable
  decision rule or human causal inference.

## Commands
From /code/CSS with cmdpad activated, switch to experiment/history-masking-audit.

    python -m unittest discover -s tests -p 'test_history_masking_audit.py' -v
    python -u -m scripts.run_history_masking_audit --smoke
    for s in Ses01 Ses02 Ses03 Ses04; do
      python -u -m scripts.run_history_masking_audit --fold "$s" || break
    done
    python -u -m scripts.run_history_masking_audit --aggregate

Outputs: results/css_session_confusion_stability/seed61078/history_masking/
with one JSON per fold and summary.json. Refuses overwriting completed reports.
If GPU memory is insufficient, --batch-size 2 or 1 controls inference batching.

## Interpretation
Look for greater reduction of history-label errors and improvement of
true-vs-prior margins from masking previous own history than from masking
nearby other history, on changed and candidate-pair transitions. Check net
correctness, stable-turn harm, earlier-own control and per-session consistency.
A reduction in history errors alone can merely convert errors to another wrong
class. If the effect is absent or indistinguishable from controls, do not
proceed with a history suppression mechanism. If it exists, design a label-free
transition-sensitive gating hypothesis; no oracle use of actual past or
current emotion. The result remains exploratory, since the candidate pairs
and Session4 were previously examined. No human causal effect is established.

Status: implementation and synthetic tests committed, not executed against
the actual IEMOCAP data/checkpoints in the server environment.

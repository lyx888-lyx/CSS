# V1 history-path attribution — no retraining

## Evidence prompting this diagnosis

Paired MELD dialogue-validation run (training seed=10068,
split seed=10068, 115 dialogues, 1044 utterances):

| Model | Validation weighted F1 | Best epoch |
| --- | ---: | ---: |
| Original CSS, independently trained under same protocol | 73.59% | 52 |
| V1 pre-context both | 73.24% | 64 |

On the same validation utterances, V1 fixed 10 CSS mistakes but
broke 14 CSS-correct predictions. 32 final predictions changed.
For speaker-stable utterances, V1 fixed 1 and broke 6.
For speaker-changed utterances, fixed 5 and broke 6.
The `changed_after_other` group fixed 5 and broke 5.

These descriptive, one-seed observations **do not** establish emotion
causation, robustness, or novelty. In particular, V1's `first`
speaker appearances can have prior OTHER-speaker turns despite no
same-speaker history.

## Question

Does V1 rely more heavily on its own-speaker or other-speaker
historical difference contribution, and which changes coincide with
stable/changed validation classification errors?

## Exact experiment

Take the **same trained** V1 checkpoint and evaluate four settings
over the **same validation dialogues**:

- `both`: intact V1, no intervention.
- `self`: zero the output of `other_diff` in each modality.
- `other`: zero the output of `self_diff` in each modality.
- `neither`: zero both outputs; every V1 residual is an identity.

The tool uses temporary PyTorch forward hooks on the two final
projection layers, so attention masks, history summaries, speaker
identities, learned feature gates, backbone weights and output head
are identical. Hooks are removed in a `finally` block.

An independently trained CSS baseline is also evaluated, but note:
**self/other/neither are NOT independently retrained architectures.**
Inference-time disabling can disturb what downstream parameters
expect. This diagnostic determines model dependence and helps
generate hypotheses; it does NOT prove a branch caused emotion
contagion or would help if retrained.

## Server instructions

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v1-history-paths
python -m unittest discover -s tests -v
R=$(cat results/v1_meld_pair_last_run.txt)
python -m scripts.diagnose_v1_history_paths \
    --css "$R/checkpoints/css_valid_best.pt" \
    --v1 "$R/checkpoints/v1_precontext_valid_best.pt" \
    --output "$R/history_paths.json"
```

This performs inference only, requires no extra checkpoints or new
training, and does not read official TEST dialogues. The printed
weighted F1 table covers overall, stable, changed and
changed-after-other utterances, plus number of changed predicted
labels compared with full V1. `history_paths.json` also includes
per-class F1 and exact corrections/regressions vs CSS and V1.

## Next decision

If disabling one branch *on the trained V1* leaves stable predictions
less disturbed, consider a **controlled training ablation** of that
branch with an independently trained checkpoint. Only after that
should we implement a selective or cross-modality reliability
mechanism in V1.1, with clear ablations.

Do not choose V1.1 on the basis of the original-author
TEST-selected scores. For final claims, repeat multiple training
seeds and another split/dataset. The original CSS backbone remains
bidirectional.

# V10 — IEMOCAP emotion-aware text representation pilot

## Decision after V9
V9's incremental transition correction degraded overall and change-group
W-F1 and did not beat the shuffled-context diagnostic. Stop that route.
Move from repeatedly adjusting existing features on MELD to evaluating
additional emotion-supervised text representations on dyadic IEMOCAP.
No claim that the original feature quality caused V9 failure.

## External resource and exact inputs
CSS/SDT IEMOCAP pickle already loads raw transcripts (videoSentence),
speaker identities (videoSpeakers), and 1024/1582/342 text/audio/visual
features. No new raw audio download required. Confirm transcripts align
with all retained input turns before extraction; fail if not.

External source:
https://huggingface.co/SamLowe/roberta-base-go_emotions
Pinned revision b3a4ef6cd1548efa18b8dc1483f28ad58538ab63,
safetensors, frozen RoBERTa-base trained on Reddit GoEmotions
28-label multilabel classification. Use final 768-dimensional CLS
hidden state, NOT a manually mapped 28-class prediction.
Model card reports GoEmotions training, not IEMOCAP; full external
training corpus provenance cannot be independently audited here.

Three arms (from scratch, same original CSS/PGM training procedure):
- original: original 1024-dimensional SDT text input.
- current: original + frozen 768D current-utterance emotion feature.
- speaker_context: original + frozen 768D feature from current
  utterance paired with up to 4 preceding retained utterances,
  explicitly marked SELF/OTHER relative to target speaker, newest first.

Target text receives up to 128 tokens and is preserved before history
truncation; combined maximum 256 tokens including special tokens.
No future turns, true emotions, gold changes or speaker identity
leak into the added encoder. CSS backbone itself is bidirectional,
so this is NOT a strictly causal online system. The intermediate
representation is a hypothesis, not a verified improvement.
The external encoder was tuned on individual Reddit texts, not dialogue
pairs, so contextual input may cause distribution shift.

Append extra feature after label-free per-utterance L2 norm matching.
Initialize all original CSS layers exactly as original; zero-initialize
only the extra 768 text projection input channels so the initial
mapping is unchanged. Current and context arms have exactly the same
parameter count and architecture; they differ only in external feature
inputs. Comparison to original changes input information and parameter
count; do NOT attribute a gain purely to context without the current arm.
No SPF/PGM change. Original audio/visual remain unchanged.
The prior research goal +1 pp is an aspiration, not a guarantee.

## Protocol
- Original CSS IEMOCAP TRAIN pickle keys only. Require Ses01–Ses04
  as original train sessions; **no session 5 inference or scoring**.
- Session 1–3 TRAIN; session 4 VALID (held-out speakers). This is a
  fixed development session, not entirely pristine historically:
  the user previously reproduced on the original official protocol.
- 200 epochs / seed61078 / LR 1e-4 / batch32 / original PGM and
  fixed original IEMOCAP class weights. Weights are inherited constants
  from the original code, not re-estimated from held-out labels.
- VALID W-F1 selects checkpoint. Fixed budget and no score-driven
  hyperparameter sweeps. No selecting by official TEST.
- Current/context encoder frozen, cached once. Only train dialogue
  labels enter CSS training. VALID text features are deterministic
  frozen encoder inference, no refitting on validation.
- Record dialogue IDs, source pickle SHA256, model revision, predictions,
  weighted and macro F1, group support and per-class results.
- Single session/seed pilot: neither novelty nor generalization proven.
  Stronger external representations alone are a feature baseline,
  not a paper-level novel algorithm. Require current-vs-context
  advantage and independent confirmation before speaker-specific claim.

## Server workflow

Use existing cmdpad environment and CSS data. Install compatible text
encoder dependencies only if not already installed (do not replace torch).

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v10-iemocap-emotion-text-features
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  transformers==4.44.2 'safetensors>=0.4.3,<1'
python -m unittest discover -s tests -p 'test_v10_emotion_text_features.py' -v
python -u -m scripts.run_v10_iemocap_features --prepare-only
python -u -m scripts.run_v10_iemocap_features --smoke
python -u -m scripts.run_v10_iemocap_features
```

First prepare downloads ~499 MB safetensors plus tokenizer/config and
caches original TRAIN-session text hidden features for both modes.
Use a supported network or prepopulate Hugging Face cache with the
pinned model revision; --local-files-only uses existing cache.
--encoder-batch-size 8 can reduce encoder GPU memory if necessary.
No API credentials required for public encoder.
Smoke uses 1 epoch on context arm and its own output directory.
Full run trains three models 200 epochs each, prints all-group report,
saves best checkpoints and results/v10_iemocap/paired/report.json.
No overwrites; if interrupted preserve logs/checkpoints, use a new
output directory only for an explicitly intended rerun, not seed search.
User GPU tests and metrics not yet run.

## Decision
If current improves original but speaker_context does not improve current,
external features are useful but no demonstrated extra history benefit.
If speaker_context improves original and current on overall W-F1 and
Changed/Changed-after-other, consider an independently locked multisession
and multiseed evaluation, capacity-matched controls and a genuinely
specific response-encoding mechanism. Do not compare raw numbers against
prior MELD or official test scores. If neither improves, do not endlessly
adjust text history window, pooling, or scale on session 4.


## User-run V10 results — fixed IEMOCAP session 4 development set

TRAIN: sessions 1–3, 90 dialogues / 4298 turns. VALID: session 4,
30 dialogues / 1512 turns. Same seed61078 and 200 epochs. No TEST.

| Group | N | Original | Current | Speaker context |
|---|---:|---:|---:|---:|
| all | 1512 | 68.41 | 68.32 | 68.08 |
| first | 60 | 73.31 | 73.15 | 67.99 |
| stable | 1034 | 75.42 | 75.70 | 75.49 |
| changed | 418 | 50.48 | 49.56 | 49.79 |
| stable_after_other | 685 | 74.60 | 74.86 | 74.82 |
| changed_after_other | 310 | 49.75 | 49.90 | 50.57 |

Selected epochs: 184 / 188 / 188.
Source: results/v10_iemocap/paired/report.json.
Both enhanced arms fail to beat original on overall W-F1. The
context arm improves changed_after_other by +0.82 pp over original
and +0.67 pp over current but does not improve all changed turns.
This is a single-session/seed exploratory subgroup observation,
not independent validation or causal evidence. The original
stable-minus-changed W-F1 gap is 24.94 pp; subgroup class mixes and
difficulty may differ. Do not infer that all of this gap is due
to other-speaker effects.

Decision: stop this frozen GoEmotions feature append route.
No window, scale, or model selection sweeps on session 4.
Retain CSS as a comparator and reconsider a baseline with
trainable task-adaptable audio/text encoders, e.g. EMART, after
checking the user's earlier reproduction result and protocol.
A new backbone alone is not a novel method. The official TEST
must not select future models or hyperparameters.

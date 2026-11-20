# V11 — Cross-speaker acoustic reaction representation on CSS (IEMOCAP)

## Decision

Stay **only in /code/CSS**. Do not run another EMART repository.
Use IEMOCAP's two-person dialogues and existing CSS (AAAI 2026)
feature pickle. Reuse the V10 **original CSS** session-4 VALID
checkpoint instead of retraining the baseline.

User-run V10 on TRAIN sessions 1–3 (4298 turns) and already-inspected
VALID session 4 (1512 turns), seed61078, 200 epochs:
- original CSS overall W-F1 68.41 / Changed 50.48 /
  Changed-after-other 49.75
- frozen GoEmotions current 68.32 overall
- frozen GoEmotions SELF/OTHER 68.08 overall
- history arm Changed-after-other +0.82 pp alone, but no
  overall or general Changed benefit.
V10 is NO-GO for simple frozen text feature concatenation.

Previous CSS V1 precontext_shift already tried generic
same/other history feature shifts, and V8/V9 gated/postprocessed
predictions failed. This V11 targets an **acoustic-only**
A(previous) -> B(intervening) -> A(current) structural
contrast in **trainable pre-context multimodal representations**,
not a generic history gate or class posterior correction.
It is a falsifiable PILOT, not a demonstrated novel method.

## Method

For current A utterance i and most recent earlier A utterance j,
collect all non-A speaker turns B strictly between j and i.
If no j or no intervening B, do not change the audio embedding.

Original projected acoustic vectors a_j, a_i, and mean
intervening B acoustic b_(j,i):
- self difference s_i = LayerNorm(a_i - a_j)
- interposed response evidence r_i = LayerNorm(b_(j,i) - a_j)
- elementwise interaction c_i = s_i * r_i

A compact trainable residual adapter receives [s,r,c], with
Linear(3D->128), GELU, Linear(128->D). The final layer is
exactly zero-initialized, so **both experimental models begin
with identical CSS predictions** before training.

Two parameter-count-matched new arms, trained from scratch:
- **self:** [s, 0, 0]; only A's own acoustic shift.
- **reaction:** [s,r,c]; includes intervening B's evidence.

Both apply to EXACTLY the same A...B...A target turns and
share all original CSS loss, audio/text/visual inputs and
optimizer. No true emotion labels, previous gold state or
changed/stable tag is used in forward inference.
The CSS dialogue encoder is still bidirectional, so the
model cannot claim strictly online/causal inference.

This differs from V9, which added a post-hoc class-Logit
transition prior, and V8, which gated the high-order SPF path.
A positive result must show **reaction > self** as well as
reaction > CSS. A similar gain in both arms only supports
self-acoustic-difference modeling, not unique cross-speaker
information. Speaker relations and representation differences
have abundant prior art; novelty NOT established.

## Strict same-protocol pilot

- Existing IEMOCAP TRAIN pickle sessions 1–4 only; session 5
  official TEST is never evaluated.
- Fixed TRAIN sessions 1–3, VALID session 4 (previously inspected).
- Original 1024D text, 1582D audio and 342D vision CSS features.
  No new feature extraction, no Hugging Face/EMART dependencies.
- Seed61078, batch32, 200 epochs, CSS original PGM, lr1e-4,
  original IEMOCAP class weights, dropout0.5.
- Original CSS checkpoint:
  `results/v10_iemocap/paired/original_valid_best.pt`;
  the runner validates its data split/configuration and checks
  re-inference W-F1=68.41%.
- Two new models differ ONLY by whether r,c are masked;
  same parameter count, identical baseline weights and
  adapter initialization; each selected on same VALID W-F1.
- First run 1-epoch smoke on reaction alone, then real 200-epoch
  paired run, and save all groups, per-class results and
  prediction correctness flips.
- No loss or learning-rate sweep; no reselecting a different
  VALID session in response to disappointing results.

### User server commands

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v11-speaker-relative-acoustic-reaction

python -m unittest discover -s tests -p 'test_v11_acoustic_reaction.py' -v

# One-epoch full-pipeline check, separate output:
python -u -m scripts.run_v11_iemocap_reaction --smoke

# If smoke passes, 200 epochs in both new arms:
python -u -m scripts.run_v11_iemocap_reaction
```

Checkpoints:
`results/v11_iemocap_acoustic_reaction/paired/self_valid_best.pt`
and
`results/v11_iemocap_acoustic_reaction/paired/reaction_valid_best.pt`.

Summary:
`results/v11_iemocap_acoustic_reaction/paired/report.json`.
Keep logs and JSON; if cleaning disk, retain at least model configs,
selected epochs, class/turn predictions and paired-fix counts.

## Predetermined decision

**Pilot GO only if** reaction exceeds original CSS by at least
+1.00 pp Overall W-F1 AND outperforms matched self by at least
+0.50 pp Overall W-F1, without reducing Changed and
Changed-after-other vs original CSS. This is a research target
and explicitly NOT a promised score or proof of statistics.

If it passes: freeze implementation, preregister multisession
IEMOCAP development comparisons across sessions 1–4 and
multiple training seeds, and perform independent TEST once only
after model selection. Additional seen-session validations
cannot count as independent proof of generalization; use
held-out speaker/session-aware methodology. Also test MELD
under a separate, locked protocol and analyze modality
response attribution before novelty claims.

If it fails: STOP modifying this adapter's bottleneck,
scale, history selection, epoch length or loss coefficients
to chase the already-inspected session-4 VALID. Record the
negative result and switch to a stronger baseline or
end-to-end trainable speech encoder under a fresh protocol.

## Development cautions

- V10 original baseline and V11 both share the previously used
  session-4 VALID; a gain is exploratory only.
- `changed_after_other` is defined from gold labels solely
  AFTER predictions for subgroup analysis, not a causal label.
- The adapter operates on CSS pre-extracted acoustic embeddings,
  not raw prosody/pitch recordings.
- Reaction arm sees more active features than the masked self
  arm, even though the parameter counts and initial weights
  match. An advantage needs negative controls and independent
  replications to attribute it to real inter-speaker response.
- No code has been run on the user's GPU by the assistant.

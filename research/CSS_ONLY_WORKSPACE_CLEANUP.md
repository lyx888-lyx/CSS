# CSS-only workspace and safe local experiment artifact retention

All future implementation and training should remain in `/code/CSS`.
Do not switch to the separate EMART repository. Earlier EMART
paper/methods may inform CSS experiments but may require raw audio
that the CSS SDT pickle alone does not contain.

## Critical warning

The assistant can inspect GitHub source and the user's screenshot,
but **cannot list local /code/CSS contents or verify every artifact**.
These are file-use recommendations, not an executed deletion.

- NEVER delete `data/iemocap_multimodal_features.pkl` or
  `data/meld_multimodal_features.pkl`.
- Preserve author reference checkpoints `checkpoints/iemocap.pt`,
  `checkpoints/meld.pt` (if verified they really are the author's
  files), and reproducible comparison baselines
  `checkpoints/iemocap_baseline_best.pt`,
  `checkpoints/meld_baseline_best.pt`,
  `checkpoints/meld_strict_base_valid_best.pt`.
  Note that original upstream baseline training used
  TEST-based checkpoint selection; preserve that provenance.
- Preserve the exact V6.1 full and residual checkpoints
  `results/v61_spf_paired_split36427_trainseed10068/checkpoints/
  v61_full_valid_best.pt` and
  `v61_residual_only_valid_best.pt`, at least until all
  original comparisons and audit results are backed up.
- Preserve `results/v10_iemocap/paired/original_valid_best.pt`:
  this is the same-session-4 reference for new work.
- ALWAYS preserve research logs, configs, selected epochs,
  prediction reports and per-seed JSON in each formal
  experiment directory, or export them to off-server backup
  before deleting checkpoints. A model checkpoint without
  a reproducible protocol is of limited research value.

## Small, low-risk first deletion

The script `scripts/cleanup_css_artifacts.py` identifies only:
- Direct root-level `checkpoints/*.pt` whose names contain
  `smoke` or `gradient_probe`.
- These explicitly named smoke directories:
  `results/v2_smoke_adaptive_other`,
  `results/v3_supcon_smoke`,
  `results/v8_other_smoke`.

No other directory is selected by this script, and the default
command deletes nothing.

```bash
cd /code/CSS
python scripts/cleanup_css_artifacts.py
# review each printed file and size; optional after backup
python scripts/cleanup_css_artifacts.py --delete-smoke
# answer DELETE_SMOKE at interactive prompt only if correct
```

These are **smoke or diagnostic** artifacts, not matched
publication-quality checkpoints. Deleting them is optional.

## Secondary review: full negative-result model weights

Keep the reports and experiment logs, but after off-server backup
the heavyweight retrainable checkpoints may be removed from:
- `results/v7_spf_seed21431_split36427/checkpoints/`
- `results/v7_spf_seed31907_split36427/checkpoints/`
- `results/v8_conditional_spf_seed10068_split36427/checkpoints/`
- `results/v10_iemocap/paired/current_valid_best.pt`
- `results/v10_iemocap/paired/speaker_context_valid_best.pt`

Do **not** delete the entire directories. Their
`paired_validation.json`, `five_arm_validation.json`,
`report.json`, training logs, and selected-epoch metadata
record important negative and control results.
V7's summary only requires per-seed paired JSON, not the
model checkpoints; V8's five-arm result is already saved
as JSON. Re-evaluating exact predictions in the future
would, however, require checkpoints.

`results/v9_iser/six_arm_validation.json` contains all
predictions/paired changes; its adjacent `.pt` is a
small state archive that is better retained.
`results/v10_iemocap/emotion_text_cache.pt` is a deterministic
external-encoder feature cache: may be regenerated, but
requires GoEmotions model access and compute; delete only
if abandoning that research avenue.

In the screenshot, `results/author_meld`,
`v1_meld_pair`, `v2_speaker_state_suite`,
`v3_css_supcon_pilot`, `v11_lambda25_pilot`
contain historical trial reports whose exact local contents
are not available to the assistant. Keep their JSON/text
summaries until audited individually. Avoid wildcard deletion.

## Space check / no deletion

```bash
cd /code/CSS
du -sh data checkpoints results
du -sh results/* | sort -h
find checkpoints -maxdepth 1 -type f -printf '%s %p\n' | sort -nr | head -25
find results -type f -name '*.pt' -printf '%s %p\n' | sort -nr | head -40
```

The command `find ... -printf` requires GNU find (typical Linux).
Output sizes are bytes. If you want a second cleanup phase,
provide the above size inventory; decide on exact real files,
their backup status, and references before deleting formal
weights.

Adding `results/` to .gitignore stops new local result
files from appearing as Git changes, but it does NOT delete
anything nor affect already-tracked files.

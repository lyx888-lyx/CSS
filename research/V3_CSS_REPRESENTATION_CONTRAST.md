# V3 pivot — train-only CSS representation contrast, not posthoc head voting
Date: 2026-11-04.

## Why the previous V3 modality-vote hypothesis is rejected

Same exact 115 MELD VALID dialogues / 1,044 utterances:
- Strict original AAAI 2026 CSS fused weighted F1 **73.59**,
  stable 82.25, changed 63.69, changed-after-other 64.42.
- Separate text head F1 72.69; audio 71.05; visual 71.27.
- Original CSS has **261 classification errors**.
  Only **11 / 261** (=4.21%) have even one unimodal head
  whose top-1 matches gold; only **5 / 261** have two or
  more heads correctly predict gold.
  On all 261 errors, 250 are shared by all four top-1 decisions.
- 17 originally correct fusion decisions would disagree with a
  *wrong* two-of-three unimodal majority.
- Preregistered no-training consensus override triggered 10 times:
  3 corrected, 6 broke, 1 changed wrong-to-other-wrong.
  Fused VALID overall weighted F1 **73.59 -> 73.16 (-0.43 pp)**;
  stable 82.25 -> 80.54, changed 63.69 -> 63.86.

Thus the model needs stronger feature discriminability or classification
boundaries, NOT merely a learned mixture of the same three top-1
classifiers. An oracle select-one-unimodal-head correction would
improve accuracy by at most 11/1044 = **+1.05 pp**, though this
does not bound W-F1 for all possible REAL-VALUED head fusion.

## 2026 related methodology and realistic reuse

- **EMART**, ACL 2026 main:
  https://aclanthology.org/2026.acl-long.1875/
  https://github.com/DILAB-HYU/EMART
  Uses text-led audio fusion plus emotion-wheel supervised contrastive
  representation learning. It has different raw text/audio embeddings;
  must not present its reported scores as CSS baseline results.

- **EII-SCL**, Interspeech 2026 accepted:
  https://arxiv.org/abs/2607.17366
  Emotional inertia informed supervised contrastive samples.
  Direct "temporal inertia contrastive" is ALREADY prior art.

- **IDCL**, ICASSP 2026 oral:
  https://github.com/hyuki0003/IDCL
  Official `official` branch contains `idcl/loss.py`, KNN
  dialog-level multimodal contrastive loss, package setup
  available in repository. This is code-backed related research
  but not a CSS native plug-in until dimensions and padding
  concerns are addressed (official loss mean-pools padded L).
  Different backbone and IEMOCAP evaluation were reported.

- **DnR**, WACV 2026:
  https://openaccess.thecvf.com/content/WACV2026/html/Mai_Divide_and_Refine_Enhancing_Multimodal_Representation_and_Explainability_for_Emotion_WACV_2026_paper.html
  Has official runnable code for MMGCN, DialogueGCN, MM-DFN and
  Late Fusion, NOT native CSS; heavy pretraining.

- **MoWER**, Knowledge-Based Systems 2026:
  https://www.sciencedirect.com/science/article/pii/S0950705126007707
  Also explores a learnable emotion-wheel prior and feature/label
  alignment. Emotion-wheel-only formulation is NOT novel.

All avenues above are already published research. **This V3
prototype does not claim any standalone novelty.** First check
improvement before contemplating a new scientific method.

## One low-intrusion test: CSS + class-balanced supervised contrastive loss

New file `research/v3_contrastive.py`:
- original CSS PGM loss remains intact;
- read CSS `last_gate` fused `[B,L,1024]` representations via
  TEMPORARY forward hook only while TRAINING;
- build supervised pairs from TRAIN gold emotion labels only:
  same-class non-self examples are positives, all other real
  utterances are candidate negatives; padding fully excluded;
- apply cosine-normalized InfoNCE with temperature 0.2;
- each present emotion class contributes equally to anchor loss,
  to mitigate domination by frequent MELD Neutral labels;
- add `0.01 * SupCon` to the original PGM loss;
- model forward and inference are unchanged, so standard CSS
  checkpoint loader and comparison work without any special code.

This is intentionally simpler than EMART/EII-SCL:
there is NO artificial emotion wheel prior, no hard-coded
psychological valence-arousal labels, and no temporal reweighting
mask to overfit the same validation split. If it fails, STOP.

### First run: unit tests + one epoch smoke ONLY

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch experiment/v3-css-representation-contrast
python -m unittest discover -s tests -v

python -u -m scripts.train_research \
  --dataset MELD --seed 10068 --split-seed 10068 \
  --speaker-mode off --precontext-mode off \
  --v3-supcon-weight 0.01 --v3-supcon-temp 0.2 \
  --epochs 1 --run-name v3_supcon_smoke \
  --checkpoint-dir results/v3_supcon_smoke
```

One-epoch log should include:
`V3_SCL_DIAG epoch=1 eligible_anchors=... mean_scl=... weight=0.01000`;
and checkpoint `results/v3_supcon_smoke/v3_supcon_smoke_valid_best.pt`.
Never treat epoch-1 scores as efficacy.

### After smoke passes: one pre-registered 60 epoch pilot

```bash
cd /code/CSS
conda activate cmdpad
PY_EXE=$(command -v python)
tmux new-session -d -s css_v3_rep \
  "cd /code/CSS && CSS_PYTHON='$PY_EXE' bash scripts/run_v3_css_supcon_pilot.sh"
```

Read summary when complete:

```bash
R=$(cat results/v3_css_supcon_pilot_last_run.txt)
cat "$R/status.txt"
cat "$R/summary.txt"
```

The script compares the original saved 100-epoch CSS
VALIDATION-selected checkpoint vs the new 60-epoch CSS+SupCon
VALIDATION-selected checkpoint, **not** official TEST.
Original CSS best epoch was 52, so 60 is a feasibility budget,
but for final comparison new and baseline must be retrained with
IDENTICAL max-epoch budget, three seeds, ideally fresh dialogue
split and second dataset. Using original 115-dialogue VALID
repeatedly risks exploratory overfitting. We do not tune
weights or temperature based on that set.

GO criterion (exploratory): nontrivial improvement in overall
weighted F1 with no degradation on changed and stable subgroup.
If test fails or no improvement, do not stack additional SupCon
gates or arbitrary weight searches; pivot to independent data
quality/feature error analysis, not blind model complexity.
No new dataset files or GPU training are performed by assistant.

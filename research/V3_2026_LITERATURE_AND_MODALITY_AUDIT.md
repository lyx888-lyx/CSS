# V3 evidence-first shortlist and CSS modality repair audit
Date: 2026-11-04.

## Goal / constraints

The real user goal is **first obtain a reproducible improvement in
MELD emotion recognition while retaining 2026 high-level literature
links and credible experimental comparisons**, not build speculative
modules or claim novelty before measurements.

Previous matched experiments:
- Original AAAI 2026 CSS VALID 73.59 overall / 82.25 stable /
  63.69 changed / 64.42 changed-after-other (seed=split=10068).
- V0 and V1 did not improve; independently retrained V1-other
  73.55 VALID. V1.1 lambda=2.5 fell to 60.16 VALID, best epoch 7.
- V2 lightweight first-order state five-arm pilot: Current 72.12,
  EMA 72.48, adaptive 71.90, adaptive-other 72.11; the
  trained adaptive-other branch changed only 16/1044 argmax
  predictions when zeroed, with no robust benefit.
- Classical GM(1,1) is not a suitable main module:
  only 253/1044 VALID positions have 4 prior own-speaker
  turns, only 1.58% of classwise historical probabilities pass
  its usual level-ratio admissibility check, and it underperformed
  persistence and EMA.

These failures suggest an **evidence selection and reliability**
bottleneck, not a reason to stack more speaker-memory gates.

## Public, verifiable 2026 source shortlist

| Source | Venue | Public code / readiness | What can be reused |
| --- | --- | --- | --- |
| [CSS](https://ojs.aaai.org/index.php/AAAI/article/view/39602) | **AAAI 2026** | Existing local valid-best checkpoint reproduced by user | Strong, already-reproduced base with PGM/SPF and THREE modality decision heads |
| [EMART](https://aclanthology.org/2026.acl-long.1875/) | **ACL 2026 main conference** | [Code](https://github.com/DILAB-HYU/EMART), author supplies train/infer scripts and some externally hosted checkpoints; **uses different raw text/audio processing** and paper metric conventions, so no direct plug-in result | Text as primary semantic channel, audio complement; affect-space-aware labels |
| [DnR](https://openaccess.thecvf.com/content/WACV2026/html/Mai_Divide_and_Refine_Enhancing_Multimodal_Representation_and_Explainability_for_Emotion_WACV_2026_paper.html) | **WACV 2026** | [Code](https://github.com/mattam301/DnR-WACV2026), executable backbones MMGCN/DialogueGCN/MM-DFN/Late Fusion; **CSS not among them**. `scripts/atv.sh` defaults to 300 pretrain epochs for refinement. | Separating unique/redundant/synergistic information, pre-fusion representation improvement; code adaptation is a substantial experiment |
| [SURE](https://arxiv.org/abs/2604.01916) | **ICASSP 2026** | [Public repo](https://github.com/swaggy66/SURE), documents external SDT features and training script, but complete reproducibility and run.sh/MELD setup not established | Uncertainty-aware experts plus conversational reasoning |
| [HiRoC](https://link.springer.com/article/10.1007/s44443-026-01163-x) | Journal, 2026 | Peer-reviewed paper; code reproducibility unverified | **Overlapping PRIOR ART**: historical relevance selection, speaker-typed routing, cross-modal disagreement-based decision correction. Do NOT claim a generic discrepancy gate is novel |
| [DialogueSSD](https://ieeexplore.ieee.org/abstract/document/11663388) | **IEEE TAFFC 2026** | Journal publication verified, code status unverified | Explicit state-space and emotion-shift supervision; warns against claiming novelty for an ODE/memory module |
| [Causal-ERC](https://ojs.aaai.org/index.php/AAAI/article/view/40402) | **AAAI 2026** | Paper verified, code and input dependencies not verified | Speaker-aware causal *prompting* for LLMs; not directly reusable within lightweight CSS |

Rankings are based on **fit to the existing trained CSS** and data
readiness, not an assertion of guaranteed F1 gain, paper CCF ranking
for every auxiliary venue, or independent novelty. An EMART run
using different pretrained features, modalities, or test selection is
NOT an architecture-matched score against CSS.

## First falsifiable V3 idea: can CSS errors be repaired by modality heads?

Original CSS exposes `outputs[0..2]` as text/audio/visual log-prob
distributions, and `outputs[4]` as fused probabilities. No new
embedding, external package, backprop or large GPU run is needed to
check if the multimodal head ignores potentially useful single-modal
evidence on misclassified utterances.

New script `scripts/audit_css_modality_repair.py`:
- Load the exact VALIDATION-selected MELD CSS checkpoint from the
  earlier strict research experiment. Reject V0/V1 checkpoints and
  TEST-selected checkpoint. Use original TRAIN partition's
  **same 115 validation dialogues**, not official TEST.
- Re-evaluate CSS and each of three modality classification heads
  on aligned utterances with subgroup metrics.
- Compute an **ORACLE upper-bound diagnostic only**, such as:
  how many FUSED CSS errors had at least one unimodal head correct;
  how many had 2 or more agreeing correct heads; how many correct
  fusion predictions would be harmed by a wrong modal majority.
  **Gold labels never enter the correction predictor.** These
  oracle counts are NOT a valid deployable method.
- Evaluate exactly ONE prespecified, label-free rule:
  if fused top confidence <0.60 AND at least two modality heads
  agree on a different class AND their mean probability for
  this class exceeds fused max probability +0.05, override
  only the argmax predicted label. NO thresholds are tuned on
  validation labels.
- Report frozen rule overall/stable/changed/after-other F1,
  paired fixed/broken counts, fraction activated, and potential
  rescue opportunity under modality disagreement.
- Save JSON diagnostics and STOP if there are not enough
  recoverable baseline errors or the untrained gate does not
  indicate a viable calibration signal.

Commands:

```bash
cd /code/CSS
conda activate cmdpad
git fetch origin
git switch analysis/v3-css-modality-disagreement
python -m unittest discover -s tests -v

python -u -m scripts.audit_css_modality_repair \
 --checkpoint results/v1_meld_pair/20261103_052259/checkpoints/css_valid_best.pt \
 --output results/v3_css_modality_repair_audit.json
```

A synthetic CPU unit test exercises override, confidence
fallback, no-consensus, oracle opportunities and paired counts.
No code uses official MELD TEST, true previous emotion labels for
inference, or model updates.

## GO/NO-GO

1. If many fused errors have modal evidence correctly disagreeing,
   investigate **modality reliability / conditional evidence
   selection** (SURE inspiration, HiRoC prior art), but train
   a small correction head on **TRAIN-only or train-OOF**
   predictions; never fit it on the 115 validation labels.
2. Compare a simple mean-probability fusion or fixed weighted
   ensemble first; a complex confidence model must beat that.
3. A conditional-repair model must prove nontrivial improvement
   on *changed* turns AND whole validation without harming
   *stable* turns, at three fixed seeds, before making a claim.
4. If CSS unimodal heads almost never provide a correct
   alternative to fused mistakes, STOP, do not tune the gates.
   Consider EMART-style affect-space contrastive supervision
   or DnR representation factorization as a separate **heavier**
   follow-up. DnR official implementation requires pretraining
   and different backbone integration; do not imply drop-in
   compatibility with CSS.

## Novelty caveat

DnR studies uniqueness/redundancy/synergy and CSS already
models synergy through polynomial fusion; those concepts overlap.
SURE studies uncertainty-aware fusion. HiRoC specifically already
studies disagreement-aware corrective classification and
speaker-typed selective routing. Therefore "CSS + confidence gate"
or "CSS + DnR" is only a **performance hypothesis**, not yet
an original top-tier contribution. If metrics improve, next
identify a *specific speaker-history-conditioned calibration
failure mode* not previously handled, or a validated new
mathematical reliability constraint; include 2026 prior-art
comparisons and second dataset.

## Research integrity

The previously reported CSS VALID 73.59% is from the
**custom 1037/115 dialogue split**; the paper's official TEST
numbers are not directly comparable to this validation score.
A positive or negative result on the same 115-dialogue
validation set is just exploratory. Further hyperparameter
search on this set creates selection bias. No method is
guaranteed to improve F1; improvements must be measured
with matched protocols and eventually an untouched test.

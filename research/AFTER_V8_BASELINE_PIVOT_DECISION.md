# After V8: baseline pivot decision (2026-11-17)

## Locked V8 decision

Single-seed five-arm pilot on the same previously inspected
MELD 846 TRAIN / 87 VALID dialogue partition:

| Model | All W-F1 | Changed W-F1 | Changed-after-other W-F1 |
|---|---:|---:|---:|
| Original CSS full SPF | 76.14 | 66.79 | 70.88 |
| Residual-only from scratch | **76.43** | **67.85** | **71.86** |
| Constant x4 polynomial | 76.02 | 66.82 | 70.42 |
| Current-modality-conditioned polynomial | 76.07 | 66.67 | 70.66 |
| Prior-other-speaker-conditioned polynomial | 75.96 | 66.70 | 70.67 |

Previous V7 3-seed control (residual-only minus original
CSS full SPF) was overall +0.18 pp mean, Changed +0.70 pp,
Changed-after-other +0.99 pp, on the SAME 87 VALID set.

**V8 NO-GO.** Strengthening the high-order SPF or gating
it by modality disagreement and previous different-speaker
features has no demonstrated gain; direct previous-other
gate is weaker than original overall by -0.18pp and
weaker than the current-only gate by -0.11pp overall.
Stop coefficient/architecture/seed/epoch changes on
the exhausted VALID cohort. No official TEST.

## What to research next, prioritized for ACTUAL metric improvement

The user's supervisor's target is speaker interaction:
how an intervening other speaker (e.g. B's utterance)
changes A's subsequent emotional state/recognized emotion.
The mere use of MELD does NOT mean the model isolates
inter-speaker influence. An explicit cross-speaker structural
model is different from scalar polynomial gain conditioned
on past other-speaker modality distance.

**Decision: do not write an unmotivated V9 CSS gate.**
Audit a genuinely speaker-relational **2026 top-venue
MERC baseline** with complete data / reproduction, then
perform a matched validation-selected reproduction.
Only after a sound baseline will we define a falsifiable
method and paired gains.

## Source-checked current candidate list

1. **CSS (AAAI 2026)**:
   https://ojs.aaai.org/index.php/AAAI/article/view/39602
   Strong reproducible 2026 top-conference baseline with
   the user's known code/results; its SPF modifications
   have failed to raise performance so far.
2. **EMART (ACL 2026 Long)**:
   https://aclanthology.org/2026.acl-long.1875/
   Code: https://github.com/DILAB-HYU/EMART
   Preprocessing already explicitly appends previous SELF/
   OTHER utterances and encodes audio-as-complement-to-text.
   User previously ran EMART. Training/feature protocol
   differs from CSS and must not be conflated; it is a
   credible top-conference reproducible comparator, not a
   straightforward 3-modality drop-in CSS module.
3. **CDCP** (claimed ACM MM 2026; venue publication still
   NOT independently verified):
   https://github.com/Tianba0116/CDCP
   Code currently exposes train.py/model.py/MOE.py/dataloader,
   MELD/IEMOCAP launchers, feature download and pretrained
   checkpoint link. Its README and training code currently
   **select the best checkpoint using TEST weighted-F1**,
   including `get_MELD_loaders(valid=0.0)` and a per-epoch
   `test_fscore` comparison; copying its published best
   TEST score as a fair validation-selected baseline
   would produce misleading comparisons. The README has
   EMPTY paper authors and DOI placeholders. A separate
   purported official repo
   https://github.com/MKMaS-GUET/CDCP
   says the full code will be released soon.
   A true paper/title/author/DOI verification and
   trusted-code identity check MUST precede reliance on
   this as an accepted 2026 top-venue paper.
   If verified, patch hard-coded
   `/opt/data/private/try1/data/meld_multimodal_features.pkl`
   to configurable local path, switch to whole-dialogue
   random train/VALID split seeded before training,
   select checkpoint by VALID only and preserve TEST.
   Its illustrative README MELD TEST F1=66.84 is not
   validation-selected, thus not a fair CSS comparison.
4. **SURE (ICASSP 2026)**:
   https://github.com/swaggy66/SURE
   Runnable instructions link shared features from SDT,
   but venue rank is not as clearly aligned with the
   user's required top-conference standard as AAAI/ACL;
   source may be useful for uncertainty and dialogue.
5. **DF-GCN (Knowledge-Based Systems 2026)**:
   https://www.sciencedirect.com/science/article/pii/S0950705126005721
   Code https://github.com/yuntaoshou/DFGCN.
   Dynamic fusion and graph-based speaker/context processing
   mean neither scalar gating nor dynamic cross-speaker
   attention alone can be claimed novel.

## First actionable gate for next research turn

A. Verify formal 2026 ACM MM CDCP paper details using official
proceedings/paper metadata. A repository README is not
sufficient independent evidence, especially with two
conflicting "official" repositories and blank DOI/authors.

B. If verified, create an **isolated reproducibility port**
for CDCP using VALID-only model selection and independent
tests of speaker time/order and data-availability; reproduce
both MELD/IEMOCAP with matching feature definitions.
Run the unmodified model for a fair baseline first.
Do not directly copy its TEST-best result or mix
incompatible raw features.

C. Regardless of which strong verified baseline is chosen,
compare equal splits/seeds/costs and track overall W-F1
and Changed / Changed-after-other. The user's working
target of mean +1.0 pp overall over a same-protocol
baseline is an aspiration, not a guarantee.

D. A novel hypothesis should be *structural*
(inter-speaker cause/influence vs each speaker's
own dynamics) and tested with matched no-history,
same-speaker-only, previous-other-speaker and full
dual-path controls; do not feed gold emotion transitions
or labels to inference, and keep TEST untouched.

At this stage do **not** launch a speculative V9 training
experiment or claim an actual W-F1 gain.

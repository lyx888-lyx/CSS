# CSS replication preflight (AAAI 2026)

This is a **preflight guide**, not a claim that the published scores have been reproduced. It keeps the original baseline training behavior unchanged.

## 1. Paper targets

| Dataset | Accuracy | Weighted F1 | Training entry point |
| --- | ---: | ---: | --- |
| IEMOCAP (6 emotions) | 75.42% | 75.66% | `train_IEMOCAP.py` |
| MELD (7 emotions) | 68.47% | 67.41% | `train_MELD.py` |

Source: DOI 10.1609/aaai.v40i29.39602, Tables 1–2. Metrics in the scripts use `sklearn.metrics.f1_score(..., average='weighted')` and `accuracy_score`, with padded utterances masked out.

## 2. Environment

Start in a clean Python 3.10 environment (avoid reusing CMDPAD or MF-EDM environments):

```bash
git clone https://github.com/lyx888-lyx/CSS.git
cd CSS
git checkout repro/preflight-2026
conda create -n css python=3.10 -y
conda activate css
python -m pip install -r requirements.txt
python -c "import torch, numpy, scipy, sklearn, pandas; print(torch.__version__, torch.cuda.is_available())"
```

If using a CUDA GPU, check `nvidia-smi` and install a PyTorch 2.3.0 wheel compatible with that system's driver/CUDA setup from the official PyTorch previous-versions instructions. Do not assume that the PyPI default wheel matches your CUDA environment. The model is heavy (`hidden_dim=1024`) and training can require significant GPU memory.

Notes: The upstream source imported `numba`, `pyarrow`, and `matplotlib` without listing them in requirements. The first two are unused and have been removed from `model.py`; matplotlib is still imported by both training files upstream but is unused. If they remain present in your checkout and `matplotlib` is not installed, install `matplotlib` (or use the prepared branch after the import cleanup). Optional `--tensorboard` requires `tensorboardX`.

## 3. Download assets

Feature folder (official README):
https://drive.google.com/drive/folders/1TwT9z6N6SJadsVkDhSNVBiF9ZygEyA6l

Checkpoints (official README):
https://drive.google.com/drive/folders/1tn7aR3sUIgSQ3zf7pMIMbcP2PGl3ErXm?usp=drive_link

Store files locally (not in Git):
```text
data/iemocap_multimodal_features.pkl
data/meld_multimodal_features.pkl
checkpoints/iemocap.pt            # rename or supply actual downloaded filename
checkpoints/meld.pt               # rename or supply actual downloaded filename
```

Only load pickles from trusted sources: Python pickle data can execute code during deserialization.

Run the structural integrity check:
```bash
python scripts/check_data.py --data-dir data
```

Expected shapes for each utterance: IEMOCAP text 1024 / audio 1582 / visual 342, and MELD text 1024 / audio 300 / visual 342.

## 4. Verify checkpoints first

Use the actual checkpoint filenames from the download:

```bash
python train_IEMOCAP.py --checkpoint checkpoints/iemocap.pt --eval-only
python train_MELD.py --checkpoint checkpoints/meld.pt --eval-only
```

Report accuracy, weighted F1 and class-wise classification report. Verify `--checkpoint` file contents and hyperparameter compatibility; the scripts instantiate architectures with their current CLI options before loading state dictionaries.

## 5. Train the unmodified baseline

```bash
python train_IEMOCAP.py --seed 61078 --epochs 200 --run-name iemocap_baseline
python train_MELD.py --seed 10068 --epochs 100 --run-name meld_baseline
```

Original CLI defaults: IEMOCAP learning rate 1e-4, fusion order 3 and temperature 1; MELD learning rate 5e-6, fusion order 4 and temperature 4. Both use batch size 32, hidden size 1024, rank 16, dropout 0.5. Save hardware, Python/PyTorch/CUDA versions, commit SHA, seed, best epoch, confusion matrix and complete stdout/stderr for each run.

**Important experimental caveat:** In the upstream code both entrypoints pass `valid=0.0`, evaluate on the test set every epoch, and select the best checkpoint by **test weighted F1**. This is a test-set-selected model, not independent held-out evaluation. Preserve this behavior only for faithful reproduction of the public implementation. For new research, separately implement and label a validation-selected protocol, with the test set used once at the end. Do not directly compare test-selected and validation-selected results without this caveat.

## 6. Suggested experimental sequence

1. Check installation and feature files.
2. Evaluate authors' released checkpoints as a reproducibility smoke test.
3. Run one original-seed IEMOCAP baseline, then MELD.
4. Repeat seeds and report mean/std rather than relying on one maximum.
5. Only after establishing a baseline, branch for emotion-influence modeling or grey-model experiments.

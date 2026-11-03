#!/usr/bin/env python3
"""CPU baseline: does classic GM(1,1) forecast speaker emotion probabilities?

Experimental protocol:
* Fit local-utterance logistic regression on MELD TRAIN dialogues only.
* Obtain local text-only probabilities for each VALIDATION utterance.
* At every target with >=4 PREVIOUS utterances by the same speaker,
  forecast its emotion from those *past local probability predictions*
  only. No previous true validation emotion labels are ever model inputs.
* Compare GM(1,1), level-ratio-guarded GM, last-value persistence,
  EMA(alpha=0.5) and a regularized short-window AR(1), on EXACTLY
  the same eligible validation targets.
* Only after prediction, use validation target labels to score accuracy,
  weighted-F1, multiclass Brier score and log loss.

Caveats:
- Classical GM assumes nonnegative, approximately exponential,
  regularly spaced samples. Speaker turns are only event-indexed,
  NOT necessarily equal in wall-clock time.
- This is NOT a MERC end-to-end model or evidence for emotional
  contagion; text features in a supplied dataset may themselves
  incorporate upstream context we cannot verify.
- No official TEST examples/labels are used. No GPU needed.
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataloader import MELDDataset
from scripts.train_research import dialogue_split_indices


EPS = 1e-8


def probability_normalize(values, fallback):
    """Clip scalar component forecasts and project to probability simplex."""
    p = np.asarray(values, dtype=float)
    if p.shape != np.asarray(fallback).shape:
        raise ValueError("probability vectors must share shape")
    if not np.all(np.isfinite(p)):
        return probability_normalize(fallback, np.full_like(p, 1 / len(p)))
    p = np.clip(p, EPS, 1.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(p, 1 / len(p))
    return p / total


def is_gm_level_ratio_admissible(x):
    """Classical GM level-ratio heuristic on a strictly positive series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or not np.isfinite(x).all() or (x <= 0).any():
        return False
    lower = np.exp(-2.0 / (n + 1))
    upper = np.exp(2.0 / (n + 1))
    ratios = x[:-1] / x[1:]
    return bool(((ratios > lower) & (ratios < upper)).all())


def gm11_next_scalar(x):
    """Least-squares classical discrete GM(1,1) one-step scalar forecast.

    Fits x0(k) + a*z1(k) = b, z1(k)=.5*(x1(k-1)+x1(k)),
    where x1 is the first-order accumulated series. Returns an
    unconstrained scalar (can be negative due to model misspecification).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("GM(1,1) needs >=4 observed scalars")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("classical GM input should be finite nonnegative")
    ago = np.cumsum(x)
    z = (ago[:-1] + ago[1:]) / 2
    matrix = np.column_stack([-z, np.ones(x.size - 1)])
    a, b = np.linalg.lstsq(matrix, x[1:], rcond=None)[0]
    if abs(a) < 1e-7:
        return float(b)
    # Forecast x0(n+1)=x1_hat(n+1)-x1_hat(n)
    # using the analytic solution of dx1/dt + a*x1=b.
    exponent = np.clip(-a * (x.size - 1), -60, 60)
    delta = np.expm1(np.clip(-a, -60, 60))
    return float((x[0] - b / a) * np.exp(exponent) * delta)


def ar1_forecast(history):
    """Ridge-stabilized local AR(1) intercept/slope per class."""
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 2 or len(h) < 4:
        raise ValueError("history must be >=4 probability vectors")
    old, nxt = h[:-1], h[1:]
    old_mean = old.mean(axis=0)
    nxt_mean = nxt.mean(axis=0)
    cov = ((old - old_mean) * (nxt - nxt_mean)).mean(axis=0)
    var = ((old - old_mean) ** 2).mean(axis=0)
    slope = cov / (var + 1e-3)
    pred = nxt_mean + slope * (h[-1] - old_mean)
    return probability_normalize(pred, h[-1])


def ema_forecast(history, alpha=0.5):
    h = np.asarray(history)
    estimate = h[0].astype(float).copy()
    for row in h[1:]:
        estimate = alpha * row + (1 - alpha) * estimate
    return probability_normalize(estimate, h[-1])


def gm_forecast(history, level_ratio_guard=False):
    """Classwise GM, followed by probability simplex projection.

    Guarded mode falls back to last observation for *each class*
    failing the level-ratio check; unguarded GM evaluates the raw
    classical model regardless of precondition, then clips to [0,1].
    """
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 2 or len(h) < 4:
        raise ValueError("history must be >=4 probability vectors")
    values = []
    diagnostics = {"admissible_components": 0, "components": h.shape[1],
                   "fit_fallbacks": 0, "clipped_components": 0}
    for j in range(h.shape[1]):
        seq = np.maximum(h[:, j], EPS)
        admissible = is_gm_level_ratio_admissible(seq)
        diagnostics["admissible_components"] += int(admissible)
        if level_ratio_guard and not admissible:
            values.append(float(h[-1, j]))
            diagnostics["fit_fallbacks"] += 1
            continue
        try:
            pred = gm11_next_scalar(seq)
        except (ValueError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            pred = float(h[-1, j])
            diagnostics["fit_fallbacks"] += 1
        if not np.isfinite(pred):
            pred = float(h[-1, j])
            diagnostics["fit_fallbacks"] += 1
        if pred < EPS or pred > 1:
            diagnostics["clipped_components"] += 1
        values.append(pred)
    return probability_normalize(np.asarray(values), h[-1]), diagnostics


def collect_dataset_arrays(dataset, indices):
    features, labels = [], []
    for ix in indices:
        vid = dataset.keys[int(ix)]
        x = np.asarray(dataset.videoText[vid], dtype=np.float32)
        y = np.asarray(dataset.videoLabels[vid], dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("invalid local MELD features/labels for " + str(vid))
        features.append(x)
        labels.append(y)
    return np.concatenate(features, axis=0), np.concatenate(labels)


def evaluate(y, prob, n_classes):
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("misaligned prediction arrays")
    pred = p.argmax(axis=-1)
    one_hot = np.eye(n_classes)[y]
    return {
        "n": len(y),
        "accuracy": round(100 * accuracy_score(y, pred), 2),
        "weighted_f1": round(
            100 * f1_score(y, pred, labels=list(range(n_classes)),
                           average="weighted", zero_division=0), 2
        ),
        "log_loss": round(float(log_loss(y, p, labels=list(range(n_classes)))), 6),
        "brier": round(float(np.mean(((p - one_hot) ** 2).sum(axis=1))), 6),
    }


def benchmark(dataset, train_idx, valid_idx, max_iter=180, window=4):
    if window < 4:
        raise ValueError("GM window must include at least 4 prior turns")
    X, y = collect_dataset_arrays(dataset, train_idx)
    print(f"Training text-only local logistic on {len(y)} TRAIN utterances "
          f"({X.shape[1]} features), max_iter={max_iter}...", flush=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs", max_iter=max_iter, C=0.2, random_state=10068
        ),
    )
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(
        issubclass(w.category, ConvergenceWarning) for w in warnings_seen
    )
    if not converged:
        print("WARNING: local logistic max_iter reached before convergence; "
              "interpret the forecast probe cautiously.", flush=True)
    classes = model[-1].classes_.tolist()
    if classes != list(range(7)):
        raise ValueError("expected all seven MELD classes in TRAIN")
    names = ("last", "ema", "ar1", "gm", "gm_guard", "current_text")
    predictions = {name: [] for name in names}
    true = []
    diag = defaultdict(int)
    for ix in valid_idx:
        vid = dataset.keys[int(ix)]
        x = np.asarray(dataset.videoText[vid], dtype=np.float32)
        labels = np.asarray(dataset.videoLabels[vid], dtype=np.int64)
        speakers = np.asarray(dataset.videoSpeakers[vid]).argmax(-1)
        if len(speakers) != len(labels) or len(x) != len(labels):
            raise ValueError("dialogue alignment mismatch: " + str(vid))
        prob = model.predict_proba(x)
        history = defaultdict(list)
        for t, speaker in enumerate(speakers.tolist()):
            own = history[int(speaker)]
            if len(own) >= 4:
                h = np.stack(own[-window:], axis=0)
                p_gm, gdiag = gm_forecast(h)
                p_guard, gdguard = gm_forecast(h, level_ratio_guard=True)
                for k, v in gdiag.items():
                    diag["ung_"+k] += v
                for k, v in gdguard.items():
                    diag["guard_"+k] += v
                predictions["last"].append(h[-1])
                predictions["ema"].append(ema_forecast(h))
                predictions["ar1"].append(ar1_forecast(h))
                predictions["gm"].append(p_gm)
                predictions["gm_guard"].append(p_guard)
                predictions["current_text"].append(prob[t])
                true.append(int(labels[t]))
            own.append(prob[t])
    if not true:
        raise RuntimeError("no GM-eligible validation target turns")
    scores = {
        name: evaluate(true, predictions[name], len(classes))
        for name in names
    }
    return {
        "experimental_design": (
            "Local text-only logistic fitted on TRAIN dialogues only; each "
            "VALIDATION forecast uses only earlier same-speaker local "
            "predicted probability vectors, NEVER prior true emotion labels."
        ),
        "warning": (
            "Speaker utterance indices are only an event-time approximation; "
            "classwise probability GM may violate exponential/level-ratio "
            "conditions. Current-text model uses current utterance and is "
            "not a history-only forecast. Supplied precomputed text features "
            "may embed upstream context; this was not audited."
        ),
        "local_text_model_converged": converged,
        "train_utterances": len(y),
        "validation_dialogues": len(valid_idx),
        "eligible_validation_targets": len(true),
        "window": window,
        "scores": scores,
        "gm_diagnostics": dict(diag),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path,
                   default=Path("data/meld_multimodal_features.pkl"))
    p.add_argument("--split-seed", type=int, default=10068)
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=180)
    p.add_argument("--output", type=Path,
                   default=Path("results/grey_local_forecast_benchmark.json"))
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f"features not found: {args.data}")
    ds = MELDDataset(str(args.data), train=True)
    train_idx, valid_idx = dialogue_split_indices(
        len(ds), args.validation_ratio, args.split_seed
    )
    result = benchmark(ds, train_idx, valid_idx,
                       max_iter=args.max_iter, window=args.window)
    result["split_seed"] = args.split_seed
    result["official_test_evaluated"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nMELD | CLASSICAL GM(1,1) LOCAL TEXT-PROBABILITY FORECAST | VALIDATION")
    print("Model          accuracy%    weighted_F1%     logloss     Brier")
    for name, m in result["scores"].items():
        print(f"{name:<14}{m['accuracy']:>8.2f} "
              f"{m['weighted_f1']:>14.2f} "
              f"{m['log_loss']:>11.4f} {m['brier']:>10.4f}")
    d = result["gm_diagnostics"]
    n = d.get("ung_components", 0)
    print(f"GM level-ratio admissible components: "
          f"{d.get('ung_admissible_components', 0)}/{n} "
          f"({100*d.get('ung_admissible_components', 0)/max(n,1):.2f}%)")
    print(f"GM raw clipped components: {d.get('ung_clipped_components', 0)}/{n}")
    print(f"GM eligible validation utterances: {len(result['scores']['gm']) if False else result['eligible_validation_targets']}")
    print("Warning: these are forecasts from local TEXT probability history, "
          "not CSS weighted F1, and not comparable to CSS 73.59%.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

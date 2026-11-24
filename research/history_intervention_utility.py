"""Lightweight paired history-intervention utility learning (TRAIN only).

Fit labels are the observed correctness differences between unmasked and
previous-own-key-masked predictions. At inference only model probabilities,
past predicted class and speaker/turn metadata are available. Gold emotions
are NEVER selector inputs. No test-time optimization or threshold search.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ARMS = ("baseline", "always_mask", "max_confidence", "shift_only",
        "probability_utility", "history_utility")


def feature_matrix(rows, history=False):
    """Use no gold fields, including no true previous emotion or shift label."""
    features = []
    for row in rows:
        b = np.asarray(row["baseline_prob"], dtype=float)
        m = np.asarray(row["previous_own_prob"], dtype=float)
        if any(x.shape != (6,) or not np.isfinite(x).all() or
               np.any(x < 0) or abs(x.sum() - 1) > 2e-4 for x in (b, m)):
            raise ValueError("bad probability vector")
        if int(b.argmax()) != row["baseline_pred"] or int(m.argmax()) != row["previous_own_pred"]:
            raise ValueError("argmax mismatch")
        v = np.r_[b, m, m-b]
        if history:
            prior = int(row["prior_pred"])
            lag = int(row["index"]) - int(row["previous_own_index"])
            if prior not in range(6) or lag < 1:
                raise ValueError("bad predicted prior or lag")
            onehot = np.eye(6)[prior]
            v = np.r_[v, onehot, np.log1p(lag), b[prior], m[prior],
                      float(b.argmax() == prior), float(m.argmax() == prior)]
        features.append(v)
    return np.asarray(features, dtype=np.float64)


class FixedLogistic:
    """Handle a one-class training outcome without inspecting held-out data."""
    def fit(self, x, y):
        y = np.asarray(y, dtype=int)
        if not len(y) or len(x) != len(y):
            raise ValueError("empty or misaligned selector training set")
        self.classes_ = np.unique(y)
        self.model = None
        if len(self.classes_) > 1:
            self.model = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                     random_state=61078))
            self.model.fit(x, y)
        return self

    def probability(self, x, label):
        if label not in self.classes_:
            return np.zeros(len(x))
        if self.model is None:
            return np.ones(len(x))
        idx = list(self.classes_).index(label)
        return self.model.predict_proba(x)[:, idx]


def fit_selectors(rows):
    if not rows:
        raise ValueError("no selector training examples")
    b = np.array([r["baseline_pred"] for r in rows])
    m = np.array([r["previous_own_pred"] for r in rows])
    y = np.array([r["true"] for r in rows])
    prev = np.array([r["previous_own_true"] for r in rows])
    delta = (m == y).astype(int) - (b == y).astype(int)
    disagree = b != m
    # For agreement utility is necessarily zero. If both candidates are
    # wrong on disagreement, utility is also zero; retain such examples.
    if not disagree.any():
        raise ValueError("no disagreements in training; no utility to learn")
    x = feature_matrix(rows)
    h = feature_matrix(rows, history=True)
    shift = FixedLogistic().fit(h, (y != prev).astype(int))
    prob = FixedLogistic().fit(x[disagree], delta[disagree])
    hist = FixedLogistic().fit(h[disagree], delta[disagree])
    diagnostics = {
        "n": len(rows), "disagreements": int(disagree.sum()),
        "fix": int((delta == 1).sum()), "harm": int((delta == -1).sum()),
        "neither_on_disagreement": int(((delta == 0) & disagree).sum()),
        "changed": int((y != prev).sum()),
    }
    return (shift, prob, hist), diagnostics


def predict_selectors(models, rows):
    """Pure inference: selection must be unchanged if gold fields are deleted."""
    if not rows:
        raise ValueError("empty evaluation set")
    shift, prob, hist = models
    x = feature_matrix(rows)
    h = feature_matrix(rows, history=True)
    b = np.array([r["baseline_pred"] for r in rows])
    m = np.array([r["previous_own_pred"] for r in rows])
    diff = b != m
    # Pre-fixed, zero expected net accuracy utility threshold.
    use = {
        "baseline": np.zeros(len(rows), dtype=bool),
        "always_mask": diff,
        "max_confidence": diff & (x[:,6:12].max(1) > x[:,:6].max(1)),
        "shift_only": diff & (shift.probability(h, 1) > .5),
        "probability_utility": diff & (prob.probability(x, 1) > prob.probability(x, -1)),
        "history_utility": diff & (hist.probability(h, 1) > hist.probability(h, -1)),
    }
    return {arm: np.where(mask, m, b) for arm, mask in use.items()}


def metrics(y, baseline, predictions):
    y, b, p = map(np.asarray, (y, baseline, predictions))
    if not len(y):
        return {"n": 0}
    if y.shape != b.shape or y.shape != p.shape:
        raise ValueError("evaluation vector alignment mismatch")
    fixed = int(((b != y) & (p == y)).sum())
    broken = int(((b == y) & (p != y)).sum())
    wf = lambda q: 100 * float(f1_score(y, q, average="weighted", zero_division=0))
    return {
        "n": len(y), "accuracy_pct": 100 * float((p == y).mean()),
        "accuracy_delta_pp": 100 * (fixed-broken)/len(y),
        "weighted_f1_pct": wf(p), "weighted_f1_delta_pp": wf(p)-wf(b),
        "fixed": fixed, "broken": broken, "net_more_correct": fixed-broken,
        "prediction_flips": int((b != p).sum()),
    }

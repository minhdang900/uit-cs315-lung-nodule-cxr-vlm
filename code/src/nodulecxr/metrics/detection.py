"""Image-level detection metrics, threshold selection and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def binary_metrics(y, yhat) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {"sens": _div(tp, tp + fn), "spec": _div(tn, tn + fp), "ppv": _div(tp, tp + fp),
            "npv": _div(tn, tn + fn), "f1": _div(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def threshold_at_specificity(y, s, target: float = 0.90) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    neg = s[y == 0]
    for t in np.unique(np.concatenate([s, [np.inf]])):
        if np.mean(neg < t) >= target:
            return float(t)
    return float("inf")


def predict_at(s, t: float) -> np.ndarray:
    return (np.asarray(s, dtype=float) >= t).astype(int)


def ece(y, s, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += abs(y[m].mean() - s[m].mean()) * m.sum()
    return float(total / len(y))


def score_metrics(y, s) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    both = len(np.unique(y)) == 2
    brier_pos = float(np.mean((s[y == 1] - 1) ** 2)) if (y == 1).any() else float("nan")
    brier_neg = float(np.mean(s[y == 0] ** 2)) if (y == 0).any() else float("nan")
    return {"auc_roc": float(roc_auc_score(y, s)) if both else float("nan"),
            "auc_pr": float(average_precision_score(y, s)) if both else float("nan"),
            "brier": float(np.mean((s - y) ** 2)),
            "balanced_brier": float(np.nanmean([brier_pos, brier_neg])),
            "ece": ece(y, s)}


def ppv_at_prevalence(sens: float, spec: float, prev: float) -> float:
    return _div(sens * prev, sens * prev + (1 - spec) * (1 - prev))

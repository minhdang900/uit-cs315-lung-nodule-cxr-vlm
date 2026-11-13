import math

import numpy as np
import pytest
from nodulecxr.metrics.detection import (binary_metrics, ppv_at_prevalence, predict_at,
                                         score_metrics, threshold_at_specificity)


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0, 0], [1, 0, 0, 0, 1])
    assert m["sens"] == 0.5 and m["spec"] == pytest.approx(2 / 3) and m["ppv"] == 0.5
    assert math.isnan(binary_metrics([0, 0], [0, 0])["sens"])


def test_threshold_from_validation():
    y = np.array([0] * 10 + [1, 1])
    s = np.array([i / 10 for i in range(10)] + [0.95, 0.5])
    t = threshold_at_specificity(y, s, 0.90)
    assert t == pytest.approx(0.9)
    m = binary_metrics(y, predict_at(s, t))
    assert m["spec"] == pytest.approx(0.9) and m["sens"] == 0.5


def test_score_metrics_and_ppv():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    m = score_metrics(y, s)
    assert m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0 and m["ece"] == 0.0 and m["brier"] == 0.0
    assert ppv_at_prevalence(0.6, 0.9, 0.02) == pytest.approx(0.012 / (0.012 + 0.098))

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from nodulecxr.metrics.stats import bootstrap_ci, delong_test, holm, mcnemar_p


def test_bootstrap_contains_point():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 150)
    s = np.where(y == 1, rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    point, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=300)
    assert lo < point < hi and 0.6 < point < 0.9


def test_mcnemar_identical_is_one():
    a = np.array([True, False, True, True])
    assert mcnemar_p(a, a) == 1.0


def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(1)
    y = np.array([1] * 200 + [0] * 200)
    good = np.where(y == 1, rng.normal(2, 1, 400), rng.normal(0, 1, 400))
    bad = rng.normal(0, 1, 400)
    a1, a2, p = delong_test(y, good, bad)
    assert a1 == pytest.approx(roc_auc_score(y, good)) and a2 == pytest.approx(roc_auc_score(y, bad))
    assert p < 1e-6
    assert delong_test(y, good, good)[2] == 1.0


def test_holm_monotone():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

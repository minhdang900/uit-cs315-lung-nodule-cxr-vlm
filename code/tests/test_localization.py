import pytest
from nodulecxr.metrics.localization import best_iou, center_distance, froc_points, iou, jsrt_hit


def test_iou_and_best():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert best_iou([], [[0, 0, 1, 1]]) == 0.0
    assert best_iou([[0, 0, 10, 10, 0.9]], [[0, 0, 10, 10]]) == 1.0


def test_jsrt_hit_and_distance():
    gt = [90, 90, 110, 110]               # radius 10
    assert jsrt_hit([100, 100, 120, 120], gt, margin_px=0) is False   # centre 110,110 -> dist 14.1
    assert jsrt_hit([100, 100, 120, 120], gt, margin_px=12.5) is True
    assert center_distance([0, 0, 2, 2], [3, 4, 5, 6]) == pytest.approx(5.0)


def test_froc_tp_then_fp():
    cases = [([[0, 0, 10, 10, 0.9], [50, 50, 60, 60, 0.8]], [[0, 0, 10, 10]])]
    pts = froc_points(cases, fp_rates=(0.5, 1))
    assert pts == {0.5: 1.0, 1: 1.0}
    pts2 = froc_points([([[50, 50, 60, 60, 0.9], [0, 0, 10, 10, 0.8]], [[0, 0, 10, 10]])], fp_rates=(0.5, 1))
    assert pts2 == {0.5: 0.0, 1: 1.0}

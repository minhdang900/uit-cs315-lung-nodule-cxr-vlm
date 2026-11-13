"""Localisation metrics: IoU, centre hits and FROC."""
from __future__ import annotations

import math


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(preds, gts) -> float:
    if not preds or not gts:
        return 0.0
    return max(iou(p[:4], g) for p in preds for g in gts)


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def center_distance(pred_box, gt_box) -> float:
    (px, py), (gx, gy) = _centre(pred_box), _centre(gt_box)
    return math.hypot(px - gx, py - gy)


def jsrt_hit(pred_box, gt_box, margin_px: float) -> bool:
    radius = (gt_box[2] - gt_box[0]) / 2.0
    return center_distance(pred_box, gt_box) <= radius + margin_px


def _match_image(preds, gts, thr):
    used, out = set(), []
    for p in sorted(preds, key=lambda q: -q[4]):
        best, bj = -1.0, None
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p[:4], g)
            if v >= thr and v > best:
                best, bj = v, j
        if bj is None:
            out.append((p[4], False))
        else:
            used.add(bj)
            out.append((p[4], True))
    return out


def froc_points(cases, iou_thr: float = 0.3, fp_rates=(0.125, 0.25, 0.5, 1, 2, 4, 8)) -> dict:
    n_gt = sum(len(g) for _, g in cases)
    if n_gt == 0:
        raise ValueError("FROC needs at least one ground-truth box")
    n_img = len(cases)
    scored = []
    for preds, gts in cases:
        scored += _match_image(preds, gts, iou_thr)
    scored.sort(key=lambda x: -x[0])
    tp = fp = 0
    curve = [(0.0, 0.0)]
    for i, (s, is_tp) in enumerate(scored):
        tp += int(is_tp)
        fp += int(not is_tp)
        if i + 1 == len(scored) or scored[i + 1][0] != s:
            curve.append((fp / n_img, tp / n_gt))
    return {r: max(sens for f, sens in curve if f <= r) for r in fp_rates}

"""Nodule size strata."""
from __future__ import annotations

import numpy as np

from nodulecxr.records import IMG_SIZE


def box_rel_area(box, size: int = IMG_SIZE) -> float:
    return (box[2] - box[0]) * (box[3] - box[1]) / float(size * size)


def tertile_edges(areas) -> tuple:
    q1, q2 = np.quantile(np.asarray(areas, dtype=float), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def size_bin(area: float, edges) -> str:
    if area <= edges[0]:
        return "small"
    if area <= edges[1]:
        return "medium"
    return "large"


def size_bin_mm(mm: float) -> str:
    """Fixed physical-size bins (JSRT gives nodule diameter in mm)."""
    if mm <= 10:
        return "≤10mm"
    if mm <= 20:
        return "10–20mm"
    return ">20mm"


def nodule_stratum(boxes, size_mm, edges):
    """mm bins when the row has size_mm (JSRT); otherwise largest-box area tertiles (VinBig/NIH)."""
    if size_mm is not None and size_mm == size_mm:  # not None / NaN
        return size_bin_mm(float(size_mm))
    if not len(boxes):
        return None
    return size_bin(max(box_rel_area(b) for b in boxes), edges)

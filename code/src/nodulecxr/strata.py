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

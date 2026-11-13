"""Occlusion audit utilities (T1 nodule, T2 random same-area, T3 swap)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from nodulecxr.records import IMG_SIZE


def expand_box(box, frac: float = 0.10, size: int = IMG_SIZE) -> list:
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = w * frac / 2, h * frac / 2
    return [max(0.0, box[0] - dx), max(0.0, box[1] - dy), min(float(size), box[2] + dx),
            min(float(size), box[3] + dy)]


def occlude(img: Image.Image, boxes, fill=None) -> Image.Image:
    a = np.array(img).copy()
    value = int(round(a.mean())) if fill is None else fill
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b[:4])
        a[y1:y2, x1:x2] = value
    return Image.fromarray(a)


def boxes_intersect(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def sample_random_box(w, h, lung_mask, forbidden, rng, min_inside: float = 0.9, max_tries: int = 2000):
    H, W = lung_mask.shape
    w, h = int(round(w)), int(round(h))
    if w >= W or h >= H or w < 1 or h < 1:
        return None
    for _ in range(max_tries):
        x1 = int(rng.integers(0, W - w + 1))
        y1 = int(rng.integers(0, H - h + 1))
        box = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
        if lung_mask[y1:y1 + h, x1:x1 + w].mean() >= min_inside and \
                not any(boxes_intersect(box, f) for f in forbidden):
            return box
    return None


def heuristic_lung_mask(size: int = IMG_SIZE) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    y1, y2 = int(0.15 * size), int(0.85 * size)
    m[y1:y2, int(0.12 * size):int(0.47 * size)] = True
    m[y1:y2, int(0.53 * size):int(0.88 * size)] = True
    return m


def flip_rate(orig_pos, new_pos) -> float:
    o = np.asarray(orig_pos).astype(int)
    n = np.asarray(new_pos).astype(int)
    sel = o == 1
    return float(np.mean(n[sel] == 0)) if sel.any() else float("nan")

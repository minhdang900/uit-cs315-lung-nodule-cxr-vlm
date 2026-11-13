import math

import numpy as np
from PIL import Image
from nodulecxr.audit.occlusion import (boxes_intersect, expand_box, flip_rate, heuristic_lung_mask,
                                       occlude, sample_random_box)


def test_expand_and_occlude_with_mean():
    assert expand_box([100, 100, 200, 200]) == [95.0, 95.0, 205.0, 205.0]
    arr = np.zeros((896, 896), dtype=np.uint8)
    arr[:, 448:] = 200
    out = np.array(occlude(Image.fromarray(arr).convert("RGB"), [[0, 0, 10, 10]]).convert("L"))
    assert out[5, 5] == 100 and out[500, 500] == 200


def test_random_box_inside_mask_and_disjoint():
    mask = heuristic_lung_mask()
    rng = np.random.default_rng(0)
    nod = [200, 300, 260, 360]
    b = sample_random_box(60, 60, mask, [nod], rng)
    x1, y1, x2, y2 = map(int, b)
    assert mask[y1:y2, x1:x2].mean() >= 0.9 and not boxes_intersect(b, nod)
    assert sample_random_box(900, 900, mask, [], rng) is None


def test_flip_rate():
    assert flip_rate([1, 1, 0, 1], [0, 1, 1, 0]) == 2 / 3
    assert math.isnan(flip_rate([0, 0], [1, 1]))

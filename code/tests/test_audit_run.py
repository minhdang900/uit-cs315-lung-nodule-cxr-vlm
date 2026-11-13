import numpy as np
import pandas as pd
from PIL import Image
from nodulecxr.audit.occlusion import heuristic_lung_mask
from nodulecxr.audit.run import audit_images, summarize_audit


def _img_with_blob(box):
    a = np.full((896, 896), 60, dtype=np.uint8)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 250
    return Image.fromarray(a).convert("RGB")


def test_scorer_that_looks_at_nodule_flips_only_on_t1():
    box = [200, 300, 260, 360]
    images = {"p": _img_with_blob(box), "n": _img_with_blob([0, 0, 0, 0])}
    score = lambda im: float(np.array(im.convert("L")).max() > 200)  # noqa: E731
    pos = pd.DataFrame({"image_id": ["p"], "path": ["p"], "boxes": [[box]], "subtlety": [3]})
    neg = pd.DataFrame({"image_id": ["n"], "path": ["n"]})
    df = audit_images(score, pos, neg, images.__getitem__, 0.5, lambda im: heuristic_lung_mask(),
                      edges=(0.001, 0.01))
    s = summarize_audit(df, 0.5)
    assert s["flip_t1"] == 1.0 and s["flip_t2"] == 0.0 and s["delta"] == 1.0 and s["yes_rate_t3"] == 0.0
    assert s["n_t2_ok"] == 1 and s["flip_t1_all"] == 1.0


def test_t2_box_area_equals_expanded_t1_area():
    box = [200, 300, 260, 360]
    img = _img_with_blob(box)
    fill = int(round(np.array(img).mean()))
    counts = []

    def score(im):
        counts.append(int((np.array(im) == fill).all(axis=-1).sum()))
        return 0.0

    pos = pd.DataFrame({"image_id": ["p"], "path": ["p"], "boxes": [[box]], "subtlety": [3]})
    neg = pd.DataFrame({"image_id": ["n"], "path": ["n"]})
    audit_images(score, pos, neg, lambda p: img, 0.5, lambda im: heuristic_lung_mask(), edges=(0.001, 0.01))
    orig, t1, t2, _ = counts
    assert t1 - orig > 0 and t2 - orig == t1 - orig


def test_audit_size_bin_uses_mm_when_present():
    box = [200, 300, 260, 360]
    img = _img_with_blob(box)
    pos = pd.DataFrame({"image_id": ["p", "q"], "path": ["p", "q"], "boxes": [[box], [box]], "subtlety": [3, 3],
                        "size_mm": [12.0, None]})
    neg = pd.DataFrame({"image_id": ["n"], "path": ["n"]})
    df = audit_images(lambda im: 0.0, pos, neg, lambda p: img, 0.5, lambda im: heuristic_lung_mask(),
                      edges=(0.001, 0.01))
    assert list(df.size_bin) == ["10–20mm", "medium"]

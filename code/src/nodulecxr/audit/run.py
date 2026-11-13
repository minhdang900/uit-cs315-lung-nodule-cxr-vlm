"""E5 occlusion audit over positive images."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nodulecxr.audit.occlusion import expand_box, flip_rate, occlude, sample_random_box
from nodulecxr.strata import box_rel_area, size_bin


def audit_images(score_fn, positives, negatives, load_image, threshold, lung_mask_fn, edges, seed=0):
    rng = np.random.default_rng(seed)
    neg_list = list(negatives.itertuples())
    rows = []
    for r in positives.itertuples():
        img = load_image(r.path)
        boxes = [list(b) for b in r.boxes]
        s_orig = score_fn(img)
        s_t1 = score_fn(occlude(img, [expand_box(b) for b in boxes]))
        mask = lung_mask_fn(img)
        rand = [sample_random_box(b[2] - b[0], b[3] - b[1], mask, boxes, rng) for b in boxes]
        t2_ok = all(b is not None for b in rand)
        s_t2 = score_fn(occlude(img, [b for b in rand if b is not None])) if t2_ok else float("nan")
        neg = neg_list[int(rng.integers(len(neg_list)))]
        s_t3 = score_fn(load_image(neg.path))
        area = max(box_rel_area(b) for b in boxes)
        rows.append({"image_id": r.image_id, "size_bin": size_bin(area, edges),
                     "subtlety": getattr(r, "subtlety", None), "s_orig": s_orig, "s_t1": s_t1,
                     "s_t2": s_t2, "s_t3": s_t3, "t2_ok": t2_ok})
    return pd.DataFrame(rows)


def _summary(df, thr):
    o = (df.s_orig >= thr).astype(int)
    ok = df.t2_ok.astype(bool)
    f1 = flip_rate(o, (df.s_t1 >= thr).astype(int))
    f2 = flip_rate(o[ok], (df.s_t2[ok] >= thr).astype(int))
    return {"n": int(len(df)), "n_orig_yes": int(o.sum()), "flip_t1": f1, "flip_t2": f2,
            "delta": f1 - f2, "yes_rate_t3": float((df.s_t3 >= thr).mean())}


def summarize_audit(df, threshold) -> dict:
    out = _summary(df, threshold)
    out["by_size"] = {k: _summary(g, threshold) for k, g in df.groupby("size_bin")}
    if df["subtlety"].notna().any():
        out["by_subtlety"] = {int(k): _summary(g, threshold) for k, g in df.dropna(subset=["subtlety"]).groupby("subtlety")}
    return out

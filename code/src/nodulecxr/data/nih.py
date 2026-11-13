"""NIH ChestX-ray14 external test set: Nodule/Mass boxes vs No Finding."""
from __future__ import annotations

import pandas as pd

from nodulecxr.records import scale_boxes

NIH_SIZE = 1024


def build_nih_manifest(data_entry: pd.DataFrame, bbox: pd.DataFrame, neg_ratio: int = 3,
                       seed: int = 0) -> pd.DataFrame:
    b = bbox.rename(columns={"Bbox [x": "x", "h]": "h"})
    b = b[b["Finding Label"].isin(["Nodule", "Mass"])]
    pid = data_entry.set_index("Image Index")["Patient ID"].astype(str)
    rows = []
    for img, g in b.groupby("Image Index"):
        boxes = [[r.x, r.y, r.x + r.w, r.y + r.h] for r in g.itertuples()]
        rows.append({"image_id": img, "dataset": "nih", "path": f"nih/{img}",
                     "orig_w": NIH_SIZE, "orig_h": NIH_SIZE, "label": 1,
                     "boxes": scale_boxes(boxes, NIH_SIZE, NIH_SIZE), "split": "test_nih",
                     "patient_id": pid.get(img, "")})
    pos_patients = {r["patient_id"] for r in rows}
    pool = data_entry[(data_entry["Finding Labels"] == "No Finding")
                      & (~data_entry["Patient ID"].astype(str).isin(pos_patients))]
    neg = pool.sample(n=min(len(pool), neg_ratio * len(rows)), random_state=seed)
    for img, p in zip(neg["Image Index"], neg["Patient ID"].astype(str)):
        rows.append({"image_id": img, "dataset": "nih", "path": f"nih/{img}",
                     "orig_w": NIH_SIZE, "orig_h": NIH_SIZE, "label": 0, "boxes": [],
                     "split": "test_nih", "patient_id": p})
    return pd.DataFrame(rows)

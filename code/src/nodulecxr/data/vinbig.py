"""VinBigData / VinDr-CXR annotations -> one row per image, Nodule/Mass positives."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

NODULE_CLASS = "Nodule/Mass"


def build_vinbig_labels(ann: pd.DataFrame, dims: pd.DataFrame, iou_thr: float = 0.4) -> pd.DataFrame:
    ann = ann.rename(columns={"rad_ID": "rad_id"})
    dims = dims.set_index("image_id")
    rows = []
    for image_id, g in ann.groupby("image_id"):
        w = float(dims.loc[image_id, "orig_w"])
        h = float(dims.loc[image_id, "orig_h"])
        nod = g[g["class_name"] == NODULE_CLASS]
        if nod.empty:
            rows.append({"image_id": image_id, "orig_w": w, "orig_h": h, "label": 0, "boxes": [], "n_rads": 0})
            continue
        boxes_list, scores_list, labels_list = [], [], []
        for _, r in nod.groupby("rad_id"):
            b = np.stack([r.x_min / w, r.y_min / h, r.x_max / w, r.y_max / h], axis=1).clip(0, 1)
            boxes_list.append(b.tolist())
            scores_list.append([1.0] * len(b))
            labels_list.append([0] * len(b))
        fused, _, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                            iou_thr=iou_thr, skip_box_thr=0.0)
        boxes = [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in fused]
        rows.append({"image_id": image_id, "orig_w": w, "orig_h": h, "label": 1, "boxes": boxes,
                     "n_rads": int(nod["rad_id"].nunique())})
    return pd.DataFrame(rows)

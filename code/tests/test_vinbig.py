import numpy as np
import pandas as pd
from nodulecxr.data.vinbig import build_vinbig_labels


def test_wbf_fuses_two_radiologists_and_keeps_negatives():
    nan = np.nan
    ann = pd.DataFrame([
        {"image_id": "p1", "class_name": "Nodule/Mass", "rad_id": "R1", "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
        {"image_id": "p1", "class_name": "Nodule/Mass", "rad_id": "R2", "x_min": 120, "y_min": 120, "x_max": 220, "y_max": 220},
        {"image_id": "p1", "class_name": "Aortic enlargement", "rad_id": "R3", "x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50},
        {"image_id": "n1", "class_name": "No finding", "rad_id": "R1", "x_min": nan, "y_min": nan, "x_max": nan, "y_max": nan},
    ])
    dims = pd.DataFrame({"image_id": ["p1", "n1"], "orig_w": [1000, 1000], "orig_h": [1000, 1000]})
    out = build_vinbig_labels(ann, dims).set_index("image_id")
    assert out.loc["p1", "label"] == 1 and out.loc["p1", "n_rads"] == 2
    boxes = out.loc["p1", "boxes"]
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [110, 110, 210, 210], atol=1.0)
    assert out.loc["n1", "label"] == 0 and out.loc["n1", "boxes"] == []


def test_accepts_physionet_rad_ID_column():
    ann = pd.DataFrame([{"image_id": "p1", "class_name": "Nodule/Mass", "rad_ID": "R8",
                         "x_min": 10, "y_min": 10, "x_max": 30, "y_max": 30}])
    dims = pd.DataFrame({"image_id": ["p1"], "orig_w": [100], "orig_h": [100]})
    out = build_vinbig_labels(ann, dims)
    assert out.loc[0, "label"] == 1

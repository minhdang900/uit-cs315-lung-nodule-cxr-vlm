import pandas as pd
from nodulecxr.data.nih import build_nih_manifest


def test_nih_positives_and_patient_disjoint_negatives():
    bbox = pd.DataFrame({"Image Index": ["a.png", "a.png", "b.png", "c.png"],
                         "Finding Label": ["Nodule", "Mass", "Nodule", "Atelectasis"],
                         "Bbox [x": [10, 100, 20, 0], "y": [10, 100, 20, 0],
                         "w": [30, 50, 40, 10], "h]": [30, 50, 40, 10]})
    entries = pd.DataFrame({"Image Index": ["a.png", "b.png", "c.png"] + [f"n{i}.png" for i in range(10)],
                            "Finding Labels": ["Nodule|Mass", "Nodule", "Atelectasis"] + ["No Finding"] * 10,
                            "Patient ID": [1, 2, 3] + [1] + list(range(100, 109))})
    m = build_nih_manifest(entries, bbox, neg_ratio=3, seed=0)
    pos = m[m.label == 1].set_index("image_id")
    assert set(pos.index) == {"a.png", "b.png"} and len(pos.loc["a.png", "boxes"]) == 2
    assert pos.loc["b.png", "boxes"][0] == [20 * 0.875, 20 * 0.875, 60 * 0.875, 60 * 0.875]
    neg = m[m.label == 0]
    assert len(neg) == 6 and "n0.png" not in set(neg.image_id)

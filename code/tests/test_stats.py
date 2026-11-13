import pandas as pd
from nodulecxr.data.stats import describe_manifest


def test_describe_manifest_counts():
    df = pd.DataFrame({"split": ["train", "train", "val", "test"], "label": [1, 0, 0, 1],
                       "boxes": [[[0, 0, 10, 10]], [], [], [[0, 0, 300, 300]]],
                       "n_rads": [2, 0, 0, 1]})
    d = describe_manifest(df, edges=(0.001, 0.01))
    tr = [r for r in d["per_split"] if r["split"] == "train"][0]
    assert tr["n"] == 2 and tr["n_pos"] == 1 and tr["prevalence"] == 0.5
    assert d["n_rads"] == {1: 1, 2: 1}
    assert d["size_bins"] == {"small": 1, "large": 1}

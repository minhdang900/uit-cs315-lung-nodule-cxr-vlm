import pandas as pd
from nodulecxr.records import (MANIFEST_COLUMNS, append_jsonl, done_ids, read_jsonl,
                               read_manifest, scale_boxes, write_manifest)


def test_scale_boxes_maps_original_to_896():
    assert scale_boxes([[0, 0, 1792, 896]], 1792, 896) == [[0.0, 0.0, 896.0, 896.0]]


def test_manifest_roundtrip(tmp_path):
    df = pd.DataFrame([{"image_id": "a", "dataset": "vinbig", "path": "vinbig/a.png",
                        "orig_w": 100, "orig_h": 100, "label": 1, "boxes": [[1, 2, 3, 4]],
                        "split": "train"},
                       {"image_id": "b", "dataset": "vinbig", "path": "vinbig/b.png",
                        "orig_w": 100, "orig_h": 100, "label": 0, "boxes": [], "split": "val"}])
    p = tmp_path / "m.csv"
    write_manifest(df, p)
    back = read_manifest(p)
    assert list(back.columns) == MANIFEST_COLUMNS
    assert back.loc[0, "boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert back.loc[1, "boxes"] == []
    assert back["label"].tolist() == [1, 0]


def test_jsonl_append_and_resume(tmp_path):
    p = tmp_path / "p.jsonl"
    append_jsonl({"image_id": "a", "config": "C0", "run": 0, "score": 0.2}, p)
    append_jsonl({"image_id": "b", "config": "C0", "run": 1, "score": 0.9}, p)
    df = read_jsonl(p)
    assert df["image_id"].tolist() == ["a", "b"]
    assert done_ids(p, config="C0", run=0) == {"a"}
    assert done_ids(tmp_path / "missing.jsonl") == set()

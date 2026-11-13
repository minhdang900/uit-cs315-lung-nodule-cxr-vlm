import pandas as pd
from PIL import Image
from nodulecxr.baselines.yolo import export_yolo, yolo_label_lines


def test_label_lines():
    assert yolo_label_lines([[0, 0, 448, 224]]) == ["0 0.250000 0.125000 0.500000 0.250000"]


def test_export(tmp_path):
    root = tmp_path / "data"
    (root / "vinbig").mkdir(parents=True)
    for i in ("a", "b"):
        Image.new("L", (896, 896)).save(root / "vinbig" / f"{i}.png")
    tr = pd.DataFrame({"image_id": ["a"], "path": ["vinbig/a.png"], "label": [1], "boxes": [[[0, 0, 448, 224]]]})
    va = pd.DataFrame({"image_id": ["b"], "path": ["vinbig/b.png"], "label": [0], "boxes": [[]]})
    yaml_path = export_yolo(tr, va, root, tmp_path / "yolo")
    assert (tmp_path / "yolo/labels/train/a.txt").read_text().strip() == "0 0.250000 0.125000 0.500000 0.250000"
    assert (tmp_path / "yolo/labels/val/b.txt").read_text() == ""
    assert "names" in yaml_path.read_text()

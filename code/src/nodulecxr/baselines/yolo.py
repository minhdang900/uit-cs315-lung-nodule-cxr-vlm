"""B2: YOLOv8 export, scoring helper."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from nodulecxr.records import IMG_SIZE


def yolo_label_lines(boxes, size: int = IMG_SIZE) -> list:
    lines = []
    for x1, y1, x2, y2 in (b[:4] for b in boxes):
        lines.append(f"0 {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                     f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}")
    return lines


def _link(src: Path, dst: Path):
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def export_yolo(df_train, df_val, data_root, out_dir) -> Path:
    out = Path(out_dir)
    for split, df in (("train", df_train), ("val", df_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in df.itertuples():
            dst = out / "images" / split / f"{r.image_id}.png"
            if not dst.exists():
                _link(Path(data_root) / r.path, dst)
            lines = yolo_label_lines(r.boxes) if int(r.label) == 1 else []
            (out / "labels" / split / f"{r.image_id}.txt").write_text("\n".join(lines))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: nodule\n")
    return yaml_path


def yolo_scorer(model, imgsz: int = IMG_SIZE):
    def score(pil):
        res = model.predict(pil, imgsz=imgsz, conf=0.001, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy().tolist()
        conf = res.boxes.conf.cpu().numpy().tolist()
        boxes = [b + [c] for b, c in zip(xyxy, conf)]
        return (max(conf) if conf else 0.0), boxes
    return score

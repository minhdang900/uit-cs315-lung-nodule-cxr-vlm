"""Train YOLOv8 on VinBigData Nodule/Mass boxes and predict all evaluation splits -> JSONL."""
import argparse

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from nodulecxr.baselines.yolo import export_yolo, yolo_scorer
from nodulecxr.data.split import subsample_negatives
from nodulecxr.images import load_image
from nodulecxr.records import append_jsonl, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([read_manifest(m) for m in a.manifests])
    tr = subsample_negatives(df[df.split == "train"], a.neg_per_pos, a.seed)
    yaml_path = export_yolo(tr, df[df.split == "val"], a.data_root, f"{a.work}/yolo_data")
    model = YOLO(a.weights)
    model.train(data=str(yaml_path), imgsz=896, epochs=a.epochs, batch=16, seed=a.seed,
                project=a.work, name=f"seed{a.seed}", exist_ok=True)
    model = YOLO(f"{a.work}/seed{a.seed}/weights/best.pt")  # best epoch, not last
    score = yolo_scorer(model)
    ev = df[df.split != "train"]
    for r in tqdm(ev.itertuples(), total=len(ev)):
        s, boxes = score(load_image(a.data_root, r.path))
        append_jsonl({"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": float(s),
                      "label_pred": None, "boxes_pred": boxes, "parse_ok": True, "raw_text": "",
                      "config": "B2", "seed": a.seed, "run": 0}, a.out)


if __name__ == "__main__":
    main()

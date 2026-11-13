"""Build manifest_nih.csv and 896px PNGs from NIH ChestX-ray14."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from nodulecxr.data.nih import build_nih_manifest
from nodulecxr.images import resize_square
from nodulecxr.records import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--neg-ratio", type=int, default=3)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    man = build_nih_manifest(pd.read_csv(raw / "Data_Entry_2017.csv"),
                             pd.read_csv(raw / "BBox_List_2017.csv"), a.neg_ratio, seed=0)
    index = {p.name: p for p in raw.rglob("*.png")}
    out = Path(a.out_root) / "nih"
    out.mkdir(parents=True, exist_ok=True)
    for img in man["image_id"]:
        resize_square(Image.open(index[img])).save(out / img)
    write_manifest(man, Path(a.out_root) / "manifest_nih.csv")
    print(man["label"].value_counts())


if __name__ == "__main__":
    main()

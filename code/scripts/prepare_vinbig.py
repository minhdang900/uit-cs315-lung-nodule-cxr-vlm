"""Build manifest_vinbig.csv and 896px PNGs from VinBigData."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from nodulecxr.data.split import stratified_split
from nodulecxr.data.vinbig import build_vinbig_labels
from nodulecxr.images import dicom_to_uint8, resize_square
from nodulecxr.records import scale_boxes, write_manifest


def dims_from_dicom(dicom_dir, ids):
    import pydicom
    rows = []
    for i in ids:
        ds = pydicom.dcmread(Path(dicom_dir) / f"{i}.dicom", stop_before_pixels=True)
        rows.append({"image_id": i, "orig_w": int(ds.Columns), "orig_h": int(ds.Rows)})
    return pd.DataFrame(rows)


def convert(labels, src_png_dir, src_dicom_dir, out_root, split_name=None):
    out_dir = Path(out_root) / "vinbig"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in tqdm(labels.itertuples(), total=len(labels)):
        dst = out_dir / f"{r.image_id}.png"
        if not dst.exists():
            if src_png_dir:
                img = Image.open(Path(src_png_dir) / f"{r.image_id}.png")
            else:
                img = dicom_to_uint8(Path(src_dicom_dir) / f"{r.image_id}.dicom")
            resize_square(img).save(dst)
        recs.append({"image_id": r.image_id, "dataset": "vinbig", "path": f"vinbig/{r.image_id}.png",
                     "orig_w": r.orig_w, "orig_h": r.orig_h, "label": r.label,
                     "boxes": scale_boxes(r.boxes, r.orig_w, r.orig_h), "n_rads": r.n_rads,
                     "split": split_name})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--png-dir")
    ap.add_argument("--meta-csv")
    ap.add_argument("--h-col", default="dim0")
    ap.add_argument("--w-col", default="dim1")
    ap.add_argument("--dicom-dir")
    ap.add_argument("--official-test-csv")
    ap.add_argument("--official-test-dicom")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ann = pd.read_csv(a.train_csv)
    if a.meta_csv:
        dims = pd.read_csv(a.meta_csv).rename(columns={a.h_col: "orig_h", a.w_col: "orig_w"})
    else:
        dims = dims_from_dicom(a.dicom_dir, ann["image_id"].unique())
    labels = build_vinbig_labels(ann, dims[["image_id", "orig_w", "orig_h"]])
    man = convert(labels, a.png_dir, a.dicom_dir, a.out_root)
    ratios = (0.85, 0.15, 0.0) if a.official_test_csv else (0.7, 0.1, 0.2)
    man["split"] = stratified_split(man.set_index("image_id")["label"], ratios, a.seed).to_numpy()
    if a.official_test_csv:
        t_ann = pd.read_csv(a.official_test_csv)
        t_dims = dims_from_dicom(a.official_test_dicom, t_ann["image_id"].unique())
        t_labels = build_vinbig_labels(t_ann, t_dims)
        man = pd.concat([man, convert(t_labels, None, a.official_test_dicom, a.out_root, "test")])
    write_manifest(man, Path(a.out_root) / "manifest_vinbig.csv")
    print(man.groupby("split")["label"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()

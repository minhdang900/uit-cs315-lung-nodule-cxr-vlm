"""Build manifest_jsrt.csv and 896px PNGs from JSRT (.IMG release or the current DICOM release)."""
import argparse
from pathlib import Path

from nodulecxr.data.jsrt import (CLN_NAMES, NN_NAMES, build_jsrt_manifest, find_clinical_file,
                                 find_jsrt_image, parse_clndat, parse_nndat)
from nodulecxr.images import dicom_to_uint8, read_jsrt_img, resize_square
from nodulecxr.records import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    cln = parse_clndat(find_clinical_file(raw, CLN_NAMES).read_text(encoding="latin-1"))
    nn = parse_nndat(find_clinical_file(raw, NN_NAMES).read_text(encoding="latin-1"))
    man = build_jsrt_manifest(cln, nn)
    out = Path(a.out_root) / "jsrt"
    out.mkdir(parents=True, exist_ok=True)
    for stem in man["image_id"]:
        src = find_jsrt_image(raw, stem)
        img = read_jsrt_img(src) if src.suffix.lower() == ".img" else dicom_to_uint8(src)
        resize_square(img).save(out / f"{stem}.png")
    write_manifest(man, Path(a.out_root) / "manifest_jsrt.csv")
    print(man["label"].value_counts())


if __name__ == "__main__":
    main()

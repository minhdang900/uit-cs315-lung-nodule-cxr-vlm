"""Build manifest_jsrt.csv and 896px PNGs from JSRT .IMG files."""
import argparse
from pathlib import Path

from nodulecxr.data.jsrt import build_jsrt_manifest, parse_clndat, parse_nndat
from nodulecxr.images import read_jsrt_img, resize_square
from nodulecxr.records import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    cln = parse_clndat((raw / "CLNDAT_EN.txt").read_text(encoding="latin-1"))
    nn = parse_nndat((raw / "NNDAT_EN.txt").read_text(encoding="latin-1"))
    man = build_jsrt_manifest(cln, nn)
    out = Path(a.out_root) / "jsrt"
    out.mkdir(parents=True, exist_ok=True)
    for stem in man["image_id"]:
        src = next(raw.rglob(f"{stem}.IMG"), None) or next(raw.rglob(f"{stem.lower()}.img"))
        resize_square(read_jsrt_img(src)).save(out / f"{stem}.png")
    write_manifest(man, Path(a.out_root) / "manifest_jsrt.csv")
    print(man["label"].value_counts())


if __name__ == "__main__":
    main()

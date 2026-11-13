"""QA: draw ground-truth boxes on n positive images per manifest."""
import argparse
from pathlib import Path

from PIL import ImageDraw

from nodulecxr.images import load_image
from nodulecxr.records import read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for r in df[df.label == 1].head(a.n).itertuples():
        im = load_image(a.data_root, r.path)
        d = ImageDraw.Draw(im)
        for b in r.boxes:
            d.rectangle(b, outline=(255, 0, 0), width=3)
        im.save(Path(a.out) / f"{r.dataset}_{r.image_id}.png")


if __name__ == "__main__":
    main()

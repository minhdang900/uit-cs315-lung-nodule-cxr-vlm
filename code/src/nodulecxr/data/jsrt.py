"""JSRT nodule database: metadata parsing and manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nodulecxr.records import IMG_SIZE, scale_boxes

JSRT_SIZE_PX = 2048
JSRT_PIXEL_MM = 0.175
JSRT_MM_PER_PX_896 = JSRT_PIXEL_MM * JSRT_SIZE_PX / IMG_SIZE

# Clinical-info file names differ between JSRT releases (matched case-insensitively).
CLN_NAMES = ("CLNDAT_EN.txt", "CLNDAT_E.TXT")
NN_NAMES = ("NNDAT_EN.txt", "CNNDAT_EN.TXT", "CNNDAT_E.TXT")


def find_clinical_file(raw, names) -> Path:
    """First file under `raw` (recursive) whose name matches one of `names`, ignoring case."""
    wanted = [n.lower() for n in names]
    hits = {p.name.lower(): p for p in Path(raw).rglob("*") if p.is_file() and p.name.lower() in wanted}
    for n in wanted:
        if n in hits:
            return hits[n]
    raise FileNotFoundError(f"none of {list(names)} found under {raw}")


def find_jsrt_image(raw, stem: str) -> Path:
    """Image file for `stem`: the original 12-bit .IMG if present, else the DICOM release (.dcm)."""
    by_ext: dict = {}
    for p in Path(raw).rglob("*"):
        if p.is_file() and p.stem.upper() == stem.upper():
            by_ext.setdefault(p.suffix.lower(), p)
    for ext in (".img", ".dcm"):
        if ext in by_ext:
            return by_ext[ext]
    raise FileNotFoundError(f"no .IMG or .dcm for {stem} under {raw}")


def parse_clndat(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 8 or not p[0].upper().startswith("JPCLN"):
            continue
        rows.append({"filename": p[0], "subtlety": int(p[1]), "size_mm": float(p[2]),
                     "x": float(p[5]), "y": float(p[6]), "malignancy": p[7].lower()})
    return pd.DataFrame(rows)


def parse_nndat(text: str) -> pd.DataFrame:
    names = [line.split()[0] for line in text.splitlines() if line.strip().upper().startswith("JPCNN")]
    return pd.DataFrame({"filename": names})


def build_jsrt_manifest(cln: pd.DataFrame, nn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cln.itertuples():
        half = r.size_mm / JSRT_PIXEL_MM / 2
        box = [r.x - half, r.y - half, r.x + half, r.y + half]
        stem = Path(r.filename).stem.upper()
        rows.append({"image_id": stem, "dataset": "jsrt", "path": f"jsrt/{stem}.png",
                     "orig_w": JSRT_SIZE_PX, "orig_h": JSRT_SIZE_PX, "label": 1,
                     "boxes": scale_boxes([box], JSRT_SIZE_PX, JSRT_SIZE_PX), "split": "test_jsrt",
                     "subtlety": r.subtlety, "size_mm": r.size_mm})
    for f in nn["filename"]:
        stem = Path(f).stem.upper()
        rows.append({"image_id": stem, "dataset": "jsrt", "path": f"jsrt/{stem}.png",
                     "orig_w": JSRT_SIZE_PX, "orig_h": JSRT_SIZE_PX, "label": 0, "boxes": [],
                     "split": "test_jsrt"})
    return pd.DataFrame(rows)

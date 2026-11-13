"""Manifest format and prediction I/O shared by every dataset, model and metric."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

IMG_SIZE = 896
MANIFEST_COLUMNS = ["image_id", "dataset", "path", "orig_w", "orig_h", "label",
                    "boxes", "split", "subtlety", "size_mm", "patient_id", "n_rads"]


def boxes_to_json(boxes) -> str:
    if not isinstance(boxes, (list, tuple)):
        return "[]"
    return json.dumps([[round(float(v), 1) for v in b[:4]] for b in boxes])


def boxes_from_json(s) -> list:
    if not isinstance(s, str) or not s.strip():
        return []
    return [[float(v) for v in b[:4]] for b in json.loads(s)]


def scale_boxes(boxes, orig_w, orig_h, size: int = IMG_SIZE) -> list:
    sx, sy = size / float(orig_w), size / float(orig_h)
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def write_manifest(df: pd.DataFrame, path) -> None:
    out = df.copy()
    for c in MANIFEST_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out["boxes"] = out["boxes"].apply(lambda b: b if isinstance(b, str) else boxes_to_json(b))
    out[MANIFEST_COLUMNS].to_csv(path, index=False)


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    df["boxes"] = df["boxes"].fillna("[]").apply(boxes_from_json)
    df["label"] = df["label"].astype(int)
    return df


def append_jsonl(row: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
    return df


def done_ids(path, **match) -> set:
    if not Path(path).exists():
        return set()
    df = read_jsonl(path)
    if df.empty:
        return set()
    for k, v in match.items():
        df = df[df[k] == v]
    return set(df["image_id"])

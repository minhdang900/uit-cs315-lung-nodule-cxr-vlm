"""Image reading and normalisation to 896x896 grayscale PNG."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nodulecxr.records import IMG_SIZE


def percentile_to_uint8(arr, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - p_lo) / (p_hi - p_lo), 0, 1) * 255).round().astype(np.uint8)


def read_jsrt_img(path, shape=(2048, 2048), invert: bool = True) -> np.ndarray:
    raw = np.fromfile(path, dtype=">u2")
    if raw.size != shape[0] * shape[1]:
        raise ValueError(f"{path}: expected {shape[0] * shape[1]} pixels, got {raw.size}")
    img = percentile_to_uint8(raw.reshape(shape))
    return 255 - img if invert else img


def dicom_to_uint8(path) -> np.ndarray:
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return percentile_to_uint8(arr, 0.0, 100.0)


def resize_square(img, size: int = IMG_SIZE) -> Image.Image:
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    return im.convert("L").resize((size, size), Image.BILINEAR)


def load_image(root, rel) -> Image.Image:
    return Image.open(Path(root) / rel).convert("RGB")

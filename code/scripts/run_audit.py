"""E5: occlusion audit for a VLM config (C0 or best LoRA) or B2 on one test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nodulecxr.audit.occlusion import heuristic_lung_mask
from nodulecxr.audit.run import audit_images, summarize_audit
from nodulecxr.images import load_image
from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC
from nodulecxr.records import read_manifest

_SEG = {}


def lung_mask_xrv(img):
    import torch
    import torchxrayvision as xrv
    from PIL import Image
    from nodulecxr.baselines.densenet import xrv_normalize
    if "m" not in _SEG:
        _SEG["m"] = xrv.baseline_models.chestx_det.PSPNet().eval()
    seg = _SEG["m"]
    x = xrv_normalize(np.array(img.convert("L").resize((512, 512))))[None, None]
    with torch.no_grad():
        out = torch.sigmoid(seg(torch.from_numpy(x)))[0]
    idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    m = (out[idx] > 0.5).any(0).numpy().astype(np.uint8) * 255
    mask = np.array(Image.fromarray(m).resize((896, 896), Image.NEAREST)) > 0
    return mask if mask.mean() > 0.05 else heuristic_lung_mask()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["vlm", "yolo"], required=True)
    ap.add_argument("--model", choices=["medgemma", "qwen"])
    ap.add_argument("--adapter")
    ap.add_argument("--prompt", choices=["loc", "cls"], default="loc")
    ap.add_argument("--yolo-weights")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="from outputs/report/thresholds.json")
    ap.add_argument("--edges-json", required=True, help="outputs/e0_data_edges.json")
    ap.add_argument("--lung-mask", choices=["xrv", "heuristic"], default="xrv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    df = df[df.split == a.split]
    edges = tuple(pd.read_json(a.edges_json, typ="series").tolist())
    if a.scorer == "vlm":
        from nodulecxr.vlm.hf import HFVLM
        vlm = HFVLM(a.model, a.adapter)
        prompt = QUESTION_CLS if a.prompt == "cls" else QUESTION_LOC
        score_fn = lambda im: vlm.yes_probability(im, prompt)  # noqa: E731
    else:
        from ultralytics import YOLO
        from nodulecxr.baselines.yolo import yolo_scorer
        ys = yolo_scorer(YOLO(a.yolo_weights))
        score_fn = lambda im: ys(im)[0]  # noqa: E731
    mask_fn = lung_mask_xrv if a.lung_mask == "xrv" else (lambda im: heuristic_lung_mask())
    res = audit_images(score_fn, df[df.label == 1], df[df.label == 0],
                       lambda p: load_image(a.data_root, p), a.threshold, mask_fn, edges)
    res.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    with open(a.out, "w") as f:
        json.dump(summarize_audit(res, a.threshold), f, indent=2, default=float)


if __name__ == "__main__":
    main()

"""B1: torchxrayvision DenseNet-121 backbone with a binary nodule/mass head."""
from __future__ import annotations

import numpy as np
from PIL import Image


def xrv_normalize(arr) -> np.ndarray:
    return (np.asarray(arr, dtype=np.float32) / 255.0) * 2048.0 - 1024.0


def to_xrv_array(pil: Image.Image, size: int = 224) -> np.ndarray:
    a = np.array(pil.convert("L").resize((size, size), Image.BILINEAR))
    return xrv_normalize(a)[None]


def build_model():
    import torch.nn as nn
    import torchxrayvision as xrv

    class NoduleDenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.head = nn.Linear(1024, 1)

        def forward(self, x):
            return self.head(self.backbone.features2(x)).squeeze(1)

    return NoduleDenseNet()

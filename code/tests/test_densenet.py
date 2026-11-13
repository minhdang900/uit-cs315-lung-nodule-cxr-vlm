import numpy as np
from PIL import Image
from nodulecxr.baselines.densenet import to_xrv_array, xrv_normalize


def test_xrv_normalize_range_and_shape():
    out = xrv_normalize(np.array([0, 255], dtype=np.uint8))
    assert out.tolist() == [-1024.0, 1024.0]
    arr = to_xrv_array(Image.new("RGB", (896, 896), (128, 128, 128)))
    assert arr.shape == (1, 224, 224) and arr.dtype == np.float32

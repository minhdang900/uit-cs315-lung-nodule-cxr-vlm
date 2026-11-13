import numpy as np
from PIL import Image
from nodulecxr.images import load_image, percentile_to_uint8, read_jsrt_img, resize_square


def test_percentile_stretch_hits_extremes():
    out = percentile_to_uint8(np.arange(100, dtype=np.float32).reshape(10, 10), 0, 100)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_constant_image_does_not_divide_by_zero():
    assert percentile_to_uint8(np.ones((4, 4))).max() == 0


def test_read_jsrt_big_endian_and_invert(tmp_path):
    raw = np.arange(16, dtype=">u2").reshape(4, 4)
    p = tmp_path / "x.img"
    raw.tofile(p)
    img = read_jsrt_img(p, shape=(4, 4), invert=True)
    assert img[0, 0] == 255 and img[-1, -1] == 0


def test_resize_and_load(tmp_path):
    im = resize_square(np.zeros((50, 80), dtype=np.uint8))
    assert im.size == (896, 896) and im.mode == "L"
    im.save(tmp_path / "a.png")
    assert load_image(tmp_path, "a.png").mode == "RGB"

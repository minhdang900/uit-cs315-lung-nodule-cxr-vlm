import pytest
from nodulecxr.data.jsrt import build_jsrt_manifest, parse_clndat, parse_nndat

CLN = "JPCLN001.IMG\t3\t15\t59\tfemale\t1634\t692\tmalignant\tl.upper lobe\tlung cancer\n"
NN = "JPCNN001.IMG\t60\tmale\tnon-nodule\n"


def test_parse_and_build():
    cln, nn = parse_clndat(CLN), parse_nndat(NN)
    assert cln.loc[0, "subtlety"] == 3 and cln.loc[0, "size_mm"] == 15.0
    m = build_jsrt_manifest(cln, nn).set_index("image_id")
    pos = m.loc["JPCLN001"]
    x1, y1, x2, y2 = pos["boxes"][0]
    assert (x1 + x2) / 2 == pytest.approx(1634 * 896 / 2048)
    assert (x2 - x1) / 2 == pytest.approx(15 / 0.175 / 2 * 896 / 2048)
    assert pos["label"] == 1 and pos["split"] == "test_jsrt" and pos["path"] == "jsrt/JPCLN001.png"
    assert m.loc["JPCNN001", "label"] == 0

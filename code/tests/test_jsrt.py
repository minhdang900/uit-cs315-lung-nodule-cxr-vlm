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


def test_find_clinical_file_accepts_current_release_names(tmp_path):
    from nodulecxr.data.jsrt import NN_NAMES, find_clinical_file
    (tmp_path / "CNNDAT_EN.TXT").write_text("JPCNN001.IMG 41 male normal\r")
    assert find_clinical_file(tmp_path, NN_NAMES).name == "CNNDAT_EN.TXT"


def test_find_clinical_file_missing_raises(tmp_path):
    import pytest
    from nodulecxr.data.jsrt import CLN_NAMES, find_clinical_file
    with pytest.raises(FileNotFoundError):
        find_clinical_file(tmp_path, CLN_NAMES)


def test_find_jsrt_image_prefers_img_then_dicom(tmp_path):
    from nodulecxr.data.jsrt import find_jsrt_image
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "JPCLN001.dcm").write_bytes(b"x")
    assert find_jsrt_image(tmp_path, "JPCLN001").suffix == ".dcm"
    (tmp_path / "JPCLN001.IMG").write_bytes(b"x")
    assert find_jsrt_image(tmp_path, "JPCLN001").suffix == ".IMG"


def test_parse_clndat_handles_cr_line_endings():
    text = "JPCLN001.IMG\t5\t15\t53\tmale\t1634\t692\tmalignant\tl.upper lobe\tlung cancer\rJPCLN002.IMG\t5\t30\t78\tfemale\t1614\t1090\tbenign\tx\ty\r"
    df = parse_clndat(text)
    assert list(df["filename"]) == ["JPCLN001.IMG", "JPCLN002.IMG"] and df.loc[0, "x"] == 1634

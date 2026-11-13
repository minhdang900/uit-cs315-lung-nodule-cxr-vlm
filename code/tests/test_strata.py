from nodulecxr.strata import box_rel_area, size_bin, tertile_edges


def test_tertiles():
    edges = tertile_edges([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    assert size_bin(0.0015, edges) == "small" and size_bin(0.0055, edges) == "large"
    assert box_rel_area([0, 0, 448, 448]) == 0.25


def test_mm_bins_and_stratum():
    from nodulecxr.strata import nodule_stratum, size_bin_mm
    assert [size_bin_mm(v) for v in (5, 10, 10.5, 20, 21)] == ["≤10mm", "≤10mm", "10–20mm", "10–20mm", ">20mm"]
    edges = (0.001, 0.01)
    assert nodule_stratum([[100, 100, 150, 150]], 25.0, edges) == ">20mm"
    assert nodule_stratum([[100, 100, 150, 150]], None, edges) == "medium"
    assert nodule_stratum([[100, 100, 150, 150]], float("nan"), edges) == "medium"
    assert nodule_stratum([], None, edges) is None

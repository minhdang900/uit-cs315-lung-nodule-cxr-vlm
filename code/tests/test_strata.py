from nodulecxr.strata import box_rel_area, size_bin, tertile_edges


def test_tertiles():
    edges = tertile_edges([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    assert size_bin(0.0015, edges) == "small" and size_bin(0.0055, edges) == "large"
    assert box_rel_area([0, 0, 448, 448]) == 0.25

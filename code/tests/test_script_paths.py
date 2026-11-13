import subprocess
import sys
from pathlib import Path

import pandas as pd
from nodulecxr.records import write_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_data_stats_edges_path_is_sibling_of_out(tmp_path):
    d = tmp_path / "x.md_dir"
    d.mkdir()
    man = d / "manifest_vinbig.csv"
    write_manifest(pd.DataFrame({"image_id": list("abcd"), "split": "train", "label": [1, 1, 1, 0],
                                 "boxes": [[[0, 0, 10, 10]], [[0, 0, 50, 50]], [[0, 0, 300, 300]], []],
                                 "n_rads": [1, 1, 1, 0]}), man)
    out = d / "e0_data.md"
    subprocess.run([sys.executable, str(ROOT / "scripts/data_stats.py"), "--manifests", str(man),
                    "--out", str(out)], check=True)
    assert out.exists() and (d / "e0_data_edges.json").exists()

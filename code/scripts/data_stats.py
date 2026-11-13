"""E0: descriptive table for all manifests -> outputs/e0_data.md."""
import argparse
from pathlib import Path

import pandas as pd

from nodulecxr.data.stats import describe_manifest, to_markdown
from nodulecxr.records import read_manifest
from nodulecxr.strata import box_rel_area, tertile_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dfs = {m: read_manifest(m) for m in a.manifests}
    vin = next(d for m, d in dfs.items() if "vinbig" in m)
    train_pos = vin[(vin.split == "train") & (vin.label == 1)]
    edges = tertile_edges([max(box_rel_area(b) for b in bs) for bs in train_pos.boxes])
    text = [f"# E0: mô tả dữ liệu\n\nTertile diện tích box (tương đối, từ VinBigData train): {edges}\n"]
    for m, d in dfs.items():
        text.append(to_markdown(m, describe_manifest(d, edges)))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    out = Path(a.out)
    pd.Series(edges).to_json(out.with_name(out.stem + "_edges.json"))


if __name__ == "__main__":
    main()

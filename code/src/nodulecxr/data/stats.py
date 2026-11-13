"""E0 descriptive statistics of a manifest."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from nodulecxr.strata import box_rel_area, size_bin


def describe_manifest(df: pd.DataFrame, edges) -> dict:
    per_split = []
    for split, g in df.groupby("split"):
        n, n_pos = len(g), int(g["label"].sum())
        per_split.append({"split": split, "n": n, "n_pos": n_pos, "prevalence": n_pos / n})
    pos = df[df["label"] == 1]
    n_rads = dict(Counter(int(v) for v in pos["n_rads"].dropna())) if "n_rads" in df else {}
    bins = Counter(size_bin(max(box_rel_area(b) for b in boxes), edges)
                   for boxes in pos["boxes"] if boxes)
    return {"per_split": per_split, "n_rads": n_rads, "size_bins": dict(bins)}


def to_markdown(name: str, d: dict) -> str:
    lines = [f"## {name}", "", "| split | n | n_pos | prevalence |", "|---|---|---|---|"]
    lines += [f"| {r['split']} | {r['n']} | {r['n_pos']} | {r['prevalence']:.4f} |" for r in d["per_split"]]
    lines += ["", f"- Số bác sĩ đồng ý (ca dương): {d['n_rads']}", f"- Tầng kích thước: {d['size_bins']}", ""]
    return "\n".join(lines)

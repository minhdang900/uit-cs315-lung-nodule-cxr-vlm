"""Build benchmark tables from manifests + prediction JSONL files."""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from nodulecxr.eval.report import (TEST_SPLITS, _with_labels, aggregate_seeds, compare_configs,
                                   compute_thresholds, evaluate_all, reparse_predictions, run_agreement,
                                   val_selection)
from nodulecxr.records import read_jsonl, read_manifest

COLS = ["sens", "spec", "sens_at_spec95", "auc_roc", "auc_pr", "ece", "balanced_brier", "ppv_prev_1pct",
        "raw_sens", "parse_fail", "unparsed", "hit_iou30", "froc_1fp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--preds", required=True, help="glob, e.g. 'outputs/preds/*.jsonl'")
    ap.add_argument("--edges-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.concat([read_manifest(m) for m in a.manifests])
    preds = pd.concat([read_jsonl(p) for p in glob.glob(a.preds)], ignore_index=True)
    preds = preds.drop_duplicates(subset=["config", "seed", "run", "split", "image_id"], keep="last")
    preds = preds[~preds.config.isin(["pilot"])]
    preds = reparse_predictions(preds)  # rows with raw_text get the current parser's label/boxes/parse_ok
    edges = tuple(pd.read_json(a.edges_json, typ="series").tolist())
    labelled = _with_labels(preds, man)
    thr = compute_thresholds(labelled)
    val_selection(labelled, thr).to_csv(out / "val_selection.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps({f"{c}|{s}|{r}": v for (c, s, r), v in thr.items()}, indent=2))
    summary, strata = evaluate_all(preds, man, edges)
    summary.to_csv(out / "summary_per_seed.csv", index=False)
    strata.to_csv(out / "strata.csv", index=False)
    agg = aggregate_seeds(summary)  # run 0 only; C0 runs 1-2 go to run_agreement.csv
    agg.to_csv(out / "summary.csv", index=False)
    pd.concat([compare_configs(preds, man, thr, s) for s in TEST_SPLITS]).to_csv(out / "comparisons.csv", index=False)
    run_agreement(preds).to_csv(out / "run_agreement.csv", index=False)
    lines = ["# Bảng benchmark (mean ± sd qua seed (C0: run 0, nhiệt độ 0); ngưỡng chọn trên val ở spec 0,90)", "",
             "*Khoảng tin cậy 95% (bootstrap) cho từng seed nằm trong `summary_per_seed.csv`.*", ""]
    for split in TEST_SPLITS:
        t = agg[agg.split == split]
        if t.empty:
            continue
        lines += [f"## {split}", "", "| config | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        for r in t.itertuples():
            cells = [f"{getattr(r, c + '_mean'):.3f} ± {getattr(r, c + '_sd'):.3f}"
                     if pd.notna(getattr(r, c + "_sd")) else f"{getattr(r, c + '_mean'):.3f}" for c in COLS]
            lines.append(f"| {r.config} | " + " | ".join(cells) + " |")
        if split == "test_jsrt":
            lines.append("\n*PPV trên JSRT không có ý nghĩa lâm sàng (tỷ lệ nốt 62%).*")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

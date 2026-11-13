"""Benchmark tables: thresholds from val, metrics with CIs on each test split, strata, comparisons."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nodulecxr.data.jsrt import JSRT_MM_PER_PX_896
from nodulecxr.metrics.detection import (binary_metrics, ppv_at_prevalence, predict_at, score_metrics,
                                         threshold_at_specificity)
from nodulecxr.metrics.localization import best_iou, center_distance, froc_points, jsrt_hit
from nodulecxr.metrics.stats import bootstrap_ci, delong_test, holm, mcnemar_p
from nodulecxr.strata import box_rel_area, size_bin

TEST_SPLITS = ("test", "test_nih", "test_jsrt")
KEY = ["config", "seed", "run"]
JSRT_MARGIN_PX = 5.0 / JSRT_MM_PER_PX_896


def _sens(y, p):
    return binary_metrics(y, p)["sens"]


def _spec(y, p):
    return binary_metrics(y, p)["spec"]


def _auc(y, s):
    return score_metrics(y, s)["auc_roc"]


def compute_thresholds(labelled) -> dict:
    """labelled = _with_labels(preds, manifest); thresholds come from the val split only."""
    out = {}
    for k, g in labelled[labelled.split == "val"].groupby(KEY):
        out[k] = {"t90": threshold_at_specificity(g.y, g.score, 0.90),
                  "t95": threshold_at_specificity(g.y, g.score, 0.95)}
    return out


def _with_labels(preds, manifest):
    m = manifest[["image_id", "split", "label", "boxes", "subtlety", "size_mm"]].rename(
        columns={"label": "y", "boxes": "gt"})
    return preds.merge(m, on=["image_id", "split"], how="inner")


def evaluate_group(g, thr, edges) -> tuple:
    rows, strata = [], []
    for split in TEST_SPLITS:
        d = g[g.split == split]
        if d.empty:
            continue
        y, s = d.y.to_numpy(), d.score.to_numpy()
        p90, p95 = predict_at(s, thr["t90"]), predict_at(s, thr["t95"])
        m = binary_metrics(y, p90)
        row = {"split": split, "n": len(d), "n_pos": int(y.sum()), "threshold": thr["t90"], **m}
        row["sens_at_spec95"] = binary_metrics(y, p95)["sens"]
        row.update(score_metrics(y, s))
        for name, fn, arr in (("sens", _sens, p90), ("spec", _spec, p90), ("auc_roc", _auc, s)):
            _, lo, hi = bootstrap_ci(fn, y, arr, n_boot=1000)
            row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
        row["ppv_prev_1pct"] = ppv_at_prevalence(m["sens"], m["spec"], 0.01)
        row["ppv_prev_3pct"] = ppv_at_prevalence(m["sens"], m["spec"], 0.03)
        generated = bool(d["raw_text"].astype(str).str.len().gt(0).any() if "raw_text" in d
                         else d["label_pred"].notna().any() if "label_pred" in d else False)
        if generated:
            raw = d.label_pred.fillna(0).astype(int).to_numpy() if "label_pred" in d else np.zeros(len(d))
            rm = binary_metrics(y, raw)
            row.update({"raw_sens": rm["sens"], "raw_spec": rm["spec"],
                        "parse_fail": float(1 - d.parse_ok.astype(bool).mean()) if "parse_ok" in d else 0.0,
                        "no_on_pos": 1 - rm["sens"], "yes_on_neg": 1 - rm["spec"]})
        else:  # config never produced text (B1, B2, C1 with --generate none): text metrics are undefined
            row.update({k: float("nan") for k in ("raw_sens", "raw_spec", "parse_fail", "no_on_pos", "yes_on_neg")})
        pos = d[d.y == 1]
        ious = [best_iou(pb, gt) for pb, gt in zip(pos.boxes_pred, pos["gt"])]
        pos_pred = predict_at(pos.score, thr["t90"])
        has_boxes = d.boxes_pred.apply(len).sum() > 0
        no_loc = not generated and not has_boxes  # localization undefined for this config
        row["mean_best_iou"] = float(np.mean(ious)) if ious and not no_loc else float("nan")
        row["hit_iou30"] = (float(np.mean([pp and i >= 0.3 for pp, i in zip(pos_pred, ious)]))
                            if ious and not no_loc else float("nan"))
        row["hit_iou50"] = (float(np.mean([pp and i >= 0.5 for pp, i in zip(pos_pred, ious)]))
                            if ious and not no_loc else float("nan"))
        if has_boxes and len(pos):
            fr = froc_points(list(zip(d.boxes_pred, d["gt"])), iou_thr=0.3)
            row["froc_0.5fp"], row["froc_1fp"] = fr[0.5], fr[1]
        elif generated and len(pos):
            row["froc_0.5fp"] = row["froc_1fp"] = 0.0
        else:
            row["froc_0.5fp"] = row["froc_1fp"] = float("nan")
        if split == "test_jsrt" and no_loc:
            row["jsrt_hit"] = row["jsrt_median_dist_mm"] = float("nan")
        elif split == "test_jsrt" and len(pos):
            hits, dists = [], []
            for pb, gt, pp in zip(pos.boxes_pred, pos["gt"], pos_pred):
                if pp and pb:
                    top = max(pb, key=lambda b: b[4])
                    hits.append(jsrt_hit(top, gt[0], JSRT_MARGIN_PX))
                    dists.append(center_distance(top, gt[0]) * JSRT_MM_PER_PX_896)
                else:
                    hits.append(False)
            row["jsrt_hit"] = float(np.mean(hits))
            row["jsrt_median_dist_mm"] = float(np.median(dists)) if dists else float("nan")
        rows.append(row)
        for r, pp in zip(pos.itertuples(), pos_pred):
            strata.append({"split": split, "size_bin": size_bin(max(box_rel_area(b) for b in r.gt), edges) if r.gt else None,
                           "subtlety": r.subtlety, "detected": int(pp)})
    return rows, strata


def evaluate_all(preds, manifest, edges):
    data = _with_labels(preds, manifest)
    thresholds = compute_thresholds(data)
    summary, strata = [], []
    for k, g in data.groupby(KEY):
        if k not in thresholds:
            continue
        rows, st = evaluate_group(g, thresholds[k], edges)
        meta = dict(zip(KEY, k))
        summary += [{**meta, **r} for r in rows]
        strata += [{**meta, **r} for r in st]
    st = pd.DataFrame(strata)
    if not st.empty:
        parts = [st.groupby(KEY + ["split", "size_bin"]).detected.agg(["mean", "count"]).reset_index()
                   .rename(columns={"size_bin": "stratum"}).assign(kind="size")]
        js = st.dropna(subset=["subtlety"])
        if not js.empty:
            parts.append(js.groupby(KEY + ["split", "subtlety"]).detected.agg(["mean", "count"]).reset_index()
                         .rename(columns={"subtlety": "stratum"}).assign(kind="subtlety"))
        st = pd.concat(parts, ignore_index=True).rename(columns={"mean": "sensitivity"})
    return pd.DataFrame(summary), st


def aggregate_seeds(summary) -> pd.DataFrame:
    num = summary.select_dtypes("number").columns.difference(["seed", "run"])
    g = summary.groupby(["config", "split"])[list(num)]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    out["n_seeds"] = summary.groupby(["config", "split"]).seed.nunique().to_numpy()
    return out


def compare_configs(preds, manifest, thresholds, split, seed=0, run=0) -> pd.DataFrame:
    data = _with_labels(preds, manifest)
    d = data[(data.split == split) & (data.run == run)]
    configs = sorted(c for c in d.config.unique() if (c, seed, run) in thresholds)
    piv = {c: d[(d.config == c) & (d.seed == seed)].set_index("image_id") for c in configs}
    rows = []
    for a, b in itertools.combinations(configs, 2):
        ids = piv[a].index.intersection(piv[b].index)
        A, B = piv[a].loc[ids], piv[b].loc[ids]
        y = A.y.to_numpy()
        pa = predict_at(A.score, thresholds[(a, seed, run)]["t90"])
        pb = predict_at(B.score, thresholds[(b, seed, run)]["t90"])
        pos, neg = y == 1, y == 0
        _, _, p_auc = delong_test(y, A.score.to_numpy(), B.score.to_numpy())
        rows.append({"split": split, "a": a, "b": b,
                     "p_sens": mcnemar_p(pa[pos] == 1, pb[pos] == 1),
                     "p_spec": mcnemar_p(pa[neg] == 0, pb[neg] == 0), "p_auc": p_auc})
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ("p_sens", "p_spec", "p_auc"):
            out[f"{col}_holm"] = holm(out[col])
    return out


def run_agreement(preds) -> pd.DataFrame:
    p = preds.copy()
    p["label_pred"] = p["label_pred"].fillna(0).astype(int)
    rows = []
    for (cfg, seed), g in p.groupby(["config", "seed"]):
        if g.run.nunique() < 2:
            continue
        same = g.groupby("image_id").label_pred.nunique() == 1
        rows.append({"config": cfg, "seed": seed, "n_runs": g.run.nunique(), "agreement": float(same.mean())})
    return pd.DataFrame(rows)


def val_selection(labelled, thresholds) -> pd.DataFrame:
    """Per (config, seed, run): val sensitivity at that run's t90 and val AUC; used to pick configs (never test)."""
    rows = []
    for k, g in labelled[labelled.split == "val"].groupby(KEY):
        if k not in thresholds:
            continue
        y, s = g.y.to_numpy(), g.score.to_numpy()
        rows.append({**dict(zip(KEY, k)), "val_sens90": _sens(y, predict_at(s, thresholds[k]["t90"])),
                     "val_auc": _auc(y, s)})
    return pd.DataFrame(rows, columns=KEY + ["val_sens90", "val_auc"])

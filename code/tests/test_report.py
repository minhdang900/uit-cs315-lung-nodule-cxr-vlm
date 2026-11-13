import numpy as np
import pandas as pd
from nodulecxr.eval.report import (_with_labels, aggregate_seeds, compare_configs, compute_thresholds,
                                   evaluate_all, run_agreement, val_selection)


def _data():
    rng = np.random.default_rng(0)
    rows, man = [], []
    for split, n in (("val", 200), ("test", 200)):
        for i in range(n):
            y = int(i < n // 5)
            iid = f"{split}{i}"
            man.append({"image_id": iid, "dataset": "vinbig", "split": split, "label": y,
                        "boxes": [[100, 100, 150, 150]] if y else [], "subtlety": None, "size_mm": None})
            for cfg, sep in (("C0", 0.2), ("C3", 2.5)):
                s = float(1 / (1 + np.exp(-(rng.normal(sep * y, 1)))))
                rows.append({"image_id": iid, "dataset": "vinbig", "split": split, "config": cfg, "seed": 0,
                             "run": 0, "score": s, "label_pred": int(s > 0.5),
                             "boxes_pred": [[100, 100, 150, 150, s]] if s > 0.5 else [], "parse_ok": True})
    return pd.DataFrame(rows), pd.DataFrame(man)


def test_thresholds_from_val_only_and_summary():
    preds, man = _data()
    thr = compute_thresholds(_with_labels(preds, man))
    assert set(thr) == {("C0", 0, 0), ("C3", 0, 0)}
    summary, strata = evaluate_all(preds, man, edges=(0.001, 0.01))
    row = summary[(summary.config == "C3") & (summary.split == "test")].iloc[0]
    assert 0.8 <= row["spec"] <= 1.0 and row["sens"] > 0.6 and row["auc_roc"] > 0.9
    for col in ("sens_lo", "sens_hi", "auc_pr", "ece", "balanced_brier", "ppv_prev_1pct",
                "raw_sens", "parse_fail", "hit_iou30", "froc_1fp"):
        assert col in summary.columns
    assert summary[summary.config == "C3"]["sens"].iloc[0] > summary[summary.config == "C0"]["sens"].iloc[0]
    agg = aggregate_seeds(summary)
    assert {"sens_mean", "sens_sd"} <= set(agg.columns)
    cmp = compare_configs(preds, man, thr, "test")
    assert {"a", "b", "p_sens", "p_spec", "p_auc", "p_auc_holm"} <= set(cmp.columns)
    assert len(strata) > 0


def test_run_agreement():
    p = pd.DataFrame({"image_id": ["a", "a", "b", "b"], "config": "C0", "seed": 0, "run": [0, 1, 0, 1],
                      "label_pred": [1, 1, 0, 1]})
    assert run_agreement(p).loc[0, "agreement"] == 0.5


def test_non_generating_group_has_nan_text_and_loc_metrics():
    preds, man = _data()
    p = preds[preds.config == "C3"].copy()
    p["config"] = "B1"
    p["label_pred"] = None
    p["boxes_pred"] = [[] for _ in range(len(p))]
    p["raw_text"] = ""
    summary, _ = evaluate_all(p, man, edges=(0.001, 0.01))
    row = summary[summary.split == "test"].iloc[0]
    for col in ("raw_sens", "raw_spec", "parse_fail", "no_on_pos", "yes_on_neg",
                "mean_best_iou", "hit_iou30", "hit_iou50", "froc_1fp"):
        assert np.isnan(row[col]), col
    assert np.isfinite([row["sens"], row["spec"], row["auc_roc"]]).all()


def test_generating_group_without_boxes_has_zero_froc():
    preds, man = _data()
    p = preds[preds.config == "C3"].copy()
    p["boxes_pred"] = [[] for _ in range(len(p))]
    p["raw_text"] = '{"nodule": "no"}'
    summary, _ = evaluate_all(p, man, edges=(0.001, 0.01))
    row = summary[summary.split == "test"].iloc[0]
    assert row["froc_1fp"] == 0.0 and row["hit_iou30"] == 0.0 and np.isfinite(row["raw_sens"])


def test_val_selection():
    preds, man = _data()
    lab = _with_labels(preds, man)
    vs = val_selection(lab, compute_thresholds(lab))
    assert list(vs.columns) == ["config", "seed", "run", "val_sens90", "val_auc"] and len(vs) == 2
    c3 = vs[vs.config == "C3"].iloc[0]
    assert c3.val_sens90 > vs[vs.config == "C0"].iloc[0].val_sens90 and c3.val_auc > 0.9


# ---- final fix wave ----
from nodulecxr.eval.report import evaluate_group, reparse_predictions  # noqa: E402

THR = {"t90": 0.5, "t95": 0.6}
GT = [100, 100, 150, 150]
WRONG = [600, 600, 650, 650]


def _group(pos_boxes, split="test", label_pred=None, raw_text="", size_mm=None, subtlety=None):
    """One row per positive (given boxes, score 0.9) + 4 negatives (no boxes, score 0.1)."""
    rows = []
    for i, pb in enumerate(pos_boxes):
        lp = label_pred[i] if label_pred is not None else None
        rows.append({"image_id": f"p{i}", "split": split, "y": 1, "score": 0.9, "boxes_pred": pb, "gt": [GT],
                     "label_pred": lp, "parse_ok": lp is not None, "raw_text": raw_text,
                     "subtlety": subtlety, "size_mm": size_mm[i] if size_mm else None})
    for i in range(4):
        lp = label_pred[len(pos_boxes) + i] if label_pred is not None else None
        rows.append({"image_id": f"n{i}", "split": split, "y": 0, "score": 0.1, "boxes_pred": [], "gt": [],
                     "label_pred": lp, "parse_ok": lp is not None, "raw_text": raw_text,
                     "subtlety": None, "size_mm": None})
    return pd.DataFrame(rows)


def test_localisation_uses_only_boxes_above_t90():
    lows = [WRONG[:2] + [WRONG[2] + k, WRONG[3]] + [0.01] for k in range(10)]
    good = [GT + [0.9]] + lows                                   # correct high-conf box
    bad = [WRONG + [0.9], GT + [0.01]] + lows                    # correct box only at low conf
    rows, _ = evaluate_group(_group([good, bad]), THR, (0.001, 0.01))
    r = rows[0]
    assert r["hit_iou30"] == 0.5 and r["hit_iou50"] == 0.5 and r["mean_best_iou"] == 0.5
    assert r["froc_1fp"] > 0  # FROC still sees all boxes


def test_only_correct_box_below_t90_gives_zero_hit():
    rows, _ = evaluate_group(_group([[WRONG + [0.9], GT + [0.2]]] * 3), THR, (0.001, 0.01))
    assert rows[0]["hit_iou30"] == 0.0 and rows[0]["mean_best_iou"] == 0.0


def test_normal_mode_metrics_on_parsed_answers_only():
    # positives: 1, 0, None, None ; negatives: 0, 1, None, 0
    g = _group([[], [], [], []], label_pred=[1, 0, None, None, 0, 1, None, 0], raw_text="x")
    r = evaluate_group(g, THR, (0.001, 0.01))[0][0]
    assert r["no_on_pos"] == 0.5 and abs(r["yes_on_neg"] - 1 / 3) < 1e-9
    assert r["unparsed"] == 3 / 8
    assert r["raw_sens"] == 0.25 and r["raw_spec"] == 0.75  # null counted as "no"


def test_run_agreement_only_fully_parsed_images():
    p = pd.DataFrame({"image_id": ["a", "a", "b", "b", "c", "c"], "config": "C0", "seed": 0,
                      "run": [0, 1, 0, 1, 0, 1], "label_pred": [1, 1, 0, 1, 1, None]})
    r = run_agreement(p).iloc[0]
    assert r["agreement"] == 0.5 and r["n_images"] == 2


def test_reparse_predictions_overwrites_generated_rows():
    df = pd.DataFrame([
        {"image_id": "a", "score": 0.8, "label_pred": None, "boxes_pred": [], "parse_ok": False,
         "raw_text": 'Sure: {"nodule_present": true, "bounding_boxes": [[1, 2, 30, 40]]} {}'},
        {"image_id": "b", "score": 0.7, "label_pred": None, "boxes_pred": [[5, 5, 50, 50, 0.7]], "parse_ok": True,
         "raw_text": ""}])
    out = reparse_predictions(df)
    a, b = out.iloc[0], out.iloc[1]
    assert a.label_pred == 1 and a.parse_ok and a.boxes_pred == [[1.0, 2.0, 30.0, 40.0, 0.8]]
    assert b.boxes_pred == [[5, 5, 50, 50, 0.7]] and b.parse_ok and b.label_pred is None
    assert df.iloc[0].label_pred is None  # input untouched


def test_aggregate_seeds_uses_run0_only():
    s = pd.DataFrame({"config": "C0", "split": "test", "seed": 0, "run": [0, 1, 2], "sens": [0.2, 0.8, 0.8]})
    agg = aggregate_seeds(s)
    assert agg.loc[0, "sens_mean"] == 0.2 and agg.loc[0, "n_seeds"] == 1


def test_jsrt_strata_in_mm():
    g = _group([[GT + [0.9]]] * 3, split="test_jsrt", size_mm=[8.0, 15.0, 25.0], subtlety=3)
    _, st = evaluate_group(g, THR, (0.001, 0.01))
    assert [s["size_bin"] for s in st] == ["≤10mm", "10–20mm", ">20mm"]
    g2 = _group([[GT + [0.9]]], split="test")
    assert evaluate_group(g2, THR, (0.001, 0.01))[1][0]["size_bin"] == "medium"

import numpy as np
import pandas as pd
from nodulecxr.eval.report import (_with_labels, aggregate_seeds, compare_configs, compute_thresholds,
                                   evaluate_all, run_agreement)


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

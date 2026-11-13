import pandas as pd
from nodulecxr.data.split import oversample_indices, stratified_split, subsample_negatives


def _labels():
    return pd.Series([1] * 30 + [0] * 970, index=[f"i{k}" for k in range(1000)])


def test_stratified_split_sizes_and_positives():
    s = stratified_split(_labels(), seed=0)
    counts = s.value_counts()
    assert abs(counts["train"] - 700) <= 1 and abs(counts["val"] - 100) <= 1 and abs(counts["test"] - 200) <= 1
    pos = _labels()[_labels() == 1].index
    assert abs((s.loc[pos] == "test").sum() - 6) <= 1
    assert s.equals(stratified_split(_labels(), seed=0))


def test_split_without_test():
    s = stratified_split(_labels(), ratios=(0.85, 0.15, 0.0), seed=0)
    assert set(s.unique()) == {"train", "val"}


def test_subsample_negatives_keeps_all_positives():
    df = pd.DataFrame({"label": [1] * 10 + [0] * 100})
    out = subsample_negatives(df, neg_per_pos=4, seed=0)
    assert (out.label == 1).sum() == 10 and (out.label == 0).sum() == 40


def test_oversample_balances_classes():
    labels = [1] * 10 + [0] * 90
    idx = oversample_indices(labels, seed=0)
    picked = [labels[i] for i in idx]
    assert len(idx) == 180 and sum(picked) == 90

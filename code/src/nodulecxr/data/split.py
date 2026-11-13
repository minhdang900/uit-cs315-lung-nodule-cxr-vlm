"""Train/val/test splitting, negative subsampling and positive oversampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(labels: pd.Series, ratios=(0.7, 0.1, 0.2), seed: int = 0) -> pd.Series:
    idx = labels.index.to_numpy()
    tr, rest = train_test_split(idx, train_size=ratios[0], stratify=labels.to_numpy(), random_state=seed)
    out = pd.Series("train", index=labels.index, dtype=object)
    if ratios[2] == 0:
        out.loc[rest] = "val"
        return out
    val_frac = ratios[1] / (ratios[1] + ratios[2])
    va, te = train_test_split(rest, train_size=val_frac, stratify=labels.loc[rest].to_numpy(),
                              random_state=seed)
    out.loc[va] = "val"
    out.loc[te] = "test"
    return out


def subsample_negatives(df: pd.DataFrame, neg_per_pos: float, seed: int = 0) -> pd.DataFrame:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    k = min(len(neg), int(round(neg_per_pos * len(pos))))
    return pd.concat([pos, neg.sample(n=k, random_state=seed)]).sort_index()


def oversample_indices(labels, seed: int = 0) -> list:
    y = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    reps = np.concatenate([np.tile(pos, len(neg) // len(pos)),
                           rng.choice(pos, len(neg) % len(pos), replace=False)])
    idx = np.concatenate([neg, reps])
    rng.shuffle(idx)
    return idx.tolist()

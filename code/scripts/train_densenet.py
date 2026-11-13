"""Train/evaluate B1 DenseNet-121 on VinBigData; predict all splits -> JSONL."""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from nodulecxr.baselines.densenet import build_model, to_xrv_array
from nodulecxr.images import load_image
from nodulecxr.records import append_jsonl, read_manifest


class DS(Dataset):
    def __init__(self, df, root):
        self.df, self.root = df.reset_index(drop=True), root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.from_numpy(to_xrv_array(load_image(self.root, r.path))), float(r.label)


def scores(model, df, root, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(df, root), batch_size=64, num_workers=2):
            out.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    df = pd.concat([read_manifest(m) for m in a.manifests])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    pos_w = torch.tensor((tr.label == 0).sum() / max(1, (tr.label == 1).sum()), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_auc, best_state = -1.0, None
    for epoch in range(a.epochs):
        model.train()
        for x, y in DataLoader(DS(tr, a.data_root), batch_size=32, shuffle=True, num_workers=2):
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.float().to(dev))
            loss.backward()
            opt.step()
        auc = roc_auc_score(va.label, scores(model, va, a.data_root, dev))
        print(f"epoch {epoch} val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ev = df[df.split != "train"]
    for r, s in zip(ev.itertuples(), scores(model, ev, a.data_root, dev)):
        append_jsonl({"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": float(s),
                      "label_pred": None, "boxes_pred": [], "parse_ok": True, "raw_text": "",
                      "config": "B1", "seed": a.seed, "run": 0}, a.out)


if __name__ == "__main__":
    main()

"""Run a VLM (zero-shot or with a LoRA adapter) over manifest splits -> JSONL (resumable)."""
import argparse

import pandas as pd
from tqdm import tqdm

from nodulecxr.images import load_image
from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC
from nodulecxr.records import append_jsonl, done_ids, read_manifest
from nodulecxr.vlm.hf import HFVLM
from nodulecxr.vlm.predict import iter_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["medgemma", "qwen"], required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prompt", choices=["loc", "cls"], default="loc")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--generate", choices=["all", "none"], default="all")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([read_manifest(m) for m in a.manifests])
    df = df[df.split.isin(a.splits)]
    if a.limit:
        df = df.groupby("split", group_keys=False).apply(lambda g: g.head(a.limit))
    vlm = HFVLM(a.model, a.adapter)
    prompt = QUESTION_CLS if a.prompt == "cls" else QUESTION_LOC
    meta = {"config": a.config, "seed": a.seed, "run": a.run, "model": a.model}
    skip = done_ids(a.out, config=a.config, seed=a.seed, run=a.run)
    rows = iter_predictions(vlm, df, prompt, lambda p: load_image(a.data_root, p), a.temperature,
                            a.seed + 1000 * a.run, a.generate == "all", a.max_new_tokens, meta, skip)
    for row in tqdm(rows, total=len(df) - len(skip)):
        append_jsonl(row, a.out)


if __name__ == "__main__":
    main()

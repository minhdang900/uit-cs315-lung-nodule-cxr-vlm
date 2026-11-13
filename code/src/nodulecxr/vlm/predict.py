"""Model-agnostic prediction loop producing JSONL rows."""
from __future__ import annotations

from nodulecxr.prompts import ParsedAnswer, parse_answer


def iter_predictions(vlm, records, prompt, load_image, temperature=0.0, seed=None, generate=True,
                     max_new_tokens=96, meta=None, skip=()):
    for r in records.itertuples():
        if r.image_id in skip:
            continue
        img = load_image(r.path)
        score = float(vlm.yes_probability(img, prompt))
        if generate:
            text = vlm.generate(img, prompt, temperature=temperature,
                                max_new_tokens=max_new_tokens, seed=seed)
            pa = parse_answer(text)
        else:
            text, pa = "", ParsedAnswer(None, [], False)
        yield {"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": score,
               "label_pred": pa.label, "boxes_pred": [b + [score] for b in pa.boxes],
               "parse_ok": pa.ok, "raw_text": text, **(meta or {})}

import math

import pandas as pd
import pytest
from PIL import Image
from nodulecxr.vlm.chat import build_chat_text
from nodulecxr.vlm.predict import iter_predictions
from nodulecxr.vlm.scoring import first_divergence, two_way_probability


class FakeProcessor:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "|".join(f"{m['role']}:" + "".join(c.get("text", "<img>") for c in m["content"]) for m in msgs)
        return s + ("|assistant:" if add_generation_prompt else "")


def test_chat_text_prefix_property():
    p = FakeProcessor()
    head = build_chat_text(p, "Q")
    full = build_chat_text(p, "Q", "A")
    assert head == "user:<img>Q|assistant:" and full == "user:<img>Q|assistant:A"
    assert full.startswith(head)


def test_scoring_helpers():
    assert first_divergence([1, 2, 3, 9], [1, 2, 4]) == 2
    assert two_way_probability(math.log(0.3), math.log(0.1)) == pytest.approx(0.75)


class FakeVLM:
    def yes_probability(self, image, prompt):
        return 0.8

    def generate(self, image, prompt, temperature=0.0, max_new_tokens=96, seed=None):
        return '{"nodule": "yes", "boxes": [[10, 10, 50, 50]]}'


def test_iter_predictions_rows_and_skip():
    recs = pd.DataFrame({"image_id": ["a", "b"], "dataset": "vinbig", "split": "test", "path": ["a", "b"]})
    rows = list(iter_predictions(FakeVLM(), recs, "Q", lambda p: Image.new("RGB", (896, 896)),
                                 meta={"config": "C0", "seed": 0, "run": 0}, skip={"b"}))
    assert len(rows) == 1
    r = rows[0]
    assert r["image_id"] == "a" and r["score"] == 0.8 and r["label_pred"] == 1 and r["parse_ok"]
    assert r["boxes_pred"] == [[10.0, 10.0, 50.0, 50.0, 0.8]] and r["config"] == "C0"


def test_token_logprob_sum():
    torch = pytest.importorskip("torch")
    from nodulecxr.vlm.scoring import token_logprob_sum
    logits = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]))
    ids = torch.tensor([0, 1, 0])
    # tokens from position 1: P(ids[1]=1 | pos0)=0.5, P(ids[2]=0 | pos1)=0.25
    assert token_logprob_sum(logits, ids, 1) == pytest.approx(math.log(0.5) + math.log(0.25))


@pytest.mark.parametrize("avail,cap,expected", [(False, (8, 0), False), (True, (7, 5), False),
                                                (True, (8, 9), True), (True, (9, 0), True)])
def test_bf16_capable_requires_ampere(monkeypatch, avail, cap, expected):
    import torch
    from nodulecxr.vlm.hf import bf16_capable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: avail)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # T4 emulation must not pass
    assert bf16_capable() is expected

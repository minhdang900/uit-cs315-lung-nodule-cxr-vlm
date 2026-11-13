import re

import pytest

torch = pytest.importorskip("torch")
from nodulecxr.train.losses import weighted_token_ce  # noqa: E402
from nodulecxr.train.sft_data import LORA_TARGET_REGEX, build_example, collate  # noqa: E402


def test_lora_regex_targets_llm_only():
    r = re.compile(LORA_TARGET_REGEX)
    assert r.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert r.fullmatch("model.layers.3.mlp.down_proj")
    assert not r.fullmatch("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not r.fullmatch("visual.blocks.0.attn.qkv")
    assert not r.fullmatch("visual.blocks.0.mlp.down_proj")
    assert not r.fullmatch("model.multi_modal_projector.mm_input_projection")


class TinyProcessor:
    """Character-level fake: each char is a token; '<img>' expands to 4 image tokens."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"[{m['role']}]" + "".join(c.get("text", "<img>") for c in m["content"]) for m in msgs)
        return s + ("[assistant]" if add_generation_prompt else "") + ("" if add_generation_prompt else "<e>")

    def __call__(self, text, images, return_tensors="pt"):
        t = text[0].replace("<img>", "@@@@")
        ids = torch.tensor([[ord(c) % 97 + 3 for c in t]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros(1, 3, 2, 2)}


def test_build_example_masks_prompt():
    from PIL import Image
    ex = build_example(TinyProcessor(), Image.new("RGB", (4, 4)), "Q?", "A!")
    n_answer = len("A!") + len("<e>")
    assert (ex["labels"] != -100).sum().item() == n_answer
    assert ex["pixel_values"].shape == (1, 3, 2, 2)


def test_collate_pads_and_concats():
    a = {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.ones(3, dtype=torch.long),
         "labels": torch.tensor([-100, 6, 7]), "pixel_values": torch.zeros(1, 3, 2, 2), "sample_weight": 2.0}
    b = {"input_ids": torch.tensor([5]), "attention_mask": torch.ones(1, dtype=torch.long),
         "labels": torch.tensor([5]), "pixel_values": torch.zeros(1, 3, 2, 2), "sample_weight": 1.0}
    out = collate([a, b], pad_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert out["labels"].tolist()[1] == [5, -100, -100]
    assert out["attention_mask"].tolist()[1] == [1, 0, 0]
    assert out["pixel_values"].shape == (2, 3, 2, 2) and out["sample_weight"].tolist() == [2.0, 1.0]


def test_weighted_ce_reduces_to_mean_and_weights_samples():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 0]])
    plain = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1),
                                              ignore_index=-100)
    assert weighted_token_ce(logits, labels, torch.ones(2)) == pytest.approx(plain.item(), rel=1e-5)
    heavy = weighted_token_ce(logits, labels, torch.tensor([10.0, 1.0]))
    only_first = weighted_token_ce(logits, labels, torch.tensor([1.0, 0.0]))
    assert abs(heavy.item() - only_first.item()) < abs(plain.item() - only_first.item())


def test_weighted_ce_denom_makes_weight_effective_at_batch_one():
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 6)
    labels = torch.tensor([[-100, -100, 2, 3, 1]])
    n = float((labels[:, 1:] != -100).sum())
    l1 = weighted_token_ce(logits, labels, torch.tensor([1.0]), denom=n)
    l2 = weighted_token_ce(logits, labels, torch.tensor([2.0]), denom=n)
    assert l2.item() == pytest.approx(2 * l1.item(), rel=1e-6)
    # denom=None keeps the normalised mean: weight cancels at batch size 1
    a = weighted_token_ce(logits, labels, torch.tensor([1.0]))
    b = weighted_token_ce(logits, labels, torch.tensor([2.0]))
    assert a.item() == pytest.approx(b.item(), rel=1e-6)

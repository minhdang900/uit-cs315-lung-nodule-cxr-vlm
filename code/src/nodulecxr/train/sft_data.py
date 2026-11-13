"""Supervised fine-tuning examples: prompt-masked labels, padding collator, dataset."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC, build_target
from nodulecxr.vlm.chat import build_chat_text

LORA_TARGET_REGEX = (r"^(?!.*(?:visual|vision_tower|vision_model|multi_modal_projector)).*\."
                     r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
SEQ_KEYS = ("input_ids", "attention_mask", "labels", "token_type_ids")
IMAGE_KEYS = ("pixel_values", "image_grid_thw")


def build_example(processor, image, prompt: str, answer: str) -> dict:
    img = image.convert("RGB")
    full = processor(text=[build_chat_text(processor, prompt, answer)], images=[img], return_tensors="pt")
    head = processor(text=[build_chat_text(processor, prompt)], images=[img], return_tensors="pt")
    n_head = head["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n_head], head["input_ids"][0]):
        raise ValueError("chat template: prompt is not a prefix of prompt+answer")
    ex = {k: v[0] for k, v in full.items() if k in SEQ_KEYS}
    for k in IMAGE_KEYS:
        if k in full:
            ex[k] = full[k]
    labels = ex["input_ids"].clone()
    labels[:n_head] = -100
    ex["labels"] = labels
    return ex


def collate(batch, pad_id: int) -> dict:
    out = {}
    maxlen = max(len(b["input_ids"]) for b in batch)
    for k in SEQ_KEYS:
        if k not in batch[0]:
            continue
        fill = pad_id if k == "input_ids" else (-100 if k == "labels" else 0)
        out[k] = torch.stack([F.pad(b[k], (0, maxlen - len(b[k])), value=fill) for b in batch])
    for k in IMAGE_KEYS:
        if k in batch[0]:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    if "sample_weight" in batch[0]:
        out["sample_weight"] = torch.tensor([float(b["sample_weight"]) for b in batch])
    return out


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, mode: str, load_image, pos_weight: float = 1.0, indices=None):
        assert mode in ("cls", "loc")
        self.df = df.reset_index(drop=True)
        self.processor, self.mode, self.load_image = processor, mode, load_image
        self.pos_weight = pos_weight
        self.indices = list(indices) if indices is not None else list(range(len(self.df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r = self.df.iloc[self.indices[i]]
        prompt = QUESTION_CLS if self.mode == "cls" else QUESTION_LOC
        answer = build_target(int(r.label), r.boxes, with_boxes=self.mode == "loc")
        ex = build_example(self.processor, self.load_image(r.path), prompt, answer)
        ex["sample_weight"] = self.pos_weight if int(r.label) == 1 else 1.0
        return ex

"""Hugging Face loaders and inference for MedGemma-4B-it and Qwen2.5-VL-3B."""
from __future__ import annotations

from nodulecxr.prompts import ANSWER_PREFIX
from nodulecxr.records import IMG_SIZE
from nodulecxr.vlm.chat import build_chat_text
from nodulecxr.vlm.scoring import first_divergence, token_logprob_sum, two_way_probability

MODEL_IDS = {"medgemma": "google/medgemma-4b-it", "qwen": "Qwen/Qwen2.5-VL-3B-Instruct"}


def load_base(key: str, dtype=None):
    import torch
    from transformers import AutoProcessor
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    mid = MODEL_IDS[key]
    if key == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        processor = AutoProcessor.from_pretrained(mid, min_pixels=IMG_SIZE * IMG_SIZE,
                                                  max_pixels=IMG_SIZE * IMG_SIZE)
    else:
        from transformers import AutoModelForImageTextToText as M
        processor = AutoProcessor.from_pretrained(mid)
    model = M.from_pretrained(mid, torch_dtype=dtype, device_map="auto")
    return model, processor


class HFVLM:
    def __init__(self, key: str, adapter_path=None):
        import torch
        self.torch = torch
        self.model, self.processor = load_base(key)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _encode(self, text, image):
        enc = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        return enc.to(self.model.device, dtype=self.model.dtype)

    def generate(self, image, prompt, temperature=0.0, max_new_tokens=96, seed=None) -> str:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        enc = self._encode(build_chat_text(self.processor, prompt), image)
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        n = enc["input_ids"].shape[1]
        return self.processor.tokenizer.decode(out[0, n:], skip_special_tokens=True)

    def _forced(self, image, prompt, stub):
        enc = self._encode(build_chat_text(self.processor, prompt) + stub, image)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
        return logits, enc["input_ids"][0]

    def yes_probability(self, image, prompt) -> float:
        ly, iy = self._forced(image, prompt, ANSWER_PREFIX + 'yes"')
        ln, in_ = self._forced(image, prompt, ANSWER_PREFIX + 'no"')
        k = first_divergence(iy.tolist(), in_.tolist())
        return two_way_probability(token_logprob_sum(ly, iy, k), token_logprob_sum(ln, in_, k))

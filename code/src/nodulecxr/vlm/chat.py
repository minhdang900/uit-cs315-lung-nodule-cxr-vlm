"""Chat-template text shared by inference and training (no heavy imports)."""
from __future__ import annotations


def build_chat_text(processor, prompt: str, answer=None) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None)

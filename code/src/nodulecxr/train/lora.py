"""LoRA configuration and Trainer subclass using the sample-weighted loss."""
from __future__ import annotations

from nodulecxr.train.losses import weighted_token_ce
from nodulecxr.train.sft_data import LORA_TARGET_REGEX


def make_lora_config():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGET_REGEX,
                      bias="none", task_type="CAUSAL_LM")


def _trainer_base():
    from transformers import Trainer
    return Trainer


class WeightedSFTTrainer(_trainer_base()):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trainer then passes num_items_in_batch and does not re-divide by grad-accum steps.
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("sample_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = weighted_token_ce(outputs.logits, labels, weights.to(outputs.logits.device),
                                 denom=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

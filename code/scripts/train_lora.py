"""LoRA fine-tuning of MedGemma-4B-it / Qwen2.5-VL-3B on VinBigData nodule targets."""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from transformers import TrainingArguments

from nodulecxr.data.split import oversample_indices, subsample_negatives
from nodulecxr.images import load_image
from nodulecxr.records import read_manifest
from nodulecxr.train.lora import WeightedSFTTrainer, make_lora_config
from nodulecxr.train.sft_data import SFTDataset, collate
from nodulecxr.vlm.hf import load_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["medgemma", "qwen"], required=True)
    ap.add_argument("--mode", choices=["cls", "loc"], required=True)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, help="timing pilot: cap training samples")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    train = subsample_negatives(df[df.split == "train"], a.neg_per_pos, a.seed).reset_index(drop=True)
    if a.max_train:
        train = train.sample(n=min(a.max_train, len(train)), random_state=a.seed).reset_index(drop=True)
    indices = oversample_indices(train.label, a.seed) if a.balance else None

    model, processor = load_base(a.model)
    model.enable_input_require_grads()
    from peft import get_peft_model
    model = get_peft_model(model, make_lora_config())
    model.print_trainable_parameters()

    ds = SFTDataset(train, processor, a.mode, lambda p: load_image(a.data_root, p), a.pos_weight, indices)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", seed=a.seed, dataloader_num_workers=2)
    trainer = WeightedSFTTrainer(model=model, args=targs, train_dataset=ds,
                                 data_collator=partial(collate, pad_id=processor.tokenizer.pad_token_id))
    t0 = time.time()
    trainer.train()
    model.save_pretrained(Path(a.out) / "final")
    Path(a.out, "timing.json").write_text(json.dumps(
        {"seconds": time.time() - t0, "n_train": len(ds), "epochs": a.epochs}))


if __name__ == "__main__":
    main()

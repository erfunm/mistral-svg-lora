import os
import argparse
import yaml
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from .data import load_svg_dataset
from .model import build_model, build_tokenizer, attach_lora

def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Read config
    base_model = cfg.get("model", {}).get("base_model", "mistralai/Mistral-7B-v0.1")
    load_4bit = bool(cfg.get("model", {}).get("load_4bit", True))
    max_seq_length = int(cfg.get("training", {}).get("max_seq_length", 1024))
    text_field = cfg.get("data", {}).get("text_field", "text")
    data_path = cfg.get("data", {}).get("path", os.getenv("SVG_DATA_PATH", "data/svg_train.jsonl"))

    output_dir = cfg.get("training", {}).get("output_dir", "mistral-svg-lora")
    num_train_epochs = float(cfg.get("training", {}).get("epochs", 3))
    per_device_train_batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    gradient_accumulation_steps = int(cfg.get("training", {}).get("grad_accum", 16))
    learning_rate = float(cfg.get("training", {}).get("lr", 2e-4))
    warmup_ratio = float(cfg.get("training", {}).get("warmup_ratio", 0.03))
    logging_steps = int(cfg.get("training", {}).get("logging_steps", 10))
    save_steps = int(cfg.get("training", {}).get("save_steps", 200))
    save_total_limit = int(cfg.get("training", {}).get("save_total_limit", 3))
    gradient_checkpointing = bool(cfg.get("training", {}).get("gradient_checkpointing", True))
    report_to = cfg.get("training", {}).get("report_to", ["none"])

    # Build components
    special_tokens = cfg.get('tokenizer', {}).get('special_tokens', [])
        tokenizer, added_tokens = build_tokenizer(base_model, special_tokens)
    model = build_model(base_model, load_4bit=load_4bit)
        # Resize embeddings if we added special tokens
        if 'added_tokens' in locals() and added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = attach_lora(model,
                        r=int(cfg.get("lora", {}).get("r", 16)),
                        alpha=int(cfg.get("lora", {}).get("alpha", 32)),
                        dropout=float(cfg.get("lora", {}).get("dropout", 0.05)),
                        target_modules=cfg.get("lora", {}).get("target_modules"))

    # Data
    ds = load_svg_dataset(path=data_path, text_field=text_field)
    train_ds = ds["train"]

    # Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit" if load_4bit else "adamw_torch",
        report_to=report_to,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field=text_field,
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args,
    )

    trainer.train()
    # Save adapter + tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)

if __name__ == "__main__":
    cli()

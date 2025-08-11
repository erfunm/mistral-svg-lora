import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # bitsandbytes optional

DEFAULT_TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

def build_tokenizer(base_model: str, special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    added_tokens = 0
    if special_tokens:
        # Only add tokens not already in vocab
        vocab = tokenizer.get_vocab()
        to_add = [t for t in special_tokens if t not in vocab]
        if to_add:
            added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer, added_tokens

def build_model(base_model: str, load_4bit: bool = True, torch_dtype=None):
    quant_cfg = None
    if load_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes not available. Install it or set load_4bit=False.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32 if torch_dtype is None else torch_dtype,
    )
    if load_4bit:
        model = prepare_model_for_kbit_training(model)
    return model

def attach_lora(model, r=16, alpha=32, dropout=0.05, target_modules=None):
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)

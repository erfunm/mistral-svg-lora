import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def load_for_inference(base_model: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tokenizer

def generate(prompt: str, base_model: str, adapter_dir: str, max_new_tokens: int = 128):
    model, tokenizer = load_for_inference(base_model, adapter_dir)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
    return out

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--adapter_dir", type=str, default="mistral-svg-lora")
    ap.add_argument("--prompt", type=str, default="Hello, Mistral!")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()
    text = generate(args.prompt, args.base_model, args.adapter_dir, args.max_new_tokens)
    print(text)

if __name__ == "__main__":
    cli()

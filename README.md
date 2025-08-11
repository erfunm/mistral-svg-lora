# Mistral Fine‑Tuning on SVG (LoRA / Q‑LoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erfunm/mistral-svg-lora/blob/main/notebooks/finetune_mistral_svg.ipynb)

Minimal, production‑style repo to fine‑tune **Mistral** on SVG text data using **LoRA / Q‑LoRA**. 
Comes with a runnable notebook and clean Python modules under `src/`.

## Features
- ⚙️ Modular code (`src/`) + runnable notebook (`notebooks/`)
- 🧩 LoRA with optional **4‑bit** quantization (Q‑LoRA) via bitsandbytes
- 📁 Config‑driven (`configs/default.yaml`) — no code edits needed for common changes
- 🚀 Colab/A100‑ready; also works on T4 with Q‑LoRA

## Structure
```text
mistral-svg-lora/
├── notebooks/
│   └── finetune_mistral_svg.ipynb
├── src/
│   ├── data.py          # dataset loader (JSON/JSONL/CSV or HF dataset ID)
│   ├── model.py         # tokenizer + model + LoRA attach
│   ├── train.py         # SFT Trainer entrypoint
│   └── inference.py     # simple generation CLI
├── configs/
│   └── default.yaml     # paths & hyperparameters
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart
1. **Install**
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset**
   - Put your data at `data/svg_train.jsonl` **or** set `SVG_DATA_PATH=/path/to/your.jsonl`  
   - File must contain a `text` field per row.
3. **Train**
   ```bash
   python -m src.train --config configs/default.yaml
   ```
4. **Inference**
   ```bash
   python -m src.inference --base_model mistralai/Mistral-7B-v0.1                            --adapter_dir mistral-svg-lora                            --prompt "Hello"
   ```
5. **Notebook (optional)**  
   Open `notebooks/finetune_mistral_svg.ipynb` locally or click the Colab badge above.

## Semantic tokens (SVG)
If your dataset benefits from **semantic tokens** (e.g., `<svg>`, `<path>`, etc.), specify them in
`tokenizer.special_tokens` in the YAML. The training script will add them to the tokenizer and
automatically **resize the model embeddings**.

## Config (YAML)
```yaml
model:
  base_model: mistralai/Mistral-7B-v0.1
  load_4bit: true

data:
  path: data/svg_train.jsonl
  text_field: text

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  # target_modules: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

training:
  output_dir: mistral-svg-lora
  epochs: 3
  batch_size: 1
  grad_accum: 16
  lr: 0.0002
  warmup_ratio: 0.03
  logging_steps: 10
  save_steps: 200
  save_total_limit: 3
  gradient_checkpointing: true
  max_seq_length: 1024
  report_to: ['none']
```

## Hardware
- ✅ **Google Colab A100 40GB**: works out‑of‑the‑box with defaults.
- ✅ **T4 16GB**: set `load_4bit: true`, keep `batch_size: 1`, consider `grad_accum: 16–32`.
- For multi‑GPU, configure `accelerate`:
  ```bash
  accelerate config
  ```

## Notes
- Saves **LoRA adapter** and tokenizer to `training.output_dir` (default: `mistral-svg-lora`).
- `requirements.txt` excludes `torch` so users can match their CUDA version.
- Data and model artifacts are ignored via `.gitignore`.

## License
Choose a license (MIT recommended). If you prefer, add this file as `LICENSE`:
- MIT: <https://choosealicense.com/licenses/mit/>
- Apache‑2.0: <https://choosealicense.com/licenses/apache-2.0/>


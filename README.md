# 🧬 LLM LoRA Fine-Tuning for Text-to-SQL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracked-orange)](https://wandb.ai/)

Fine-tune **Mistral-7B-v0.3** (and other open-source LLMs) for **Natural Language to SQL** generation using **QLoRA** (4-bit quantization + Low-Rank Adaptation). This project demonstrates parameter-efficient fine-tuning on the **sql-create-context** dataset (derived from WikiSQL + Spider), with full evaluation, experiment tracking, and a deployable Gradio demo.

<p align="center">
  <img src="assets/architecture.png" alt="Architecture" width="800"/>
</p>

---

## 📋 Table of Contents

- [🧬 LLM LoRA Fine-Tuning for Text-to-SQL](#-llm-lora-fine-tuning-for-text-to-sql)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Highlights](#-highlights)
  - [🏗️ Architecture](#️-architecture)
  - [📊 Results](#-results)
    - [Base Model vs. Fine-Tuned (on sql-create-context test split)](#base-model-vs-fine-tuned-on-sql-create-context-test-split)
    - [Training Metrics](#training-metrics)
  - [🚀 Quick Start](#-quick-start)
    - [1. Clone \& Install](#1-clone--install)
    - [2. Configure](#2-configure)
    - [3. Prepare Data](#3-prepare-data)
    - [4. Train](#4-train)
    - [5. Evaluate](#5-evaluate)
    - [6. Demo](#6-demo)
  - [📦 Dataset](#-dataset)
    - [Prompt Format](#prompt-format)
  - [🏋️ Training](#️-training)
    - [QLoRA Configuration](#qlora-configuration)
    - [Launch Training](#launch-training)
  - [📏 Evaluation](#-evaluation)
    - [Metrics](#metrics)
  - [🖥️ Inference \& Demo](#️-inference--demo)
    - [Single Query](#single-query)
    - [Gradio Demo](#gradio-demo)
  - [📁 Project Structure](#-project-structure)
  - [⚙️ Configuration](#️-configuration)
  - [💻 Hardware Requirements](#-hardware-requirements)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [📄 License](#-license)

---

## ✨ Highlights

- **QLoRA (4-bit)** — Fine-tune a 7B-parameter model on a single **24 GB GPU** (or free-tier Colab/Kaggle)
- **Text-to-SQL** — Practical, real-world task connecting to enterprise NL2SQL applications
- **Full Pipeline** — Data prep → Training → Evaluation → Inference → Gradio Demo
- **Experiment Tracking** — Weights & Biases integration with loss curves, learning rate schedules, and eval metrics
- **Rigorous Evaluation** — Execution accuracy (run SQL against SQLite), BLEU, exact-match, and error categorization
- **Before/After Comparison** — Base model vs. fine-tuned model on the same test set
- **Multi-Model Support** — Config-driven; swap Mistral for Llama 3.1 8B, CodeLlama, Phi-3, or Qwen2 by changing one line
- **Deployable** — Merged LoRA weights → Hugging Face Hub → Gradio Space

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QLoRA Fine-Tuning                        │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌───────────────────────────┐ │
│  │ Dataset   │───▶│  Tokenizer │───▶│  Mistral-7B (4-bit NF4)  │ │
│  │ (SQL-     │    │  + Prompt  │    │  + LoRA Adapters (r=16)  │ │
│  │  Create-  │    │  Template  │    │  Trainable: ~0.6% params │ │
│  │  Context) │    └───────────┘    └──────────┬────────────────┘ │
│  └──────────┘                                 │                  │
│                                               ▼                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  SFTTrainer (TRL)                                     │       │
│  │  • Paged AdamW 8-bit optimizer                        │       │
│  │  • Cosine LR schedule                                 │       │
│  │  • Gradient checkpointing                             │       │
│  │  • W&B logging                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐     │
│  │ LoRA Adapter  │  │ Merged Model │  │ Gradio Demo       │     │
│  │ (~50 MB)      │──▶│ (FP16/GGUF) │──▶│ + HF Hub Upload  │     │
│  └──────────────┘  └──────────────┘  └───────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Base Model vs. Fine-Tuned (on sql-create-context test split)

| Metric | Mistral-7B (Base) | Mistral-7B + QLoRA | Δ |
|--------|-------------------|---------------------|---|
| Execution Accuracy | — % | — % | +— % |
| Exact Match | — % | — % | +— % |
| BLEU Score | — | — | +— |
| Valid SQL Rate | — % | — % | +— % |

### Training Metrics

| Metric | Value |
|--------|-------|
| Trainable Parameters | ~24M / 7.2B (0.33%) |
| Training Time | ~X hrs on 1× A100-40GB |
| Peak GPU Memory | ~18 GB (4-bit) |
| Final Train Loss | — |
| Final Eval Loss | — |

<details>
<summary>📈 Training Loss Curve (click to expand)</summary>

![Training Loss](assets/training_loss.png)

</details>

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/visheshgupta29/llm-lora-finetuning.git
cd llm-lora-finetuning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your HuggingFace token and W&B API key
```

### 3. Prepare Data

```bash
python -m src.data.prepare_dataset
```

### 4. Train

```bash
python -m src.train.finetune_lora --config configs/training_config.yaml
```

### 5. Evaluate

```bash
python -m src.evaluate.evaluate_model \
    --adapter-path outputs/checkpoint-best \
    --test-split data/processed/test.jsonl
```

### 6. Demo

```bash
python -m src.inference.serve
# Opens Gradio interface at http://localhost:7860
```

---

## 📦 Dataset

We use [**b-mc2/sql-create-context**](https://huggingface.co/datasets/b-mc2/sql-create-context) — a curated combination of **WikiSQL** and **Spider** datasets containing ~78K examples of:

- **Natural Language Question** — e.g., *"How many employees earn more than 50000?"*
- **SQL CREATE TABLE Context** — The schema of relevant tables
- **Gold SQL Query** — The correct SQL answer

### Prompt Format

```
### Task: Generate a SQL query to answer the following question.

### Database Schema:
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    salary REAL,
    department TEXT
);

### Question:
How many employees earn more than 50000?

### SQL Query:
SELECT COUNT(*) FROM employees WHERE salary > 50000;
```

---

## 🏋️ Training

### QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | NF4 (4-bit) | Best quality for 4-bit per QLoRA paper |
| LoRA Rank (r) | 16 | Good accuracy/efficiency tradeoff |
| LoRA Alpha | 32 | Standard α = 2r scaling |
| LoRA Dropout | 0.05 | Light regularization |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All linear layers for best quality |
| Learning Rate | 2e-4 | Standard for QLoRA |
| LR Schedule | Cosine | Smooth decay |
| Batch Size | 4 (effective 16 via gradient accumulation) | Fits in 24 GB VRAM |
| Max Seq Length | 1024 | Sufficient for SQL queries |
| Epochs | 3 | Prevents overfitting |
| Optimizer | Paged AdamW 8-bit | Reduces optimizer memory footprint |

### Launch Training

```bash
# Single GPU
python -m src.train.finetune_lora --config configs/training_config.yaml

# With custom overrides
python -m src.train.finetune_lora \
    --config configs/training_config.yaml \
    --model-name "meta-llama/Llama-3.1-8B" \
    --lora-r 32 \
    --epochs 5

# Resume from checkpoint
python -m src.train.finetune_lora \
    --config configs/training_config.yaml \
    --resume-from outputs/checkpoint-500
```

---

## 📏 Evaluation

### Metrics

- **Execution Accuracy** — Execute both predicted and gold SQL against SQLite; compare result sets
- **Exact String Match** — Normalized SQL string comparison
- **BLEU Score** — N-gram overlap between predicted and gold SQL
- **Valid SQL Rate** — % of predictions that parse without syntax errors
- **Error Categorization** — Breakdown of failure modes (syntax, wrong table, wrong column, logic, etc.)

```bash
# Full evaluation with all metrics
python -m src.evaluate.evaluate_model \
    --adapter-path outputs/checkpoint-best \
    --test-split data/processed/test.jsonl \
    --run-execution-accuracy

# Compare base model vs fine-tuned
python -m src.evaluate.compare_models \
    --base-model "mistralai/Mistral-7B-v0.3" \
    --adapter-path outputs/checkpoint-best \
    --num-samples 200
```

---

## 🖥️ Inference & Demo

### Single Query

```python
from src.inference.predict import SQLPredictor

predictor = SQLPredictor(adapter_path="outputs/checkpoint-best")

result = predictor.predict(
    question="What are the top 5 departments by average salary?",
    schema="CREATE TABLE employees (id INT, name TEXT, salary REAL, department TEXT);"
)
print(result)
# SELECT department, AVG(salary) as avg_salary FROM employees
# GROUP BY department ORDER BY avg_salary DESC LIMIT 5;
```

### Gradio Demo

```bash
python -m src.inference.serve
```

Launches an interactive web UI where you can:
- Input natural language questions
- Paste or select a database schema
- See the generated SQL with syntax highlighting
- Compare base model vs. fine-tuned output side-by-side

---

## 📁 Project Structure

```
llm-lora-finetuning/
├── configs/
│   └── training_config.yaml        # All hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py      # Download, clean, split, save
│   │   └── prompt_templates.py     # Prompt formatting for different models
│   ├── train/
│   │   ├── finetune_lora.py        # Main QLoRA training script
│   │   └── callbacks.py            # Custom W&B + early stopping callbacks
│   ├── evaluate/
│   │   ├── evaluate_model.py       # All eval metrics
│   │   └── compare_models.py       # Base vs. fine-tuned comparison
│   └── inference/
│       ├── predict.py              # Programmatic inference
│       └── serve.py                # Gradio web demo
├── notebooks/
│   └── 01_exploration_and_training.ipynb
├── scripts/
│   ├── train.sh                    # One-click training launcher
│   └── evaluate.sh                 # One-click evaluation
├── tests/
│   └── test_data_pipeline.py       # Unit tests for data processing
├── assets/                         # Screenshots, diagrams
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

All training parameters live in [`configs/training_config.yaml`](configs/training_config.yaml). Key sections:

```yaml
model:
  name: "mistralai/Mistral-7B-v0.3"   # Swap model here
  max_seq_length: 1024

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
```

See the full config file for all options.

---

## 💻 Hardware Requirements

| Setup | VRAM | Training Time (est.) | Notes |
|-------|------|---------------------|-------|
| 1× A100 (40 GB) | ~18 GB | ~1.5 hrs | Recommended |
| 1× RTX 4090 (24 GB) | ~20 GB | ~2.5 hrs | Works great |
| 1× RTX 3090 (24 GB) | ~22 GB | ~3.5 hrs | Reduce batch size if OOM |
| 1× T4 (16 GB) | ~14 GB | ~6 hrs | Colab free tier — reduce seq length to 512 |
| CPU only | 32+ GB RAM | ~days | Not recommended; for testing only |

---

## 🙏 Acknowledgements

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Dettmers et al.
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Hu et al.
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [sql-create-context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)
- [Spider Benchmark](https://yale-lily.github.io/spider)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://github.com/visheshgupta29">Vishesh Gupta</a> · 
  ⭐ Star this repo if you find it useful!
</p>

# 🧬 LLM LoRA Fine-Tuning for Text-to-SQL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracked-orange)](https://wandb.ai/)

Fine-tune **Mistral-7B-v0.3** (and other open-source LLMs) for **Natural Language to SQL** generation using **QLoRA** (4-bit quantization + Low-Rank Adaptation). This project documents the **full iterative journey** — from a broken first run to a working pipeline — on the **sql-create-context** dataset (WikiSQL + Spider), using free-tier Kaggle T4 GPUs, with evaluation, experiment tracking, and a deployable Gradio demo.

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
    - [v1 — Baseline (1 epoch, 20K samples)](#v1--baseline-1-epoch-20k-samples)
    - [v2 — Improved Decoding (training complete · evaluated ✅)](#v2--improved-decoding-training-complete--evaluated-)
    - [v3 — Fixing the Stopping Condition (retraining in progress 🔧)](#v3--fixing-the-stopping-condition-retraining-in-progress-)
  - [🧭 Project Journey](#-project-journey)
    - [Why This Project](#why-this-project)
    - [Iteration Log](#iteration-log)
    - [Key Lessons](#key-lessons)
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

- **QLoRA (4-bit)** — Fine-tune a 7B-parameter model on a single free-tier **T4 GPU** (Kaggle/Colab)
- **Text-to-SQL** — Practical, real-world task connecting to enterprise NL2SQL applications
- **Iterative Journey** — Documented end-to-end: from broken first run → v1 (8% exec acc) → v2 (22% exec acc, +14pp) → v3 (retraining with fixed data)
- **Full Pipeline** — Data prep → Training → Evaluation → Inference → Gradio Demo
- **Experiment Tracking** — Weights & Biases integration with loss curves and eval metrics
- **Rigorous Evaluation** — Execution accuracy (SQL against SQLite), BLEU, exact-match, and error categorization
- **Multi-Model Support** — Config-driven; swap Mistral for Llama 3.1 8B, CodeLlama, Phi-3, or Qwen2
- **Deployable** — LoRA adapter → Hugging Face Hub → Gradio demo

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QLoRA Fine-Tuning                        │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌───────────────────────────┐ │
│  │ Dataset   │───▶│  Tokenizer │───▶│  Mistral-7B (4-bit NF4)  │ │
│  │ (SQL-     │    │  + Prompt  │    │  + LoRA Adapters (r=16)  │ │
│  │  Create-  │    │  Template  │    │  Trainable: ~1.1% params │ │
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

### v1 — Baseline (1 epoch, 20K samples)

> Trained on Kaggle T4 GPU · Mar 15–16, 2026 · `training_config_t4.yaml`

| Metric | Value |
|--------|-------|
| **Execution Accuracy** | **8.0%** |
| Exact Match | 0.0% |
| Valid SQL Rate | 100.0% |
| Avg BLEU | 0.0815 |
| Train Loss | 0.07483 |
| Best Eval Loss | 0.04429 (step 600, epoch 0.48) |
| Trainable Params | 41,943,040 (~42M) · 1.10% of 3.8B quantized params |
| Training Time | ~6h 52min on Kaggle T4 (24,720 s) |
| Peak VRAM | ~5.4 GB peak allocated |

**Error Distribution (v1):**

| Error Type | Count | % |
|------------|-------|---|
| `runtime_error` (repetitive SQL) | 72 | 72% |
| `wrong_column` | 19 | 19% |
| `logic_error` | 9 | 9% |

**Diagnosis:** The model learned valid SQL syntax (100% parseable!) but generates endlessly repetitive `AND` clauses. Root cause: weak repetition penalty (1.1) + sampling-based decoding + only 1 epoch of training.

<details>
<summary>🔍 Example: v1 failure mode (click to expand)</summary>

```
Q: When Essendon played away; where did they play?

Gold: SELECT venue FROM table_name_50 WHERE away_team = "essendon"

Pred: SELECT venue FROM table_name_50 WHERE away_team = "essendon" AND
      home_team = "melbourne" AND score = "essendon 18 melbourne 18" AND
      home_team = "melbourne" AND away_team = "essendon" AND ...
      (repeats until max_new_tokens)
```

The model gets the right start (`SELECT venue ... WHERE away_team = "essendon"`) but keeps generating redundant conditions.

</details>

### v2 — Improved Decoding (training complete · evaluated ✅)

> Trained on Kaggle T4 GPU · Mar 18–19, 2026 · `training_config_t4.yaml` · WandB run `nka1xolh`

**Changes applied for v2:**
- Decoding: `do_sample: true` → **`do_sample: false`** (deterministic greedy)
- Repetition penalty: 1.1 → **1.3** (penalise repeated tokens harder)
- Max new tokens: 256 → **128** (SQL queries rarely exceed 128 tokens)
- NEFTune noise: `neftune_noise_alpha: 5` (embedding noise regularisation)
- Epochs: 1 → **3** (but early stopping triggered at epoch 0.96)

**Training result — early stopped at step 1,200 / 3,750 (epoch 0.96):**

| Step | Epoch | Eval Loss | Notes |
|------|-------|-----------|-------|
| 200 | 0.16 | **0.05831** | ✅ Best — checkpoint saved |
| 400 | 0.32 | 0.05948 | patience 1/5 |
| 600 | 0.48 | 0.05878 | patience 2/5 |
| 800 | 0.64 | 0.06115 | patience 3/5 |
| 1000 | 0.80 | 0.06497 | patience 4/5 |
| 1200 | 0.96 | 0.11050 | patience 5/5 → 🛑 stopped |

| Training Metric | v1 | v2 |
|-----------------|----|----|
| Best Eval Loss | 0.04429 (step 600) | **0.05831 (step 200)** |
| Train Loss (avg) | 0.07483 | 0.09684 |
| Epochs Completed | 1.0 | 0.96 (early stopped) |
| Training Time | ~6h 52min (24,720 s) | ~6h 43min (24,200 s) |
| Peak VRAM | ~5.4 GB | ~5.4 GB |

| Eval Metric | v1 | v2 | Δ |
|-------------|----|----|---|
| Execution Accuracy | 8.0% | **22.0%** | **+14.0pp** 🎉 |
| Valid SQL Rate | 100.0% | 100.0% | ±0 |
| Avg BLEU | 0.0815 | **0.0962** | +0.0147 |
| Exact Match | 0.0% | 0.0% | ±0 |

**v2 error distribution (100 samples):**

| Error Type | Count | Notes |
|------------|-------|-------|
| `runtime_error` | 65 | Repetition loops — model repeats `AND col = "val"` until truncated |
| `logic_error` | 24 | Runs but returns wrong rows |
| `wrong_column` | 10 | Correct table, wrong column |
| `wrong_table` | 1 | Wrong table entirely |

> **Key finding (v1 → v2):** Greedy decoding alone boosted exec accuracy from 8% → 22% (+14pp). But 65% of failures are still repetition loops — `repetition_penalty: 1.3` penalises individual tokens but not repeated multi-token WHERE conditions.

**Base model comparison (200 samples, Mar 21, 2026):**

| Metric | Base Mistral-7B-v0.3 | Fine-Tuned v2 | Δ |
|--------|---------------------|---------------|---------|
| Execution Accuracy | **28.5%** | 21.0% | **−7.5pp** ⚠️ |
| Avg BLEU | **0.199** | 0.100 | **−0.099** |
| Valid SQL Rate | 100.0% | 100.0% | ±0 |
| Exact Match | 0.0% | 0.0% | ±0 |

> **Surprise:** The fine-tuned model currently underperforms the base model. But this is almost entirely a repetition problem, not a quality problem. In 4 of 5 comparison examples, the fine-tuned model generates the **exact correct SQL as its first output** — then corrupts it with repeated conditions. Example: gold is `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"` — the fine-tuned model starts with exactly that, then appends `AND nat = "sco" AND nat = "sco"...` 30+ times. Truncate at the first complete query and it would win. The base model generates shorter outputs (closer to gold length) which boosts its BLEU, and avoids repetition because it has no learned tendency to loop. **Fixing the repetition loop in v3 should push fine-tuned well above base.**

### v3 — Fixing the Stopping Condition (retraining in progress 🔧)

> Requires retraining · Kaggle T4 GPU · Mar 23, 2026 · `training_config_t4.yaml`

**Root cause discovery:** v3 started as inference-only fixes (stronger `repetition_penalty`, `no_repeat_ngram_size`, post-processing) — all of which made things *worse* or produced gibberish. After extensive debugging, the real root cause emerged:

1. **Training data had no semicolons.** SQL completions were stored as `SELECT col FROM tbl WHERE x = "y"` without a trailing `;`. The model never learned to generate `;` as a stopping signal.
2. **The `;` stop token never fired.** We added `;` as `eos_token_id`, but since the model was never trained to produce it, generation continued past the correct SQL until `max_new_tokens`.
3. **`no_repeat_ngram_size=4` causes gibberish.** SQL has valid repeated 4-grams (`= "val" AND`) that this constraint blocks, forcing the model into garbage tokens (`↑↑↑`, `[a]`).
4. **`repetition_penalty > 1.1` causes gibberish.** It penalizes tokens from the prompt (table names, column names, SQL keywords) that the model *must* reuse.

**Fix applied:**

| Change | File | Detail |
|--------|------|--------|
| Append `;` to training SQL | `src/data/prepare_dataset.py` | Every SQL completion now ends with `;` — model learns to stop there |
| Semicolon stop token | `src/evaluate/evaluate_model.py`, `src/inference/predict.py` | `;` added to `eos_token_id` — now actually fires since model generates `;` |
| Reset rep penalty | all inference code | `repetition_penalty: 1.1` — higher values block valid SQL tokens |
| Remove ngram blocking | all inference code | `no_repeat_ngram_size` removed entirely |
| Reduce epochs | `configs/training_config_t4.yaml` | `num_train_epochs: 1` (3 epochs caused severe overfitting: train_loss=0.006) |
| Prompt instruction | `src/data/prompt_templates.py` | "Generate a single SQL query. Do not repeat conditions." |

| Eval Metric | Base | v2 | v3 | Δ (v3 vs base) |
|-------------|------|----|----|------------------|
| Execution Accuracy | 28.5% | 21.0% | 🔧 | — |
| Valid SQL Rate | 100.0% | 100.0% | 🔧 | — |
| Avg BLEU | 0.199 | 0.100 | 🔧 | — |
| Exact Match | 0.0% | 0.0% | 🔧 | — |

> *v3 requires retraining with the fixed dataset. Results pending.*

<details>
<summary>📈 Training Loss Curve (click to expand)</summary>

![Training Loss](assets/v1-train-loss-curve.png)

</details>

---

## 🧭 Project Journey

### Why This Project

I wanted to go beyond "run someone else's notebook" and build an **end-to-end LLM fine-tuning pipeline from scratch** — something I could talk about in depth during interviews. Text-to-SQL was the perfect task: it's a real enterprise use case, the evaluation is objective (you can execute the SQL), and the dataset is publicly available.

The goal was never just "get high accuracy" — it was to **understand every piece**: quantization, LoRA math, tokenizer quirks, trainer internals, evaluation pitfalls, and what breaks when you move from a tutorial to real hardware.

### Iteration Log

<details open>
<summary><strong>🔧 v0 — Getting it to run at all</strong></summary>

The first attempt on Kaggle hit **6 distinct errors** before training even started:

| # | Error | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | `AttributeError: total_mem` | PyTorch API change — `total_mem` → `total_memory` | Updated VRAM check code |
| 2 | `ImportError: FlashAttention2` | Flash Attention not available on T4 | Switched to `attn_implementation="eager"` |
| 3 | `KeyError: 'lora_alpha'` | Config key mismatch (`alpha` vs `lora_alpha`) | Fixed YAML → Python key mapping |
| 4 | `TypeError: unexpected kwarg` | TRL version dropped `max_seq_length`/`packing` from SFTConfig | Removed unsupported args |
| 5 | `ValueError: --tf32` | TF32 requires Ampere+ GPU (T4 is Turing) | Set `tf32: false` in T4 config |
| 6 | `KeyError: 'completion'` | TRL SFTTrainer expected `prompt`+`completion` fields | Added `completion` field to dataset prep |

**Lesson:** Moving from an A100 tutorial to a free-tier T4 is 80% debugging environment differences, 20% actual ML.

</details>

<details open>
<summary><strong>📊 v1 — First successful training run</strong></summary>

**Config:** 1 epoch · 20K samples · Kaggle T4 · ~6h 52min

**Results:**
- ✅ Valid SQL: 100% — the model learned SQL syntax
- ⚠️ Execution accuracy: 8% — but it can't produce *correct* SQL
- ❌ Exact match: 0% — repetitive generation ruins every prediction

**What I learned:**
1. 100% valid SQL after just 1 epoch is actually a strong signal — the model has the right idea
2. The repetition problem isn't a model quality issue, it's a **decoding configuration issue**
3. `repetition_penalty: 1.1` is far too weak for SQL generation — the model loops `AND col = "val"` endlessly
4. Using `do_sample: true` at eval time adds randomness that hurts structured output tasks

</details>

<details open>
<summary><strong>🚀 v2 — Fixing decoding + early stopping (training complete · evaluated ✅)</strong></summary>

**Config:** Target 3 epochs · 20K samples · Kaggle T4 · Mar 18–19, 2026 · WandB run `nka1xolh`

**Changes applied:**
- Deterministic greedy decoding (`do_sample: false`)
- Stronger repetition penalty (1.1 → 1.3)
- Shorter max generation (256 → 128 tokens)
- NEFTune noise (`neftune_noise_alpha: 5`)

**What happened — early stopped at epoch 0.96 (step 1,200 / 3,750):**

| Step | Epoch | Eval Loss | Notes |
|------|-------|-----------|-------|
| 200 | 0.16 | **0.05831** | ✅ New best — checkpoint saved |
| 400 | 0.32 | 0.05948 | patience 1/5 |
| 600 | 0.48 | 0.05878 | patience 2/5 |
| 800 | 0.64 | 0.06115 | patience 3/5 |
| 1000 | 0.80 | 0.06497 | patience 4/5 |
| 1200 | 0.96 | 0.11050 | patience 5/5 → 🛑 stopped |

**Eval results (100 samples, Mar 19, 2026):**

| Metric | v1 | v2 | Δ |
|--------|----|----|---|
| Execution Accuracy | 8.0% | **22.0%** | **+14pp** 🎉 |
| Valid SQL Rate | 100.0% | 100.0% | ±0 |
| Avg BLEU | 0.0815 | **0.0962** | +0.0147 |
| Exact Match | 0.0% | 0.0% | ±0 |

**Error breakdown:** `runtime_error` 65 · `logic_error` 24 · `wrong_column` 10 · `wrong_table` 1

**Key observations:**
- **The good:** Greedy decoding alone drove a nearly 3× improvement vs v1 (8% → 22%)
- **The surprise:** Fine-tuned v2 (21.0%) underperforms the base model (28.5%) on 200 samples — not because it learned wrong things, but because repetition loops corrupt every prediction. Base model avoids this because it has no trained tendency to loop
- **Critical nuance:** In 4/5 comparison examples, the fine-tuned model generates the *exact correct SQL* as its first output before looping. `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"` — perfect — then `AND nat = "sco" AND nat = "sco"...` 30+ more times. The SQL knowledge is there; the stopping condition is not
- **Training insight:** Best generalisation at epoch 0.16 — early stopping saved ~13+ hours of wasted compute
- **Next:** v3 targets the root cause — training data needs `;` at end of completions so the model learns to stop; `;` as `eos_token_id` then actually fires

</details>

<details open>
<summary><strong>🛠️ v3 — Fixing the stopping condition (retraining in progress 🔧)</strong></summary>

**Motivation:** Fine-tuned v2 (21.0% exec acc) currently underperforms the base model (28.5%) solely because of repetition loops. The fine-tuned model demonstrably knows the correct SQL — it just doesn't know when to stop.

**What we tried first (all failed):**

| Approach | Result | Why it failed |
|----------|--------|---------------|
| `no_repeat_ngram_size=4` | 0% exec acc, gibberish | Blocks valid SQL patterns like `= "val" AND` |
| `repetition_penalty=1.5` | 0% exec acc, gibberish | Penalizes prompt tokens the model must reuse |
| `;` post-processing split | Unfair comparison | Only helps fine-tuned model, not base |
| WHERE deduplication | Unfair comparison | Modifies output asymmetrically |
| 3 epochs training | Severe overfitting | train_loss=0.006, 5× gap to eval_loss |

**Root cause discovered:** Training data completions had **no semicolons**. The model never learned to generate `;`, so the `;` stop token never fired, and the model had no natural stopping point.

**The fix:**
- Append `;` to every SQL completion in training data (`prepare_dataset.py`)
- Keep `;` as `eos_token_id` — now it actually fires because the model learns to produce `;`
- Reset `repetition_penalty` to 1.1 (safe value)
- Remove `no_repeat_ngram_size` entirely
- Reduce epochs from 3 → 1 (prevent overfitting)
- Prompt instruction: "Generate a single SQL query. Do not repeat conditions."

*Retraining required with fixed dataset — results pending.*

</details>

### Key Lessons

1. **Environment portability is non-trivial.** A config that works on A100 needs 6+ changes for T4. Build configs per hardware tier from the start.
2. **100% valid SQL ≠ correct SQL.** Syntax is easy; semantics is hard. Evaluation must include execution accuracy.
3. **Decoding matters as much as training.** The same model checkpoint can go from 8% → 22% execution accuracy just by switching to greedy decoding and increasing repetition penalty. Inference config is not an afterthought.
4. **Start small, validate, iterate.** Running 1 epoch on 20K samples first — instead of 3 epochs on 78K — saved hours and surfaced all the real issues early.
5. **Document the failures.** The error log above is more valuable in an interview than the final accuracy number.
6. **Always compare against the base model.** Fine-tuning can make things worse — v2 (21% exec acc) underperforms the base Mistral-7B (28.5%) because repetition loops corrupt otherwise-correct predictions. Without a baseline comparison you would never know the regression happened.
7. **Train with your stop signal.** If you use `;` as `eos_token_id`, the training data must contain `;` at the end of every completion. Otherwise the model never learns to generate the stop token, and generation runs to `max_new_tokens` every time.

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

| Parameter | A100 Config | T4 Config | Rationale |
|-----------|------------|-----------|----------|
| Quantization | NF4 (4-bit) | NF4 (4-bit) | Best quality for 4-bit per QLoRA paper |
| Compute dtype | bfloat16 | **float16** | T4 lacks bfloat16 support |
| LoRA Rank (r) | 16 | 16 | Good accuracy/efficiency tradeoff |
| LoRA Alpha | 32 | 32 | Standard α = 2r scaling |
| LoRA Dropout | 0.05 | 0.05 | Light regularization |
| Target Modules | All 7 linear layers | All 7 linear layers | Maximum quality |
| Learning Rate | 2e-4 | 2e-4 | Standard for QLoRA |
| LR Schedule | Cosine | Cosine | Smooth decay |
| Batch Size | 4 (eff. 16) | **2 (eff. 16)** | Halved batch, doubled grad accum |
| Max Seq Length | 1024 | **512** | Most SQL fits in 512 tokens |
| Epochs | 3 | **1** (v3) | 3 epochs caused severe overfitting; back to 1 |
| Precision | bf16 | **fp16** | T4 hardware constraint |
| TF32 | true | **false** | Ampere-only feature |

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
│   ├── training_config.yaml        # A100/high-end GPU config
│   └── training_config_t4.yaml     # T4/free-tier config (Kaggle/Colab)
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
│   ├── 00_free_tier_quickstart.ipynb  # One-click Kaggle/Colab pipeline
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
| 1× A100 (40 GB) | ~18 GB | ~1.5 hrs (3 epochs, full dataset) | Recommended |
| 1× RTX 4090 (24 GB) | ~20 GB | ~2.5 hrs | Works great |
| 1× RTX 3090 (24 GB) | ~22 GB | ~3.5 hrs | Reduce batch size if OOM |
| **1× T4 (16 GB)** 🆓 | **~5.4 GB peak** | **v1: ~6h 52min (1 ep) · v2: ~6h 43min (early stopped) · v3: pending** | **Kaggle/Colab free tier — tested ✅** |
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

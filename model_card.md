---
library_name: peft
license: mit
base_model: mistralai/Mistral-7B-v0.3
datasets:
- b-mc2/sql-create-context
language:
- en
pipeline_tag: text-generation
tags:
- text-to-sql
- qlora
- lora
- mistral
- sql
- fine-tuned
model-index:
- name: mistral-7b-text2sql-qlora
  results:
  - task:
      type: text-generation
      name: Text-to-SQL
    dataset:
      name: sql-create-context
      type: b-mc2/sql-create-context
    metrics:
    - type: accuracy
      value: 94.0
      name: Execution Accuracy (%)
    - type: bleu
      value: 0.923
      name: BLEU
    - type: exact_match
      value: 74.0
      name: Exact Match (%)
---

# Mistral-7B Text-to-SQL (QLoRA)

A **QLoRA fine-tuned** adapter for [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) that converts natural language questions into SQL queries.

Trained on the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset (WikiSQL + Spider, 20K examples) using a single **Kaggle T4 GPU** in **4h 45min**.

## Results

| Metric | Base Mistral-7B | This Adapter (v3) | Delta |
|--------|----------------|-------------------|-------|
| **Execution Accuracy** | 57.5% | **94.0%** | **+36.5pp** |
| Exact Match | 1.0% | **74.0%** | +73pp |
| Avg BLEU | 0.428 | **0.923** | +0.495 |
| Valid SQL Rate | 100.0% | 100.0% | ±0 |

Evaluated on 200 samples with SQL executed against SQLite.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model = "mistralai/Mistral-7B-v0.3"
adapter = "visheshgupta29/mistral-7b-text2sql-qlora"

# Load quantized base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter)

# Build prompt
prompt = """### Task: Generate a single SQL query to answer the following question.
Do not repeat any conditions or output multiple queries.

### Database Schema:
CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL);

### Question:
What is the average salary per department?

### SQL Query:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Find all ';' token variants for stop condition
semicolon_ids = set(tokenizer.encode(";", add_special_tokens=False))
for tok_str, tok_id in tokenizer.get_vocab().items():
    if tok_str.replace("\u2581", "").strip() == ";":
        semicolon_ids.add(tok_id)
stop_ids = list({tokenizer.eos_token_id} | semicolon_ids)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=stop_ids,
    )

generated = output[0][inputs["input_ids"].shape[1]:]
sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
if ";" in sql:
    sql = sql.split(";")[0].strip()
print(sql)
# SELECT department, AVG(salary) FROM employees GROUP BY department
```

Or use the project's `SQLPredictor` class:

```python
from src.inference.predict import SQLPredictor

predictor = SQLPredictor(adapter_path="visheshgupta29/mistral-7b-text2sql-qlora")
sql = predictor.predict(
    question="What is the average salary per department?",
    schema="CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL);"
)
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Mistral-7B-v0.3 |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down proj |
| Trainable params | 42M (1.1% of 3.8B) |
| Dataset | sql-create-context (20K train) |
| Epochs | 1 |
| Batch size | 2 (effective 16 via grad accum) |
| Optimizer | Paged AdamW 8-bit |
| LR | 2e-4, cosine schedule |
| Train loss | 0.038 |
| Eval loss | 0.025 |
| Training time | 4h 45min on Kaggle T4 |
| Peak VRAM | 5.37 GB |

## Key Design Decisions

1. **Semicolon in training data** — Every SQL completion ends with `;` so the model learns to stop generating. Without this, the model repeats conditions endlessly.
2. **SentencePiece vocab scan** — `tokenizer.encode(";")` returns `▁;` but the model generates bare `;` (different token ID). We scan `get_vocab()` for all variants.
3. **`repetition_penalty=1.1`** — Higher values (1.3, 1.5) cause gibberish on SQL because they penalize tokens from the prompt that must be reused.
4. **No `no_repeat_ngram_size`** — SQL has valid repeated n-grams like `= "val" AND` that this constraint blocks.

## Full Project

See the [GitHub repository](https://github.com/visheshgupta29/llm-lora-finetuning) for the complete pipeline: data prep, training, evaluation, comparison, and Gradio demo.

## License

MIT

# CLAUDE.md — Project Intelligence for AI Assistants

> This file is read by Claude Code, Claude in VS Code, and other AI assistants
> to understand the project deeply before making changes.

---

## 1. Project Identity

**What:** QLoRA fine-tuning pipeline for Text-to-SQL (Mistral-7B-v0.3 → SQL generation)
**Who:** Vishesh Gupta — personal portfolio project
**Where:** Runs on Kaggle T4 GPUs (free tier, 16 GB VRAM). Dev/editing happens on Windows/VS Code.
**Repo:** `visheshgupta29/llm-lora-finetuning`

This is an **iterative learning project** — the README documents every failure, not just successes.
The journey (v0 → v1 → v2 → v3) is as important as the final accuracy number.

---

## 2. Architecture Overview

```
src/
├── data/
│   ├── prepare_dataset.py      # Downloads b-mc2/sql-create-context, formats, splits, saves JSONL
│   └── prompt_templates.py     # Model-specific prompt formatting (Mistral, Llama3, ChatML, generic)
├── train/
│   ├── finetune_lora.py        # Main training script — QLoRA with SFTTrainer (TRL)
│   └── callbacks.py            # WandB metrics logger + EarlyStoppingOnPlateau
├── evaluate/
│   ├── evaluate_model.py       # Exec accuracy, BLEU, exact match, error categorization
│   └── compare_models.py       # Side-by-side base vs fine-tuned comparison
└── inference/
    ├── predict.py              # SQLPredictor class for programmatic inference
    └── serve.py                # Gradio web demo
```

**Config-driven:** Everything flows from `configs/training_config_t4.yaml`. The YAML file controls
model name, quantization, LoRA params, training hyperparams, inference settings, and eval config.

**Entry points are always `python -m src.<module>`**, not direct file execution.

---

## 3. Critical Conventions

### Code Style
- Python 3.10+ — uses `X | None` union syntax, not `Optional[X]`
- `from __future__ import annotations` at top of every module
- Logging: `logging` + `rich.console.Console` for pretty terminal output
- Docstrings: Google/NumPy hybrid style with `Parameters` / `Returns` sections
- All modules have a docstring with `Usage:` examples showing CLI invocation

### Module Imports
- Always import from `src.*` (package-style): `from src.data.prompt_templates import format_prompt`
- Never relative imports
- `compare_models.py` imports `generate_sql` from `evaluate_model.py` — changes to generation logic
  in evaluate automatically propagate to comparisons

### Configuration
- All config lives in YAML files under `configs/`
- `training_config.yaml` = A100/high-end config
- `training_config_t4.yaml` = T4/free-tier config (the one actually used)
- CLI args can override YAML values via `merge_cli_overrides()` in `finetune_lora.py`
- The YAML header comments contain the full experiment history — keep them updated

### Data Flow
```
HuggingFace Dataset → prepare_dataset.py → data/processed/{train,test}.jsonl
                                                    ↓
                                        finetune_lora.py (reads JSONL)
                                                    ↓
                                        outputs/final-adapter (LoRA weights)
                                                    ↓
                                    evaluate_model.py / compare_models.py / predict.py
```

### Prompt Template System
- `prompt_templates.py` has a registry `_TEMPLATE_REGISTRY` mapping model families to formatters
- `get_formatter(model_name)` inspects the model name string to auto-detect the right template
- `format_prompt()` is the convenience entry point — all other modules call this
- The generic template (used for base Mistral) has the instruction: "Generate a single SQL query"
- **Training completions end with `;`** — this is the stop signal the model learns

### Generation Parameters (CRITICAL — hard-won lessons)
- **`eos_token_id`** includes ALL `;` token IDs — must scan `tokenizer.get_vocab()` for variants
  because SentencePiece encodes standalone `";"` as `▁;` (different token ID from bare `";"` in context)
- **`repetition_penalty`** must stay at **1.1** — higher values (1.3, 1.5) cause gibberish
  because they penalize tokens from the prompt that the model MUST reuse (table names, column names)
- **`no_repeat_ngram_size`** must NOT be used — SQL has valid repeated n-grams (`= "val" AND`)
  and blocking them forces the model into garbage tokens (↑↑↑, [a], unicode symbols)
- **`do_sample: false`** for evaluation (deterministic greedy decoding)
- These lessons cost ~20+ hours of GPU time to discover. Do not revert them.

---

## 4. Current State (keep this updated)

### Branch: `feat/v3-inference-optimization`

**Status:** v3 training + eval complete. 🎉 **93% execution accuracy.**

**What happened:**
- v1: 8% exec acc (repetition loops)
- v2: 22% exec acc (greedy decoding helped, but still loops)
- Base Mistral-7B: 57.5% exec acc (after fair code-fence stripping; was 28.5% in v2 era before fence fix)
- v3 attempt 1: 0% exec acc — `no_repeat_ngram_size=4` caused gibberish
- v3 attempt 2: 0% exec acc — `repetition_penalty=1.5` caused gibberish
- v3 attempt 3: 0% exec acc — 3 epochs caused severe overfitting (train_loss=0.006)
- v3 attempt 4: 0% exec acc — SentencePiece `;` token mismatch (stop token never fired)
- **v3 final: 93% exec acc (100 samples), 94% (200 samples), 74% exact match, 0.923 BLEU** 🎉

**Comparison (200 samples, Mar 24 2026):**
| Metric | Base | Fine-Tuned v3 | Δ |
|--------|------|---------------|----|
| Exec Accuracy | 57.5% | **94.0%** | **+36.5pp** |
| BLEU | 0.428 | **0.923** | +0.495 |
| Exact Match | 1.0% | **74.0%** | +73pp |

**Root causes found (three issues):**
1. Training data had no semicolons → model never learned to generate `;` → fixed in `prepare_dataset.py`
2. SentencePiece encodes `";"` as `▁;` (space-prefix), but model generates bare `";"` (different token ID) → fixed by scanning `get_vocab()` for all `;` variants
3. Base model wraps SQL in `` ```sql...``` `` markdown code fences → SQLite syntax errors → fixed by stripping fences in `generate_sql()` for fair comparison

**v3 Results (100 samples, Mar 24 2026):**
| Metric | Value |
|--------|-------|
| Execution Accuracy | **93.0%** |
| Exact Match | 76.0% |
| Valid SQL Rate | 100.0% |
| BLEU | 0.921 |
| Train Loss | 0.038 |
| Training Time | 4h 45min |

### Key Files Modified in v3
| File | What Changed |
|------|-------------|
| `src/data/prepare_dataset.py` | `;` appended to all SQL completions; `gold_sql` stays WITHOUT `;` (for eval) |
| `src/evaluate/evaluate_model.py` | `;` as eos_token_id, rep_penalty=1.1 |
| `src/inference/predict.py` | Same generation param changes |
| `configs/training_config_t4.yaml` | epochs=1, rep_penalty=1.1, header comments updated |
| `src/data/prompt_templates.py` | Concise SQL instruction added |
| `README.md` | v3 root cause analysis, failed approaches table, lesson #7 |

---

## 5. Things That Will Bite You

### Do NOT:
- Set `repetition_penalty > 1.1` for SQL generation (gibberish output)
- Use `no_repeat_ngram_size` for SQL (blocks valid patterns)
- Train for 3+ epochs on 20K samples (severe overfitting, train_loss drops to 0.006)
- Remove the `;` from training completions (the model needs it to learn when to stop)
- Add `;` to `gold_sql` field (eval comparison strips trailing `;` via `normalize_sql`)
- Use `do_sample=True` at evaluation time (adds randomness, hurts structured output)
- Use `bf16=True` on T4 GPUs (T4 doesn't support bfloat16)
- Use `tf32=True` on T4 GPUs (TF32 is Ampere-only)
- Post-process only the fine-tuned model's output (unfair comparison vs base)

### Do:
- Keep `gold_sql` without `;` — `normalize_sql()` strips trailing `;` for comparison
- Use `format_prompt()` everywhere (not individual formatters) — it auto-detects model family
- Test with `python -m src.<module>` not `python src/<module>.py`
- Check the YAML header comments before modifying training config — they contain full history
- Keep the `text`, `prompt`, `completion`, `gold_sql` field structure in dataset examples
- Use `eos_token_id=stop_ids` which includes both the real EOS token and `;` token IDs

### Dataset Fields (from prepare_dataset.py)
```python
{
    "text": "full prompt + sql_with_stop",       # For SFT training
    "prompt": "prompt without SQL",              # For inference
    "completion": "sql_with_stop",               # SQL with ';' at end
    "question": "natural language question",     # Raw question
    "schema": "CREATE TABLE ...",                # Raw schema
    "gold_sql": "sql without ';'"               # For eval comparison
}
```

---

## 6. Running the Pipeline

### On Kaggle (training):
```bash
# Always use the T4 config
python -m src.data.prepare_dataset --config configs/training_config_t4.yaml
python -m src.train.finetune_lora --config configs/training_config_t4.yaml
python -m src.evaluate.evaluate_model \
    --adapter-path outputs/final-adapter \
    --test-split data/processed/test.jsonl \
    --num-samples 100 --run-execution-accuracy
python -m src.evaluate.compare_models \
    --base-model mistralai/Mistral-7B-v0.3 \
    --adapter-path outputs/final-adapter \
    --num-samples 200
```

### Locally (editing/testing only — no GPU):
```bash
python -m pytest tests/ -v          # Unit tests (no GPU needed)
python -m src.data.prepare_dataset  # Dataset prep (no GPU needed)
```

### Resume from checkpoint:
```bash
python -m src.train.finetune_lora \
    --config configs/training_config_t4.yaml \
    --resume-from outputs/checkpoint-1000
```

---

## 7. Environment

- **Python:** 3.10+
- **Key deps:** torch, transformers, datasets, peft, trl, bitsandbytes, wandb, gradio
- **Secrets:** `HF_TOKEN` and `WANDB_API_KEY` from `.env` file (loaded via `python-dotenv`)
- **Config:** `pip install -r requirements.txt` (or `pip install -e .` for editable)

---

## 8. Test Suite

Tests are in `tests/test_data_pipeline.py`. They test:
- Prompt template formatting for all model families
- JSONL serialization round-trips
- YAML config validity and required keys

**Known issue:** Tests import old function names (`format_prompt_generic`, `format_prompt_chatml`)
that may not match current `prompt_templates.py` exports. The function signatures evolved during
v3 development. If tests fail on imports, update the test imports to match current module exports.

Run: `python -m pytest tests/ -v`

---

## 9. Git Workflow

- **main branch:** Stable, PR-merged only
- **Feature branches:** `feat/<description>`, `docs/<description>`, `fix/<description>`
- **Current branch:** `feat/v3-inference-optimization`
- **Commit style:** Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **PR flow:** Create on GitHub → merge to main

### Files that should never be committed:
- `outputs/` (model checkpoints, eval results)
- `data/processed/` (generated JSONL files)
- `wandb/` (experiment logs)
- `.env` (secrets)
- `*.safetensors`, `*.bin`, `*.pt` (model weights)

---

## 10. Iteration History (TL;DR)

| Version | Exec Acc | What Happened |
|---------|----------|---------------|
| v0 | — | 6 errors before training even started (T4 compat) |
| v1 | 8% | Valid SQL but endless repetition loops |
| v2 | 22% | Greedy decoding helped; still loops; base model beats it (28.5% pre-fence-fix) |
| v3 (failed ×4) | 0% | `no_repeat_ngram_size` + high rep_penalty = gibberish; SentencePiece token mismatch |
| **v3 (final)** | **94%** | Fixed: `;` in training data + vocab scan for all `;` token IDs + code fence stripping. Base: 57.5% on same 200 samples. |

The full story with error tables, training curves, and comparison data is in `README.md`.

# Copilot Instructions — llm-lora-finetuning

## Project Context
QLoRA fine-tuning pipeline for Mistral-7B Text-to-SQL on Kaggle T4 GPUs.
Config-driven (`configs/training_config_t4.yaml`). All modules run via `python -m src.<module>`.

## Code Conventions
- Python 3.10+ with `from __future__ import annotations` in every file
- Type hints: `X | None` (not `Optional[X]`), `list[str]` (not `List[str]`)
- Imports: always `from src.*` (absolute), never relative
- Logging: `logging` module + `rich.console.Console` for formatted output
- Docstrings: Google/NumPy hybrid with `Parameters` / `Returns`
- Every module has a top-level docstring with `Usage:` CLI examples

## Architecture Rules
- `prompt_templates.py` → `format_prompt()` is the single entry point for all prompt formatting
- `compare_models.py` imports `generate_sql` from `evaluate_model.py` — do not duplicate generation logic
- Dataset examples have 6 fields: `text`, `prompt`, `completion`, `question`, `schema`, `gold_sql`
- `completion` has trailing `;` (stop signal). `gold_sql` does NOT (for eval comparison)
- `normalize_sql()` strips trailing `;` — so gold_sql and predicted SQL compare fairly

## Generation Parameters — DO NOT CHANGE
These were discovered through ~20 hours of failed GPU experiments:
- `repetition_penalty`: MUST be 1.1 — higher values (1.3, 1.5) cause gibberish on SQL
- `no_repeat_ngram_size`: MUST NOT be used — blocks valid SQL patterns like `= "val" AND`
- `eos_token_id`: includes both real EOS token AND `;` token IDs
- `do_sample: false` for evaluation (deterministic greedy decoding)

## Hardware Constraints (T4 GPU)
- `bf16: false` (T4 doesn't support bfloat16)
- `tf32: false` (TF32 is Ampere-only)
- `max_seq_length: 512` (saves ~2 GB VRAM vs 1024)
- `per_device_train_batch_size: 2` with `gradient_accumulation_steps: 8`

## Testing
- Tests in `tests/test_data_pipeline.py`
- Run: `python -m pytest tests/ -v`
- Test imports may be out of sync with current `prompt_templates.py` exports

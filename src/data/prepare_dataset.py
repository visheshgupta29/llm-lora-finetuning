"""
Dataset preparation pipeline.

Downloads the `b-mc2/sql-create-context` dataset from Hugging Face,
formats prompts, splits into train/eval, and saves as JSONL files.

Usage:
    python -m src.data.prepare_dataset
    python -m src.data.prepare_dataset --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from datasets import load_dataset, Dataset, DatasetDict
from rich.console import Console
from rich.table import Table

from src.data.prompt_templates import format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# ─── Defaults ────────────────────────────────────────────────────────
DEFAULT_DATASET = "b-mc2/sql-create-context"
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.3"
DEFAULT_TEST_SIZE = 0.1
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "data/processed"


def load_config(config_path: str | None) -> dict:
    """Load YAML config file, or return empty dict."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def download_and_prepare(
    dataset_name: str = DEFAULT_DATASET,
    model_name: str = DEFAULT_MODEL,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> DatasetDict:
    """Download dataset, format prompts, split, and save.

    Returns
    -------
    DatasetDict with 'train' and 'test' splits.
    """
    console.rule("[bold blue]Step 1: Download Dataset")
    logger.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    logger.info(f"Total examples: {len(ds):,}")

    # ── Show sample ──────────────────────────────────────────────────
    console.rule("[bold blue]Step 2: Inspect Sample")
    sample = ds[0]
    table = Table(title="Sample Row")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green", max_width=80)
    for key, value in sample.items():
        table.add_row(key, str(value)[:200])
    console.print(table)

    # ── Format prompts ───────────────────────────────────────────────
    console.rule("[bold blue]Step 3: Format Prompts")
    logger.info(f"Formatting prompts for model: {model_name}")

    def _format_example(example: dict) -> dict:
        """Format a single example into a training prompt."""
        # Dataset columns: question, context (CREATE TABLE), answer (SQL)
        question = example.get("question", "")
        schema = example.get("context", "")
        sql = example.get("answer", "")

        # Ensure SQL ends with ';' so model learns to stop there
        sql_with_stop = sql.strip().rstrip(";") + ";"

        # Full prompt with answer (for training)
        text = format_prompt(
            model_name=model_name,
            question=question,
            schema=schema,
            sql=sql_with_stop,
        )

        # Prompt without answer (for inference / eval)
        prompt_only = format_prompt(
            model_name=model_name,
            question=question,
            schema=schema,
            sql=None,
        )

        return {
            "text": text,
            "prompt": prompt_only,
            "completion": sql_with_stop,
            "question": question,
            "schema": schema,
            "gold_sql": sql,
        }

    ds = ds.map(
        _format_example,
        remove_columns=ds.column_names,
        desc="Formatting prompts",
        num_proc=4,
    )

    # ── Train/Test split ─────────────────────────────────────────────
    console.rule("[bold blue]Step 4: Train/Test Split")
    split_ds = ds.train_test_split(test_size=test_size, seed=seed)
    logger.info(
        f"Train: {len(split_ds['train']):,} | Test: {len(split_ds['test']):,}"
    )

    # ── Subsample if requested ───────────────────────────────────────
    if max_train_samples:
        split_ds["train"] = split_ds["train"].select(range(min(max_train_samples, len(split_ds["train"]))))
        logger.info(f"Subsampled train to {len(split_ds['train']):,}")
    if max_eval_samples:
        split_ds["test"] = split_ds["test"].select(range(min(max_eval_samples, len(split_ds["test"]))))
        logger.info(f"Subsampled test to {len(split_ds['test']):,}")

    # ── Save ─────────────────────────────────────────────────────────
    console.rule("[bold blue]Step 5: Save to Disk")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "test"]:
        file_path = output_path / f"{split_name}.jsonl"
        split_ds[split_name].to_json(str(file_path))
        logger.info(f"Saved {split_name} → {file_path} ({len(split_ds[split_name]):,} rows)")

    # ── Summary ──────────────────────────────────────────────────────
    console.rule("[bold green]✅ Dataset Ready")
    summary = Table(title="Dataset Summary")
    summary.add_column("Split", style="cyan")
    summary.add_column("Rows", style="green", justify="right")
    summary.add_column("File", style="blue")
    summary.add_row("Train", f"{len(split_ds['train']):,}", str(output_path / "train.jsonl"))
    summary.add_row("Test", f"{len(split_ds['test']):,}", str(output_path / "test.jsonl"))
    console.print(summary)

    # Show a formatted prompt example
    console.rule("[bold blue]Sample Formatted Prompt")
    console.print(split_ds["train"][0]["text"][:600] + "\n...")

    return split_ds


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for QLoRA fine-tuning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--model-name", type=str, default=None, help="Model name (for prompt format)")
    parser.add_argument("--test-size", type=float, default=None, help="Test split ratio")
    parser.add_argument("--max-train", type=int, default=None, help="Max training samples")
    parser.add_argument("--max-eval", type=int, default=None, help="Max eval samples")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Load config, then override with CLI args
    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    download_and_prepare(
        dataset_name=args.dataset or ds_cfg.get("name", DEFAULT_DATASET),
        model_name=args.model_name or model_cfg.get("name", DEFAULT_MODEL),
        test_size=args.test_size or ds_cfg.get("test_size", DEFAULT_TEST_SIZE),
        seed=ds_cfg.get("seed", DEFAULT_SEED),
        max_train_samples=args.max_train or ds_cfg.get("max_train_samples"),
        max_eval_samples=args.max_eval or ds_cfg.get("max_eval_samples"),
        output_dir=args.output_dir or ds_cfg.get("processed_dir", DEFAULT_OUTPUT_DIR),
    )


if __name__ == "__main__":
    main()

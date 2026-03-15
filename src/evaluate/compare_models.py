"""
Compare base model vs. fine-tuned model side-by-side.

Generates a comparison report showing predictions from both models
on the same test examples, with metrics for each.

Usage:
    python -m src.evaluate.compare_models \
        --base-model mistralai/Mistral-7B-v0.3 \
        --adapter-path outputs/final-adapter \
        --num-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from rich.console import Console
from rich.progress import track
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluate.evaluate_model import (
    compute_bleu,
    compute_execution_accuracy,
    generate_sql,
    is_valid_sql,
    normalize_sql,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_base_model(model_name: str):
    """Load base model (no LoRA adapter) for comparison."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compare(
    base_model_name: str,
    adapter_path: str,
    test_path: str = "data/processed/test.jsonl",
    num_samples: int = 50,
    output_file: str = "outputs/comparison_results.json",
):
    """Run side-by-side comparison of base vs. fine-tuned model."""

    # ── Load both models ─────────────────────────────────────────────
    console.rule("[bold blue]Loading Base Model")
    base_model, base_tokenizer = load_base_model(base_model_name)

    console.rule("[bold blue]Loading Fine-Tuned Model")
    ft_model, ft_tokenizer = load_finetuned_model(base_model_name, adapter_path)

    # ── Load test data ───────────────────────────────────────────────
    test_ds = load_dataset("json", data_files=test_path, split="train")
    test_ds = test_ds.select(range(min(num_samples, len(test_ds))))
    logger.info(f"Comparing on {len(test_ds):,} samples")

    # ── Generate from both ───────────────────────────────────────────
    console.rule("[bold blue]Running Comparison")
    results = []
    base_metrics = {"exact_match": 0, "valid_sql": 0, "bleu_sum": 0.0, "exec_acc": 0}
    ft_metrics = {"exact_match": 0, "valid_sql": 0, "bleu_sum": 0.0, "exec_acc": 0}

    for example in track(test_ds, description="Comparing models..."):
        prompt = example["prompt"]
        gold_sql = example["gold_sql"]
        schema = example["schema"]

        # Base model prediction
        base_pred = generate_sql(base_model, base_tokenizer, prompt)
        # Fine-tuned model prediction
        ft_pred = generate_sql(ft_model, ft_tokenizer, prompt)

        # Metrics for base
        base_em = normalize_sql(base_pred) == normalize_sql(gold_sql)
        base_valid = is_valid_sql(base_pred)
        base_bleu = compute_bleu(base_pred, gold_sql)
        base_exec = compute_execution_accuracy(base_pred, gold_sql, schema) if base_valid else False

        base_metrics["exact_match"] += int(base_em)
        base_metrics["valid_sql"] += int(base_valid)
        base_metrics["bleu_sum"] += base_bleu
        base_metrics["exec_acc"] += int(base_exec)

        # Metrics for fine-tuned
        ft_em = normalize_sql(ft_pred) == normalize_sql(gold_sql)
        ft_valid = is_valid_sql(ft_pred)
        ft_bleu = compute_bleu(ft_pred, gold_sql)
        ft_exec = compute_execution_accuracy(ft_pred, gold_sql, schema) if ft_valid else False

        ft_metrics["exact_match"] += int(ft_em)
        ft_metrics["valid_sql"] += int(ft_valid)
        ft_metrics["bleu_sum"] += ft_bleu
        ft_metrics["exec_acc"] += int(ft_exec)

        results.append({
            "question": example["question"],
            "gold_sql": gold_sql,
            "base_prediction": base_pred,
            "finetuned_prediction": ft_pred,
            "base_exact_match": base_em,
            "finetuned_exact_match": ft_em,
            "base_bleu": round(base_bleu, 4),
            "finetuned_bleu": round(ft_bleu, 4),
            "base_exec_accuracy": base_exec,
            "finetuned_exec_accuracy": ft_exec,
        })

    # ── Display comparison table ─────────────────────────────────────
    n = len(results)
    console.rule("[bold green]📊 Comparison Results")
    table = Table(title=f"Base vs. Fine-Tuned ({n} samples)")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Base ({base_model_name.split('/')[-1]})", style="red", justify="right")
    table.add_column("Fine-Tuned (QLoRA)", style="green", justify="right")
    table.add_column("Δ", style="yellow", justify="right")

    def _fmt(base_val, ft_val):
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        return f"{base_val:.1f}%", f"{ft_val:.1f}%", f"{sign}{delta:.1f}%"

    em_base = base_metrics["exact_match"] / n * 100
    em_ft = ft_metrics["exact_match"] / n * 100
    table.add_row("Exact Match", *_fmt(em_base, em_ft))

    valid_base = base_metrics["valid_sql"] / n * 100
    valid_ft = ft_metrics["valid_sql"] / n * 100
    table.add_row("Valid SQL Rate", *_fmt(valid_base, valid_ft))

    bleu_base = base_metrics["bleu_sum"] / n * 100
    bleu_ft = ft_metrics["bleu_sum"] / n * 100
    table.add_row("Avg BLEU (×100)", *_fmt(bleu_base, bleu_ft))

    exec_base = base_metrics["exec_acc"] / n * 100
    exec_ft = ft_metrics["exec_acc"] / n * 100
    table.add_row("Execution Accuracy", *_fmt(exec_base, exec_ft))

    console.print(table)

    # ── Show sample comparisons ──────────────────────────────────────
    console.rule("[bold blue]Sample Predictions (first 5)")
    for i, r in enumerate(results[:5]):
        console.print(f"\n[bold]Example {i+1}:[/bold] {r['question']}")
        console.print(f"  [green]Gold:[/green]       {r['gold_sql']}")
        console.print(f"  [red]Base:[/red]       {r['base_prediction']}")
        console.print(f"  [blue]Fine-tuned:[/blue] {r['finetuned_prediction']}")

    # ── Save ─────────────────────────────────────────────────────────
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_model": base_model_name,
        "adapter_path": adapter_path,
        "num_samples": n,
        "base_exact_match": round(em_base, 2),
        "finetuned_exact_match": round(em_ft, 2),
        "base_valid_sql_rate": round(valid_base, 2),
        "finetuned_valid_sql_rate": round(valid_ft, 2),
        "base_avg_bleu": round(base_metrics["bleu_sum"] / n, 4),
        "finetuned_avg_bleu": round(ft_metrics["bleu_sum"] / n, 4),
        "base_execution_accuracy": round(exec_base, 2),
        "finetuned_execution_accuracy": round(exec_ft, 2),
    }
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"✅ Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs. fine-tuned model")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-split", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-file", type=str, default="outputs/comparison_results.json")
    args = parser.parse_args()

    compare(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        test_path=args.test_split,
        num_samples=args.num_samples,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

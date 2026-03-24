"""
Evaluation script for the fine-tuned Text-to-SQL model.

Metrics computed:
    1. Execution Accuracy  — Run SQL against SQLite, compare result sets
    2. Exact String Match  — Normalized SQL comparison
    3. BLEU Score           — N-gram overlap
    4. Valid SQL Rate       — % of predictions that parse
    5. Error Categorization — Breakdown of failure types

Usage:
    python -m src.evaluate.evaluate_model \
        --adapter-path outputs/final-adapter \
        --test-split data/processed/test.jsonl \
        --num-samples 500 \
        --run-execution-accuracy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import torch
import sqlparse
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel
from rich.console import Console
from rich.progress import track
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# ═══════════════════════════════════════════════════════════════════════
#  SQL Utilities
# ═══════════════════════════════════════════════════════════════════════

def normalize_sql(sql: str) -> str:
    """Normalize SQL for string comparison.

    Lowercases, strips whitespace, removes trailing semicolons,
    and formats with sqlparse.
    """
    sql = sql.strip().rstrip(";").strip()
    sql = sqlparse.format(sql, reindent=False, keyword_case="lower", strip_comments=True)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def is_valid_sql(sql: str) -> bool:
    """Check if SQL parses without errors."""
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and parsed[0].tokens is not None
    except Exception:
        return False


def execute_sql_on_schema(schema: str, sql: str, timeout: float = 5.0) -> list | str:
    """Execute SQL against an in-memory SQLite database created from schema.

    Returns the result set as a list of tuples, or an error string.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            conn = sqlite3.connect(tmp.name, timeout=timeout)
            conn.execute("PRAGMA journal_mode=WAL;")

            # Create tables from schema
            for statement in sqlparse.split(schema):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)

            # Execute the query
            cursor = conn.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return results
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


def categorize_error(predicted_sql: str, gold_sql: str, schema: str) -> str:
    """Classify the error type for a wrong prediction."""
    if not is_valid_sql(predicted_sql):
        return "syntax_error"

    pred_result = execute_sql_on_schema(schema, predicted_sql)
    if isinstance(pred_result, str) and "ERROR" in pred_result:
        error_msg = pred_result.lower()
        if "no such table" in error_msg:
            return "wrong_table"
        if "no such column" in error_msg:
            return "wrong_column"
        return "runtime_error"

    # SQL is valid and runs, but returns wrong results
    pred_norm = normalize_sql(predicted_sql)
    gold_norm = normalize_sql(gold_sql)

    if "where" in gold_norm and "where" not in pred_norm:
        return "missing_where"
    if "join" in gold_norm and "join" not in pred_norm:
        return "missing_join"
    if "group by" in gold_norm and "group by" not in pred_norm:
        return "missing_groupby"
    if "order by" in gold_norm and "order by" not in pred_norm:
        return "missing_orderby"

    return "logic_error"


# ═══════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    adapter_path: str,
    base_model_name: str | None = None,
) -> tuple:
    """Load the base model + LoRA adapter for inference."""
    # Detect base model from adapter config
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists() and not base_model_name:
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path")

    if not base_model_name:
        raise ValueError("Cannot determine base model. Pass --base-model explicitly.")

    logger.info(f"Loading base model: {base_model_name}")
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

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_sql(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate SQL from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Stop generation at ';' (end of first complete SQL statement) or EOS.
    # SentencePiece encodes standalone ";" as ▁; (with space prefix), but in
    # context the model may emit bare ";" (different token ID).  Scan the
    # vocab so we catch EVERY variant.
    semicolon_ids = set(tokenizer.encode(";", add_special_tokens=False))
    for tok_str, tok_id in tokenizer.get_vocab().items():
        if tok_str.replace("\u2581", "").strip() == ";":
            semicolon_ids.add(tok_id)
    stop_ids = list({tokenizer.eos_token_id} | semicolon_ids)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=temperature > 0,
        repetition_penalty=1.1,                # v3: keep low — ';' stop token handles stopping
        eos_token_id=stop_ids,                 # v3: stop at first ';'
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Format cleanup only (not SQL modification)
    # Strip markdown code fences — base models often wrap SQL in ```sql...```
    if sql.startswith("```"):
        sql = sql.strip("`").strip()
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
    if sql.endswith("```"):
        sql = sql[: sql.rfind("```")].strip()
    if "###" in sql:
        sql = sql.split("###")[0].strip()
    if "\n\n" in sql:
        sql = sql.split("\n\n")[0].strip()
    # Safety net: take only the first complete statement if the stop token
    # missed a ';' variant.  Applied equally to all models (fair comparison).
    if ";" in sql:
        sql = sql.split(";")[0].strip()

    return sql


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_bleu(predicted: str, reference: str) -> float:
    """Compute BLEU score between predicted and reference SQL."""
    pred_tokens = normalize_sql(predicted).split()
    ref_tokens = normalize_sql(reference).split()

    if not ref_tokens:
        return 0.0

    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )


def compute_execution_accuracy(
    predicted: str,
    gold: str,
    schema: str,
) -> bool:
    """Check if predicted SQL returns the same result set as gold SQL."""
    pred_result = execute_sql_on_schema(schema, predicted)
    gold_result = execute_sql_on_schema(schema, gold)

    # If either errored, compare as strings
    if isinstance(pred_result, str) or isinstance(gold_result, str):
        return False

    # Compare result sets (order-insensitive)
    return sorted(map(str, pred_result)) == sorted(map(str, gold_result))


# ═══════════════════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════

def evaluate(
    adapter_path: str,
    test_path: str,
    base_model_name: str | None = None,
    num_samples: int | None = None,
    run_execution_accuracy: bool = False,
    output_file: str = "outputs/eval_results.json",
):
    """Run full evaluation pipeline."""
    # ── Load model ───────────────────────────────────────────────────
    console.rule("[bold blue]Loading Model")
    model, tokenizer = load_model_and_tokenizer(adapter_path, base_model_name)

    # ── Load test data ───────────────────────────────────────────────
    console.rule("[bold blue]Loading Test Data")
    test_ds = load_dataset("json", data_files=test_path, split="train")
    if num_samples:
        test_ds = test_ds.select(range(min(num_samples, len(test_ds))))
    logger.info(f"Evaluating on {len(test_ds):,} samples")

    # ── Generate predictions ─────────────────────────────────────────
    console.rule("[bold blue]Generating Predictions")
    results = []
    metrics = {
        "exact_match": 0,
        "valid_sql": 0,
        "execution_accuracy": 0,
        "bleu_scores": [],
        "error_types": Counter(),
    }

    for example in track(test_ds, description="Evaluating..."):
        prompt = example["prompt"]
        gold_sql = example["gold_sql"]
        schema = example["schema"]

        # Generate
        predicted_sql = generate_sql(model, tokenizer, prompt)

        # Exact match
        exact = normalize_sql(predicted_sql) == normalize_sql(gold_sql)
        metrics["exact_match"] += int(exact)

        # Valid SQL
        valid = is_valid_sql(predicted_sql)
        metrics["valid_sql"] += int(valid)

        # BLEU
        bleu = compute_bleu(predicted_sql, gold_sql)
        metrics["bleu_scores"].append(bleu)

        # Execution accuracy
        exec_acc = False
        if run_execution_accuracy and valid:
            exec_acc = compute_execution_accuracy(predicted_sql, gold_sql, schema)
            metrics["execution_accuracy"] += int(exec_acc)

        # Error categorization (if wrong)
        error_type = None
        if not exact:
            error_type = categorize_error(predicted_sql, gold_sql, schema)
            metrics["error_types"][error_type] += 1

        results.append({
            "question": example["question"],
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "exact_match": exact,
            "valid_sql": valid,
            "bleu": round(bleu, 4),
            "execution_accuracy": exec_acc if run_execution_accuracy else None,
            "error_type": error_type,
        })

    # ── Compute aggregates ───────────────────────────────────────────
    n = len(results)
    summary = {
        "num_samples": n,
        "exact_match_rate": round(metrics["exact_match"] / n * 100, 2),
        "valid_sql_rate": round(metrics["valid_sql"] / n * 100, 2),
        "avg_bleu": round(sum(metrics["bleu_scores"]) / n, 4),
        "error_distribution": dict(metrics["error_types"].most_common()),
    }
    if run_execution_accuracy:
        summary["execution_accuracy_rate"] = round(
            metrics["execution_accuracy"] / n * 100, 2
        )

    # ── Display ──────────────────────────────────────────────────────
    console.rule("[bold green]📊 Evaluation Results")
    table = Table(title="Metrics Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Samples", str(n))
    table.add_row("Exact Match", f"{summary['exact_match_rate']}%")
    table.add_row("Valid SQL Rate", f"{summary['valid_sql_rate']}%")
    table.add_row("Avg BLEU", f"{summary['avg_bleu']}")
    if run_execution_accuracy:
        table.add_row("Execution Accuracy", f"{summary['execution_accuracy_rate']}%")
    console.print(table)

    if metrics["error_types"]:
        err_table = Table(title="Error Distribution")
        err_table.add_column("Error Type", style="red")
        err_table.add_column("Count", style="yellow", justify="right")
        for err_type, count in metrics["error_types"].most_common():
            err_table.add_row(err_type, str(count))
        console.print(err_table)

    # ── Save ─────────────────────────────────────────────────────────
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"✅ Results saved to: {output_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Text-to-SQL model")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--test-split", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--run-execution-accuracy", action="store_true")
    parser.add_argument("--output-file", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    evaluate(
        adapter_path=args.adapter_path,
        test_path=args.test_split,
        base_model_name=args.base_model,
        num_samples=args.num_samples,
        run_execution_accuracy=args.run_execution_accuracy,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

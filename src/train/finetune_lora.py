"""
QLoRA Fine-Tuning Script for Text-to-SQL.

Loads a pre-trained causal LM in 4-bit (NF4), attaches LoRA adapters,
and trains using the TRL SFTTrainer on a formatted text-to-SQL dataset.

Usage:
    python -m src.train.finetune_lora --config configs/training_config.yaml
    python -m src.train.finetune_lora --config configs/training_config.yaml --model-name meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from src.train.callbacks import (
    EarlyStoppingOnPlateau,
    LogModelInfoCallback,
    WandbMetricsCallback,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# ─── Config Loader ────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments (if provided)."""
    if args.model_name:
        cfg.setdefault("model", {})["name"] = args.model_name
    if args.lora_r:
        cfg.setdefault("lora", {})["r"] = args.lora_r
    if args.epochs:
        cfg.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.lr:
        cfg.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size:
        cfg.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.resume_from:
        cfg["resume_from_checkpoint"] = args.resume_from
    return cfg


# ─── Model Loading ────────────────────────────────────────────────────
def load_quantized_model(cfg: dict):
    """Load model with 4-bit quantization (BitsAndBytes)."""
    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]

    # Map string dtype to torch dtype
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(quant_cfg["bnb_4bit_compute_dtype"], torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading model: {model_cfg['name']} (4-bit quantized)")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        token=os.getenv("HF_TOKEN"),
        attn_implementation="eager",
    )

    # Prepare model for k-bit training (freeze base, enable gradient for adapters)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Required for gradient checkpointing

    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        token=os.getenv("HF_TOKEN"),
    )
    # Set pad token for models that don't have one (e.g., Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Required for SFTTrainer
    return tokenizer


# ─── LoRA Configuration ──────────────────────────────────────────────
def create_lora_config(cfg: dict) -> LoraConfig:
    """Create LoRA configuration from YAML config."""
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )


# ─── Training Arguments ──────────────────────────────────────────────
def create_training_args(cfg: dict) -> SFTConfig:
    """Build SFTConfig (extends TrainingArguments) from config."""
    t = cfg["training"]
    sft = cfg.get("sft", {})

    return SFTConfig(
        packing=sft.get("packing", False),
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optim", "paged_adamw_8bit"),
        learning_rate=t["learning_rate"],
        weight_decay=t.get("weight_decay", 0.01),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=t.get("logging_first_step", True),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        bf16=t.get("bf16", True),
        tf32=t.get("tf32", True),
        seed=t.get("seed", 42),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        report_to=t.get("report_to", "wandb"),
        run_name=t.get("run_name", "qlora-text2sql"),
        # SFT-specific
        # max_seq_length=sft.get("max_seq_length", 1024),
        # packing=sft.get("packing", False),
        neftune_noise_alpha=sft.get("neftune_noise_alpha", None),
        dataset_text_field="text",
    )


# ─── Main Training Pipeline ──────────────────────────────────────────
def train(cfg: dict):
    """Full training pipeline: load → configure → train → save."""
    model_name = cfg["model"]["name"]
    ds_cfg = cfg.get("dataset", {})

    # ── 1. Load model & tokenizer ────────────────────────────────────
    console.rule("[bold blue]Loading Model & Tokenizer")
    model = load_quantized_model(cfg)
    tokenizer = load_tokenizer(model_name)

    # ── 2. Attach LoRA adapters ──────────────────────────────────────
    console.rule("[bold blue]Configuring LoRA Adapters")
    lora_config = create_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    # Print parameter summary
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    summary_table = Table(title="Model Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")
    summary_table.add_row("Base Model", model_name)
    summary_table.add_row("Total Parameters", f"{total:,}")
    summary_table.add_row("Trainable Parameters", f"{trainable:,}")
    summary_table.add_row("Trainable %", f"{100 * trainable / total:.2f}%")
    summary_table.add_row("LoRA Rank", str(cfg["lora"]["r"]))
    summary_table.add_row("LoRA Alpha", str(cfg["lora"]["alpha"]))
    console.print(summary_table)

    # ── 3. Load dataset ──────────────────────────────────────────────
    console.rule("[bold blue]Loading Dataset")
    processed_dir = ds_cfg.get("processed_dir", "data/processed")
    train_path = Path(processed_dir) / "train.jsonl"
    test_path = Path(processed_dir) / "test.jsonl"

    if not train_path.exists():
        console.print(
            f"[bold red]❌ Training data not found at {train_path}[/]\n"
            f"Run: python -m src.data.prepare_dataset --config configs/training_config.yaml"
        )
        return

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    eval_ds = load_dataset("json", data_files=str(test_path), split="train")
    logger.info(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    # ── 4. Build training arguments ──────────────────────────────────
    console.rule("[bold blue]Configuring Trainer")
    training_args = create_training_args(cfg)

    # ── 5. Initialize SFTTrainer ─────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            WandbMetricsCallback(),
            EarlyStoppingOnPlateau(patience=5),
            LogModelInfoCallback(),
        ],
    )

    # ── 6. Train ─────────────────────────────────────────────────────
    console.rule("[bold green]🚀 Starting Training")
    resume = cfg.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume)

    # ── 7. Save final adapter ────────────────────────────────────────
    console.rule("[bold blue]Saving Model")
    final_path = Path(cfg["training"]["output_dir"]) / "final-adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"✅ LoRA adapter saved to: {final_path}")

    # ── 8. Push to Hub (optional) ────────────────────────────────────
    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub", False):
        console.rule("[bold blue]Pushing to Hugging Face Hub")
        hub_id = hub_cfg.get("hub_model_id", "visheshgupta29/mistral-7b-text2sql-qlora")
        model.push_to_hub(hub_id, private=hub_cfg.get("hub_private", False))
        tokenizer.push_to_hub(hub_id, private=hub_cfg.get("hub_private", False))
        logger.info(f"✅ Pushed to Hub: {hub_id}")

    console.rule("[bold green]✅ Training Complete!")


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Text-to-SQL")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    # Print full config
    console.rule("[bold blue]Configuration")
    console.print_json(data=cfg)

    train(cfg)


if __name__ == "__main__":
    main()

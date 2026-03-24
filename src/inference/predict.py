"""
Programmatic inference for the fine-tuned Text-to-SQL model.

Provides a clean `SQLPredictor` class for single/batch predictions.

Usage:
    from src.inference.predict import SQLPredictor

    predictor = SQLPredictor(adapter_path="outputs/final-adapter")
    sql = predictor.predict(
        question="How many employees earn over 50000?",
        schema="CREATE TABLE employees (id INT, name TEXT, salary REAL);"
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.prompt_templates import format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class SQLPredictor:
    """High-level inference wrapper for the fine-tuned Text-to-SQL model.

    Parameters
    ----------
    adapter_path : str
        Path to the saved LoRA adapter directory.
    base_model_name : str, optional
        Base model name. If not provided, reads from adapter_config.json.
    load_in_4bit : bool
        Whether to quantize the base model to 4-bit.
    device_map : str
        Device mapping strategy.
    max_new_tokens : int
        Max tokens to generate.
    temperature : float
        Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        adapter_path: str,
        base_model_name: str | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Resolve base model name
        if base_model_name is None:
            config_path = Path(adapter_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                base_model_name = cfg.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError("Cannot determine base model. Provide base_model_name explicitly.")

        self.base_model_name = base_model_name
        logger.info(f"Loading base model: {base_model_name}")

        # Quantization config
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            token=os.getenv("HF_TOKEN"),
        )

        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("✅ Model loaded and ready for inference")

    @torch.inference_mode()
    def predict(
        self,
        question: str,
        schema: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate SQL for a single question.

        Parameters
        ----------
        question : str
            Natural language question.
        schema : str
            SQL CREATE TABLE statement(s) describing the database schema.
        max_new_tokens : int, optional
            Override default max_new_tokens.
        temperature : float, optional
            Override default temperature.

        Returns
        -------
        str
            Generated SQL query.
        """
        max_tok = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        # Format prompt (without SQL answer)
        prompt = format_prompt(
            model_name=self.base_model_name,
            question=question,
            schema=schema,
            sql=None,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        # Stop at first ';' (complete SQL statement) or EOS.
        # SentencePiece encodes standalone ";" as ▁; but the model may emit
        # bare ";" in context (different token ID).  Scan the vocab.
        semicolon_ids = set(self.tokenizer.encode(";", add_special_tokens=False))
        for tok_str, tok_id in self.tokenizer.get_vocab().items():
            if tok_str.replace("\u2581", "").strip() == ";":
                semicolon_ids.add(tok_id)
        stop_ids = list({self.tokenizer.eos_token_id} | semicolon_ids)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=0.9,
            do_sample=temp > 0,
            repetition_penalty=1.1,                # v3: keep low — ';' stop token handles stopping
            eos_token_id=stop_ids,                 # v3: stop at first ';'
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        sql = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Format cleanup only
        if "###" in sql:
            sql = sql.split("###")[0].strip()
        if "\n\n" in sql:
            sql = sql.split("\n\n")[0].strip()
        # Safety net: first complete statement only
        if ";" in sql:
            sql = sql.split(";")[0].strip()

        return sql

    def predict_batch(
        self,
        examples: list[dict],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate SQL for multiple examples.

        Parameters
        ----------
        examples : list of dict
            Each dict must have 'question' and 'schema' keys.

        Returns
        -------
        list of str
            Generated SQL queries.
        """
        return [
            self.predict(
                question=ex["question"],
                schema=ex["schema"],
                max_new_tokens=max_new_tokens,
            )
            for ex in examples
        ]


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    """Interactive CLI for testing predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive SQL prediction")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()

    predictor = SQLPredictor(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
    )

    print("\n🤖 Text-to-SQL Predictor (type 'quit' to exit)")
    print("=" * 60)

    while True:
        print("\nPaste your schema (end with an empty line):")
        schema_lines = []
        while True:
            line = input()
            if line == "":
                break
            schema_lines.append(line)
        schema = "\n".join(schema_lines)

        if schema.lower() == "quit":
            break

        question = input("\nQuestion: ")
        if question.lower() == "quit":
            break

        sql = predictor.predict(question=question, schema=schema)
        print(f"\n📝 Generated SQL:\n{sql}")


if __name__ == "__main__":
    main()

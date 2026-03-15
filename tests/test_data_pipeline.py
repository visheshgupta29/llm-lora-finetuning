"""Unit tests for the data-processing pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.data.prompt_templates import (
    format_prompt,
    format_prompt_chatml,
    format_prompt_generic,
    format_prompt_llama3,
    format_prompt_mistral,
    get_formatter,
)


# ───────────────────────────────── prompt template tests ──────────────
class TestPromptTemplates:
    """Tests for every prompt formatter."""

    QUESTION = "How many employees are there?"
    SCHEMA = "CREATE TABLE employees (id INT, name TEXT);"
    SQL = "SELECT COUNT(*) FROM employees;"

    def test_generic_with_sql(self):
        prompt = format_prompt_generic(self.QUESTION, self.SCHEMA, self.SQL)
        assert self.QUESTION in prompt
        assert self.SCHEMA in prompt
        assert self.SQL in prompt
        assert "### Answer:" in prompt

    def test_generic_without_sql(self):
        prompt = format_prompt_generic(self.QUESTION, self.SCHEMA, sql=None)
        assert self.QUESTION in prompt
        assert self.SCHEMA in prompt
        assert prompt.endswith("### Answer:\n")

    def test_mistral_format(self):
        prompt = format_prompt_mistral(self.QUESTION, self.SCHEMA, self.SQL)
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_llama3_format(self):
        prompt = format_prompt_llama3(self.QUESTION, self.SCHEMA, self.SQL)
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt

    def test_chatml_format(self):
        prompt = format_prompt_chatml(self.QUESTION, self.SCHEMA, self.SQL)
        assert "<|im_start|>system" in prompt
        assert "<|im_end|>" in prompt

    @pytest.mark.parametrize(
        "model_name,expected_keyword",
        [
            ("mistralai/Mistral-7B-Instruct-v0.3", "[INST]"),
            ("meta-llama/Llama-3.1-8B-Instruct", "<|begin_of_text|>"),
            ("Qwen/Qwen2-7B-Instruct", "<|im_start|>"),
            ("microsoft/Phi-3-mini-128k-instruct", "<|im_start|>"),
            ("codellama/CodeLlama-7b-hf", "### Input:"),
        ],
    )
    def test_get_formatter_dispatches(self, model_name, expected_keyword):
        formatter = get_formatter(model_name)
        prompt = formatter(self.QUESTION, self.SCHEMA, self.SQL)
        assert expected_keyword in prompt

    def test_format_prompt_convenience(self):
        prompt = format_prompt(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            question=self.QUESTION,
            schema=self.SCHEMA,
            sql=self.SQL,
        )
        assert "[INST]" in prompt


# ──────────────────────────────── data pipeline smoke test ────────────
class TestPrepareDataset:
    """Smoke tests for dataset preparation helpers.

    These do NOT download from HF — they test the transform logic only.
    """

    def test_saved_jsonl_format(self, tmp_path: Path):
        """Verify that writing a formatted example as JSONL produces valid JSON."""
        sample = {
            "text": format_prompt_generic(
                question="List all names",
                schema="CREATE TABLE users (id INT, name TEXT);",
                sql="SELECT name FROM users;",
            )
        }
        out = tmp_path / "test.jsonl"
        with open(out, "w") as f:
            f.write(json.dumps(sample) + "\n")

        with open(out) as f:
            data = json.loads(f.readline())

        assert "text" in data
        assert "List all names" in data["text"]

    def test_multiple_examples_round_trip(self, tmp_path: Path):
        """Ensure multiple JSONL rows survive a write/read cycle."""
        examples = [
            {"text": f"example {i}", "index": i}
            for i in range(5)
        ]
        out = tmp_path / "multi.jsonl"
        with open(out, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        with open(out) as f:
            loaded = [json.loads(line) for line in f]

        assert len(loaded) == 5
        assert loaded[3]["index"] == 3


# ──────────────────────────────── config loading ──────────────────────
class TestConfigLoading:
    """Ensure the training config file is valid YAML."""

    @pytest.fixture
    def config_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml"

    def test_config_exists(self, config_path: Path):
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_config_is_valid_yaml(self, config_path: Path):
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "lora" in cfg
        assert "training" in cfg

    def test_config_has_required_keys(self, config_path: Path):
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "name" in cfg["model"]
        assert "r" in cfg["lora"]
        assert "lora_alpha" in cfg["lora"]
        assert "per_device_train_batch_size" in cfg["training"]

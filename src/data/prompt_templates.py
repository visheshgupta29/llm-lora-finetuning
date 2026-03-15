"""
Prompt templates for different base models.

Each model family has its own chat/instruction format. This module provides
consistent prompt construction so that the same dataset can be used with
Mistral, Llama, CodeLlama, Phi-3, Qwen, etc.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────
# Generic (works with any causal LM — used as default / for Mistral)
# ─────────────────────────────────────────────────────────────────────

GENERIC_SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. Given a database schema and a natural "
    "language question, generate the correct SQL query. Output ONLY the SQL "
    "query, nothing else."
)


def format_generic(
    question: str,
    schema: str,
    sql: str | None = None,
) -> str:
    """Plain instruction-style prompt (no special tokens).

    Used for base models (Mistral-base, Llama-base) where we train the
    model to follow this format via SFT.
    """
    prompt = (
        "### Task: Generate a SQL query to answer the following question.\n\n"
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        "### SQL Query:\n"
    )
    if sql is not None:
        prompt += sql.strip()
    return prompt


# ─────────────────────────────────────────────────────────────────────
# Mistral Instruct format  ([INST] ... [/INST])
# ─────────────────────────────────────────────────────────────────────

def format_mistral_instruct(
    question: str,
    schema: str,
    sql: str | None = None,
) -> str:
    """Mistral-Instruct chat template."""
    user_content = (
        f"{GENERIC_SYSTEM_PROMPT}\n\n"
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}"
    )
    prompt = f"<s>[INST] {user_content} [/INST]"
    if sql is not None:
        prompt += f" {sql.strip()}</s>"
    return prompt


# ─────────────────────────────────────────────────────────────────────
# Llama 3.x chat format
# ─────────────────────────────────────────────────────────────────────

def format_llama3_chat(
    question: str,
    schema: str,
    sql: str | None = None,
) -> str:
    """Llama 3 / 3.1 chat template."""
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{GENERIC_SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if sql is not None:
        prompt += f"{sql.strip()}<|eot_id|>"
    return prompt


# ─────────────────────────────────────────────────────────────────────
# ChatML format  (Qwen, Phi-3, many others)
# ─────────────────────────────────────────────────────────────────────

def format_chatml(
    question: str,
    schema: str,
    sql: str | None = None,
) -> str:
    """ChatML template (Qwen2, Phi-3, etc.)."""
    prompt = (
        "<|im_start|>system\n"
        f"{GENERIC_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if sql is not None:
        prompt += f"{sql.strip()}<|im_end|>"
    return prompt


# ─────────────────────────────────────────────────────────────────────
# Registry — pick the right formatter by model name
# ─────────────────────────────────────────────────────────────────────

_TEMPLATE_REGISTRY: dict[str, callable] = {
    "generic": format_generic,
    "mistral": format_generic,             # Base Mistral uses generic
    "mistral-instruct": format_mistral_instruct,
    "llama": format_generic,               # Base Llama uses generic
    "llama3-chat": format_llama3_chat,
    "codellama": format_generic,
    "chatml": format_chatml,
    "qwen": format_chatml,
    "phi": format_chatml,
}


def get_formatter(model_name: str) -> callable:
    """Return the appropriate prompt formatter for a model name.

    Tries to match known model families by inspecting the name string.
    Falls back to the generic template.
    """
    model_lower = model_name.lower()

    if "instruct" in model_lower and "mistral" in model_lower:
        return _TEMPLATE_REGISTRY["mistral-instruct"]
    if "llama-3" in model_lower and ("instruct" in model_lower or "chat" in model_lower):
        return _TEMPLATE_REGISTRY["llama3-chat"]
    if "qwen" in model_lower:
        return _TEMPLATE_REGISTRY["chatml"]
    if "phi" in model_lower:
        return _TEMPLATE_REGISTRY["chatml"]

    # Default: generic instruction format (works for base models)
    return _TEMPLATE_REGISTRY["generic"]


def format_prompt(
    model_name: str,
    question: str,
    schema: str,
    sql: str | None = None,
) -> str:
    """Convenience wrapper: picks the right template and formats."""
    formatter = get_formatter(model_name)
    return formatter(question=question, schema=schema, sql=sql)

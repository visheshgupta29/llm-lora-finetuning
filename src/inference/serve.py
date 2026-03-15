"""
Gradio web demo for the fine-tuned Text-to-SQL model.

Launches a browser UI where users can enter a question + schema
and get back a SQL query — showcasing the fine-tuned adapter.

Usage:
    python -m src.inference.serve --adapter-path outputs/final-adapter
"""

from __future__ import annotations

import argparse
import logging

import gradio as gr

from src.inference.predict import SQLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Example inputs ──────────────────────────────────────────────────
EXAMPLES = [
    [
        "How many departments have a budget greater than 100000?",
        "CREATE TABLE departments (id INT PRIMARY KEY, name TEXT, budget REAL, head TEXT);",
    ],
    [
        "Show the names of students who scored above 90 in math",
        (
            "CREATE TABLE students (id INT PRIMARY KEY, name TEXT, grade TEXT);\n"
            "CREATE TABLE scores (student_id INT, subject TEXT, score REAL);"
        ),
    ],
    [
        "List all products and their categories ordered by price descending",
        (
            "CREATE TABLE products (id INT, name TEXT, price REAL, category_id INT);\n"
            "CREATE TABLE categories (id INT, category_name TEXT);"
        ),
    ],
    [
        "What is the average salary per department?",
        "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL);",
    ],
    [
        "Find customers who placed more than 3 orders",
        (
            "CREATE TABLE customers (id INT, name TEXT, email TEXT);\n"
            "CREATE TABLE orders (id INT, customer_id INT, total REAL, created_at TEXT);"
        ),
    ],
]


def build_demo(predictor: SQLPredictor) -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    custom_css = """
    .gradio-container { max-width: 900px !important; }
    .output-sql { font-family: 'Fira Code', 'Cascadia Code', monospace; }
    """

    with gr.Blocks(
        title="🔮 Text → SQL  (LoRA Fine-Tuned)",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            # 🔮 Text-to-SQL Generator
            ### Fine-tuned with QLoRA on Mistral-7B

            Enter your **database schema** and a **natural-language question**
            to get an AI-generated SQL query.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                schema_input = gr.Textbox(
                    label="📋 Database Schema",
                    placeholder="CREATE TABLE employees (\n  id INT PRIMARY KEY,\n  name TEXT,\n  salary REAL\n);",
                    lines=8,
                    max_lines=20,
                )
                question_input = gr.Textbox(
                    label="❓ Question",
                    placeholder="How many employees earn over 50000?",
                    lines=2,
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    submit_btn = gr.Button("🚀 Generate SQL", variant="primary")

            with gr.Column(scale=1):
                sql_output = gr.Code(
                    label="📝 Generated SQL",
                    language="sql",
                    interactive=False,
                    lines=8,
                    elem_classes=["output-sql"],
                )
                status_output = gr.Textbox(
                    label="ℹ️ Status",
                    interactive=False,
                    lines=1,
                )

        # ── Examples ──────────────────────────────────────────────────
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, schema_input],
            label="💡 Try These Examples",
        )

        # ── Accordion: model info ─────────────────────────────────────
        with gr.Accordion("🔧 Model Details", open=False):
            gr.Markdown(
                f"""
                | Setting | Value |
                |---------|-------|
                | **Base model** | `{predictor.base_model_name}` |
                | **Adapter** | `{predictor.adapter_path}` |
                | **Max tokens** | `{predictor.max_new_tokens}` |
                | **Temperature** | `{predictor.temperature}` |
                | **Quantisation** | 4-bit NF4 |
                """
            )

        # ── Event handlers ────────────────────────────────────────────
        def _predict(question: str, schema: str) -> tuple[str, str]:
            if not question.strip():
                return "", "⚠️ Please enter a question."
            if not schema.strip():
                return "", "⚠️ Please enter a database schema."
            try:
                sql = predictor.predict(question=question, schema=schema)
                return sql, "✅ Query generated successfully"
            except Exception as e:
                logger.exception("Prediction failed")
                return "", f"❌ Error: {e}"

        submit_btn.click(
            fn=_predict,
            inputs=[question_input, schema_input],
            outputs=[sql_output, status_output],
        )
        question_input.submit(
            fn=_predict,
            inputs=[question_input, schema_input],
            outputs=[sql_output, status_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[question_input, schema_input, sql_output, status_output],
        )

    return demo


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model name")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    logger.info("Loading model …")
    predictor = SQLPredictor(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
    )

    demo = build_demo(predictor)

    logger.info(f"Launching Gradio on http://localhost:{args.port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

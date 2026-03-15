#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  evaluate.sh — Evaluate fine-tuned model and compare vs base
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

ADAPTER="${1:-outputs/final-adapter}"
TEST_SPLIT="${2:-data/processed/test.jsonl}"
NUM_SAMPLES="${3:-200}"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════"
echo "  📊  Evaluation Pipeline"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Adapter     : $ADAPTER"
echo "  Test split  : $TEST_SPLIT"
echo "  Samples     : $NUM_SAMPLES"
echo ""

# ── Step 1: Evaluate fine-tuned model ────────────────────────
echo "🔍  Step 1/2 — Evaluating fine-tuned model …"
python -m src.evaluate.evaluate_model \
    --adapter-path "$ADAPTER" \
    --test-split "$TEST_SPLIT" \
    --num-samples "$NUM_SAMPLES" \
    --run-execution-accuracy \
    --output-file "$OUTPUT_DIR/eval_finetuned.json"

echo ""

# ── Step 2: Side-by-side comparison ──────────────────────────
echo "⚖️  Step 2/2 — Comparing base vs fine-tuned …"
python -m src.evaluate.compare_models \
    --adapter-path "$ADAPTER" \
    --test-split "$TEST_SPLIT" \
    --num-samples "$NUM_SAMPLES" \
    --output-file "$OUTPUT_DIR/comparison.json"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅  Evaluation complete — results in $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════"

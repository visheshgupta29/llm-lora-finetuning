#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  train.sh — One-click QLoRA fine-tuning launcher
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "═══════════════════════════════════════════════════"
echo "  🚀  QLoRA Fine-Tuning Pipeline"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Prepare dataset ──────────────────────────────────
echo "📦  Step 1/2 — Preparing dataset …"
python -m src.data.prepare_dataset \
    --config configs/training_config.yaml \
    "$@"

echo ""

# ── Step 2: Fine-tune ────────────────────────────────────────
echo "🔧  Step 2/2 — Fine-tuning with QLoRA …"
python -m src.train.finetune_lora \
    --config configs/training_config.yaml \
    "$@"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅  Training complete!"
echo "═══════════════════════════════════════════════════"

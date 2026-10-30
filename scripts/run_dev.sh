#!/usr/bin/env bash
# Startup script for Persian and Arabic Manuscript HTR/OCR AI Service
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

echo "=================================================================="
echo " Starting Persian & Arabic Manuscript HTR/OCR AI Service"
echo "=================================================================="

# Check Python environment
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

export PYTHONPATH="$WORKSPACE_ROOT"

# Ensure storage directories exist
mkdir -p storage/models storage/checkpoints storage/datasets /tmp/htr_service

# Generate sample dataset if missing
if [ ! -f "storage/datasets/sample_persian_arabic/manifest.jsonl" ]; then
    echo "Sample dataset not found. Generating sample manuscript dataset..."
    python3 scripts/prepare_dataset.py
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Uvicorn HTTP server on http://${HOST}:${PORT}..."
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level info

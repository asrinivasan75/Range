#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Range API on http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo

PYTHONPATH="$ROOT_DIR" python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

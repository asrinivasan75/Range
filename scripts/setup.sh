#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════╗"
echo "║        Range — Setup Script          ║"
echo "╚══════════��═══════════════════════════╝"
echo

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Python setup
echo "→ Setting up Python environment..."
if command -v uv &>/dev/null; then
    echo "  Using uv"
    uv venv --python 3.11 .venv 2>/dev/null || uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
elif command -v python3 &>/dev/null; then
    echo "  Using python3 + pip"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
else
    echo "ERROR: No Python 3 found. Install Python 3.11+ first."
    exit 1
fi
echo "  ✓ Python environment ready"

# Frontend setup
echo
echo "→ Setting up frontend..."
cd apps/web
if command -v pnpm &>/dev/null; then
    pnpm install
elif command -v npm &>/dev/null; then
    npm install
else
    echo "WARNING: No npm/pnpm found. Install Node.js 18+ to use the frontend."
fi
cd "$ROOT_DIR"
echo "  ✓ Frontend dependencies installed"

# Create data directories
echo
echo "→ Creating data directories..."
mkdir -p data/runs data/checkpoints
echo "  �� Data directories ready"

echo
echo "╔══════════════════════════════════════╗"
echo "║          Setup Complete!             ║"
echo "╠═══════════���══════════════════════════╣"
echo "║                                      ║"
echo "║  Start API:      ./scripts/api.sh    ║"
echo "║  Start Frontend:  ./scripts/web.sh   ║"
echo "║  Run Training:    ./scripts/train.sh ║"
echo "║  Generate Demo:   ./scripts/seed.sh  ║"
echo "║                                      ║"
echo "╚══════════════════════════════════════╝"

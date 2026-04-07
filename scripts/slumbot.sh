#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

USERNAME="${1:-}"
PASSWORD="${2:-}"
HANDS="${3:-50}"

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Usage: ./scripts/slumbot.sh <username> <password> [n_hands]"
    echo ""
    echo "Benchmark your bot against Slumbot (free, strong HUNL bot)."
    echo "Register at https://slumbot.com first."
    echo ""
    echo "Example: ./scripts/slumbot.sh myuser mypass 100"
    exit 1
fi

echo "╔══════════════════════════════════════╗"
echo "║   Range vs Slumbot Benchmark         ║"
echo "╚══════════════════════════════════════╝"
echo

PYTHONPATH="$ROOT_DIR" python -m packages.solver.slumbot "$USERNAME" "$PASSWORD" "$HANDS"

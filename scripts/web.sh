#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/apps/web"

echo "Starting Range frontend on http://localhost:3000"
echo

if command -v pnpm &>/dev/null; then
    pnpm dev
else
    npm run dev
fi

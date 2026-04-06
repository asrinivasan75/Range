#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Generating demo training data..."
echo

PYTHONPATH="$ROOT_DIR" python -c "
import sys
sys.path.insert(0, '.')
from packages.solver.trainer import TrainingOrchestrator, TrainingConfig

orchestrator = TrainingOrchestrator(data_dir='data')

# 1. Kuhn Poker — quick validation
print('Training Kuhn Poker (CFR, 5000 iterations)...')
config1 = TrainingConfig(game_type='kuhn', algorithm='cfr', n_iterations=5000)
run1 = orchestrator.start_training(config=config1, name='kuhn-demo', blocking=True)
print(f'  ✓ {run1.name} — {run1.status.value}')

# 2. Kuhn Poker — MCCFR
print('Training Kuhn Poker (MCCFR, 10000 iterations)...')
config2 = TrainingConfig(game_type='kuhn', algorithm='mccfr', n_iterations=10000)
run2 = orchestrator.start_training(config=config2, name='kuhn-mccfr-demo', blocking=True)
print(f'  ✓ {run2.name} — {run2.status.value}')

# 3. Simplified Hold'em — the main event
print('Training Simplified Hold\\'em (MCCFR, 10000 iterations)...')
config3 = TrainingConfig(
    game_type='simplified_holdem',
    algorithm='mccfr',
    n_iterations=10000,
    preflop_buckets=8,
    flop_buckets=8,
)
run3 = orchestrator.start_training(config=config3, name='holdem-demo', blocking=True)
print(f'  ✓ {run3.name} — {run3.status.value}')

print()
print('Demo data generated! Start the API and frontend to explore results.')
print('  ./scripts/api.sh')
print('  ./scripts/web.sh')
"

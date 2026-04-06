#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

GAME="${1:-simplified_holdem}"
ITERS="${2:-10000}"
ALGO="${3:-mccfr}"

echo "╔��═════════════════════════════════════╗"
echo "║     Range — Training Runner          ║"
echo "╚══════════��═══════════════════════════╝"
echo
echo "  Game:       $GAME"
echo "  Iterations: $ITERS"
echo "  Algorithm:  $ALGO"
echo

PYTHONPATH="$ROOT_DIR" python -c "
import sys
sys.path.insert(0, '.')
from packages.solver.trainer import TrainingOrchestrator, TrainingConfig

config = TrainingConfig(
    game_type='${GAME}',
    algorithm='${ALGO}',
    n_iterations=${ITERS},
)

orchestrator = TrainingOrchestrator(data_dir='data')
print(f'Starting training: {config.game_type} / {config.algorithm} / {config.n_iterations} iterations')
print()

run = orchestrator.start_training(config=config, blocking=True)
print()
print(f'Run ID:     {run.id}')
print(f'Status:     {run.status.value}')
print(f'Iterations: {run.current_iteration}')

# Print summary
metrics = orchestrator.get_run_metrics(run.id)
if metrics and 'summary' in metrics:
    s = metrics['summary']
    print(f'Info sets:  {s.get(\"n_info_sets\", \"?\")}')
    print(f'Exploit:    {s.get(\"final_exploitability\", \"?\"):.6f}')
    print(f'Time:       {s.get(\"total_time_seconds\", \"?\"):.2f}s')
"

"""Parallel Slumbot training — run multiple accounts concurrently.

Each account gets its own copy of the base RL weights and learns independently.
All results logged to separate files for comparison in the replayer.
"""

from __future__ import annotations
import shutil
import time
import threading
from pathlib import Path

from packages.solver.slumbot import run_benchmark


def run_parallel(
    accounts: list[dict],
    n_hands_each: int,
    base_weights: str = "data/rl_agent.pkl",
    verbose: bool = True,
) -> dict:
    """Run multiple Slumbot sessions in parallel with independent RL agents."""
    threads = []
    results = {}
    lock = threading.Lock()

    for acc in accounts:
        label = acc.get("label", acc["username"])
        log_path = f"data/slumbot_{label}_log.json"
        weights_path = f"data/rl_agent_{label}.pkl"

        # Copy base weights for this agent
        if Path(base_weights).exists():
            shutil.copy(base_weights, weights_path)
            print(f"[{label}] Loaded base weights")
        else:
            print(f"[{label}] Starting fresh")

        def _run(username, password, label, log_path, weights_path, n_hands):
            try:
                result = run_benchmark(
                    username=username,
                    password=password,
                    n_hands=n_hands,
                    verbose=verbose,
                    save_path=log_path,
                    use_rl=True,
                    weights_path=weights_path,
                )
                with lock:
                    results[label] = result
            except Exception as e:
                print(f"[{label}] Error: {e}")
                with lock:
                    results[label] = {"error": str(e)}

        t = threading.Thread(
            target=_run,
            args=(acc["username"], acc["password"], label, log_path, weights_path, n_hands_each),
            daemon=True,
        )
        threads.append((label, t))

    print(f"\nStarting {len(threads)} parallel sessions, {n_hands_each} hands each...")
    print(f"Total hands: {len(threads) * n_hands_each}\n")

    for label, t in threads:
        t.start()
        time.sleep(1)  # stagger to avoid login collision

    # Wait for all
    for label, t in threads:
        t.join()

    # Summary
    print(f"\n{'='*60}")
    print(f"PARALLEL RESULTS")
    print(f"{'='*60}")
    total_hands = 0
    total_bb = 0
    for label, r in sorted(results.items()):
        if "error" not in r:
            h = r.get("hands_played", r.get("hands", 0))
            bb = r.get("total_bb", 0)
            rate = r.get("bb_per_100", 0)
            total_hands += h
            total_bb += bb
            print(f"  {label:20s}: {h:>6} hands  {bb:>+9.1f}bb  ({rate:>+8.1f} bb/100)")

    if total_hands > 0:
        print(f"  {'COMBINED':20s}: {total_hands:>6} hands  {total_bb:>+9.1f}bb  ({total_bb/total_hands*100:>+8.1f} bb/100)")

    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6250
    accounts = [
        {"username": "eventreinforcev4", "password": "password", "label": "rl_v2"},
        {"username": "eventreinforcev5", "password": "password", "label": "rl_v5"},
        {"username": "eventreinforcev6", "password": "password", "label": "rl_v6"},
        {"username": "eventreinforcev7", "password": "password", "label": "rl_v7"},
    ]
    run_parallel(accounts, n_hands_each=n)

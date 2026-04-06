"""Training orchestrator — manages training runs, persistence, and progress."""

from __future__ import annotations
import json
import time
import uuid
import pickle
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from packages.solver.cfr import CFRTrainer
from packages.solver.kuhn import KuhnPoker
from packages.solver.holdem_simplified import SimplifiedHoldem, SimplifiedHoldemConfig
from packages.solver.holdem_full import FullStreetHoldem, FullHoldemConfig
from packages.solver.metrics import MetricsCollector


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GameType(str, Enum):
    KUHN = "kuhn"
    SIMPLIFIED_HOLDEM = "simplified_holdem"


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    game_type: str = "simplified_holdem"
    algorithm: str = "mccfr"  # "cfr" or "mccfr"
    n_iterations: int = 10000
    checkpoint_interval: int = 1000
    # Simplified Hold'em specific
    preflop_buckets: int = 8
    flop_buckets: int = 8
    max_raises: int = 2

    def to_dict(self) -> dict:
        return {
            "game_type": self.game_type,
            "algorithm": self.algorithm,
            "n_iterations": self.n_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "preflop_buckets": self.preflop_buckets,
            "flop_buckets": self.flop_buckets,
            "max_raises": self.max_raises,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingRun:
    """Represents a single training run."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    status: RunStatus = RunStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    current_iteration: int = 0
    error: str | None = None

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.config.game_type}-{self.config.algorithm}-{self.id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_iteration": self.current_iteration,
            "error": self.error,
        }


class TrainingOrchestrator:
    """Manages training runs with persistence and background execution."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.runs_dir = self.data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self._active_runs: dict[str, dict] = {}  # run_id -> {thread, trainer, collector, run}
        self._load_existing_runs()

    def _load_existing_runs(self) -> None:
        """Load metadata for existing runs from disk."""
        self._run_metadata: dict[str, TrainingRun] = {}
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                meta_file = run_dir / "run.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        data = json.load(f)
                    config = TrainingConfig.from_dict(data.get("config", {}))
                    run = TrainingRun(
                        id=data["id"],
                        name=data.get("name", ""),
                        config=config,
                        status=RunStatus(data.get("status", "completed")),
                        created_at=data.get("created_at", 0),
                        started_at=data.get("started_at"),
                        completed_at=data.get("completed_at"),
                        current_iteration=data.get("current_iteration", 0),
                        error=data.get("error"),
                    )
                    self._run_metadata[run.id] = run

    def list_runs(self) -> list[dict]:
        """List all training runs."""
        runs = []
        for run in sorted(self._run_metadata.values(), key=lambda r: r.created_at, reverse=True):
            d = run.to_dict()
            # Add live progress for active runs
            if run.id in self._active_runs:
                active = self._active_runs[run.id]
                d["current_iteration"] = active["trainer"].iterations_done
                d["status"] = "running"
            runs.append(d)
        return runs

    def get_run(self, run_id: str) -> dict | None:
        """Get details for a specific run."""
        run = self._run_metadata.get(run_id)
        if not run:
            return None
        d = run.to_dict()
        if run_id in self._active_runs:
            active = self._active_runs[run_id]
            d["current_iteration"] = active["trainer"].iterations_done
            d["status"] = "running"
        return d

    def get_run_metrics(self, run_id: str) -> dict | None:
        """Get metrics for a run."""
        # Check active runs first
        if run_id in self._active_runs:
            collector = self._active_runs[run_id]["collector"]
            return {
                "series": collector.get_metrics_series(),
                "summary": collector.get_summary(),
            }

        # Check saved metrics
        run_dir = self.runs_dir / run_id
        series_file = run_dir / "metrics_series.json"
        summary_file = run_dir / "summary.json"

        result = {}
        if series_file.exists():
            with open(series_file) as f:
                result["series"] = json.load(f)
        if summary_file.exists():
            with open(summary_file) as f:
                result["summary"] = json.load(f)
        return result if result else None

    def get_run_strategy(self, run_id: str) -> dict | None:
        """Get strategy data for a run."""
        if run_id in self._active_runs:
            trainer = self._active_runs[run_id]["trainer"]
            return trainer.get_strategy_summary()

        strategy_file = self.runs_dir / run_id / "strategy_latest.json"
        if strategy_file.exists():
            with open(strategy_file) as f:
                data = json.load(f)
            return data.get("strategies", data)
        return None

    def start_training(
        self,
        config: TrainingConfig | None = None,
        name: str = "",
        blocking: bool = False,
    ) -> TrainingRun:
        """Start a new training run."""
        config = config or TrainingConfig()
        run = TrainingRun(config=config, name=name)
        self._run_metadata[run.id] = run

        # Save initial metadata
        run_dir = self.runs_dir / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_meta(run)

        if blocking:
            self._execute_training(run)
        else:
            thread = threading.Thread(
                target=self._execute_training,
                args=(run,),
                daemon=True,
            )
            thread.start()

        return run

    def _execute_training(self, run: TrainingRun) -> None:
        """Execute training (runs in background thread or foreground)."""
        run.status = RunStatus.RUNNING
        run.started_at = time.time()
        self._save_run_meta(run)

        try:
            # Create game and trainer
            if run.config.game_type == "kuhn":
                game = KuhnPoker()
            elif run.config.game_type == "full_holdem":
                full_config = FullHoldemConfig(
                    preflop_buckets=run.config.preflop_buckets,
                    flop_buckets=run.config.flop_buckets,
                    turn_buckets=run.config.preflop_buckets,  # reuse
                    river_buckets=run.config.preflop_buckets,  # reuse
                    max_raises_per_street=run.config.max_raises,
                )
                game = FullStreetHoldem(full_config)
            else:
                holdem_config = SimplifiedHoldemConfig(
                    preflop_buckets=run.config.preflop_buckets,
                    flop_buckets=run.config.flop_buckets,
                    max_raises_per_street=run.config.max_raises,
                )
                game = SimplifiedHoldem(holdem_config)

            trainer = CFRTrainer(game)
            collector = MetricsCollector()

            self._active_runs[run.id] = {
                "trainer": trainer,
                "collector": collector,
                "run": run,
            }

            def on_metrics(raw: dict):
                collector.record_metrics(raw)
                run.current_iteration = raw["iteration"]

                # Checkpoint strategy at intervals
                if raw["iteration"] % run.config.checkpoint_interval == 0:
                    strategy = trainer.get_strategy_summary()
                    collector.record_strategy(raw["iteration"], strategy)

            # Run training
            if run.config.algorithm == "cfr":
                trainer.train_vanilla(run.config.n_iterations, callback=on_metrics)
            else:
                trainer.train_mccfr(run.config.n_iterations, callback=on_metrics)

            # Final strategy snapshot
            strategy = trainer.get_strategy_summary()
            collector.record_strategy(trainer.iterations_done, strategy)

            # Save results
            run_dir = self.runs_dir / run.id
            collector.save(run_dir)

            # Save trainer state
            with open(run_dir / "trainer_state.pkl", "wb") as f:
                pickle.dump({
                    "info_sets": {k: {
                        "cumulative_regrets": v.cumulative_regrets.tolist(),
                        "strategy_sum": v.strategy_sum.tolist(),
                        "n_actions": v.n_actions,
                    } for k, v in trainer.info_sets.items()},
                    "iterations": trainer.iterations_done,
                }, f)

            run.status = RunStatus.COMPLETED
            run.completed_at = time.time()

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            run.completed_at = time.time()

        finally:
            self._save_run_meta(run)
            self._active_runs.pop(run.id, None)

    def _save_run_meta(self, run: TrainingRun) -> None:
        """Save run metadata to disk."""
        run_dir = self.runs_dir / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "run.json", "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running training job."""
        if run_id in self._active_runs:
            run = self._active_runs[run_id]["run"]
            run.status = RunStatus.CANCELLED
            run.completed_at = time.time()
            self._save_run_meta(run)
            # Thread will check status and stop
            return True
        return False

    def delete_run(self, run_id: str) -> bool:
        """Delete a training run and its data."""
        import shutil
        self.cancel_run(run_id)
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        self._run_metadata.pop(run_id, None)
        return True

    def get_available_games(self) -> list[dict]:
        """List available game types."""
        return [
            {
                "id": "kuhn",
                "name": "Kuhn Poker",
                "description": "3-card toy game for CFR validation. 2 players, 12 info sets.",
                "complexity": "trivial",
                "recommended_iterations": 10000,
                "estimated_time": "< 5 seconds",
            },
            {
                "id": "simplified_holdem",
                "name": "Simplified Hold'em",
                "description": "Heads-up Hold'em with preflop+flop, coarse abstraction. ~2000-5000 info sets.",
                "complexity": "moderate",
                "recommended_iterations": 50000,
                "estimated_time": "1-3 minutes",
            },
        ]

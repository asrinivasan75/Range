"""Training metrics computation and analysis."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingMetrics:
    """Snapshot of training metrics at a point in time."""
    iteration: int
    timestamp: float
    n_info_sets: int
    total_regret: float
    avg_regret: float
    max_regret: float
    exploitability_proxy: float
    iteration_time_ms: float
    ev_estimates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingMetrics:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategySnapshot:
    """A snapshot of the strategy at a given iteration."""
    iteration: int
    timestamp: float
    strategies: dict[str, dict[str, Any]]  # info_set_key -> {strategy, regret, visits}

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "strategies": self.strategies,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategySnapshot:
        return cls(
            iteration=d["iteration"],
            timestamp=d["timestamp"],
            strategies=d["strategies"],
        )


class MetricsCollector:
    """Collects and manages training metrics over time."""

    def __init__(self) -> None:
        self.metrics: list[TrainingMetrics] = []
        self.strategy_snapshots: list[StrategySnapshot] = []
        self._start_time: float = time.time()

    def record_metrics(self, raw: dict) -> TrainingMetrics:
        """Record a metrics snapshot from CFR trainer output."""
        m = TrainingMetrics(
            iteration=raw["iteration"],
            timestamp=time.time() - self._start_time,
            n_info_sets=raw["n_info_sets"],
            total_regret=raw["total_regret"],
            avg_regret=raw["avg_regret"],
            max_regret=raw["max_regret"],
            exploitability_proxy=raw["exploitability_proxy"],
            iteration_time_ms=raw["iteration_time_ms"],
            ev_estimates=raw.get("utilities", []),
        )
        self.metrics.append(m)
        return m

    def record_strategy(self, iteration: int, strategies: dict) -> StrategySnapshot:
        """Record a strategy snapshot."""
        snap = StrategySnapshot(
            iteration=iteration,
            timestamp=time.time() - self._start_time,
            strategies=strategies,
        )
        self.strategy_snapshots.append(snap)
        return snap

    def get_metrics_series(self) -> dict[str, list]:
        """Get time series data for charting."""
        if not self.metrics:
            return {}
        return {
            "iterations": [m.iteration for m in self.metrics],
            "exploitability": [m.exploitability_proxy for m in self.metrics],
            "total_regret": [m.total_regret for m in self.metrics],
            "avg_regret": [m.avg_regret for m in self.metrics],
            "n_info_sets": [m.n_info_sets for m in self.metrics],
            "iteration_time_ms": [m.iteration_time_ms for m in self.metrics],
            "timestamps": [m.timestamp for m in self.metrics],
        }

    def get_summary(self) -> dict:
        """Get a summary of the training run."""
        if not self.metrics:
            return {"status": "no_data"}

        latest = self.metrics[-1]
        first = self.metrics[0]

        total_time = latest.timestamp - first.timestamp if len(self.metrics) > 1 else 0
        avg_iter_time = np.mean([m.iteration_time_ms for m in self.metrics])

        return {
            "total_iterations": latest.iteration,
            "n_info_sets": latest.n_info_sets,
            "final_exploitability": latest.exploitability_proxy,
            "final_avg_regret": latest.avg_regret,
            "total_time_seconds": total_time,
            "avg_iteration_ms": float(avg_iter_time),
            "n_metrics_points": len(self.metrics),
            "n_strategy_snapshots": len(self.strategy_snapshots),
            "regret_reduction": (
                (first.exploitability_proxy - latest.exploitability_proxy)
                / max(first.exploitability_proxy, 1e-10)
                if len(self.metrics) > 1 else 0
            ),
        }

    def save(self, path: Path) -> None:
        """Save metrics to JSON files."""
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "metrics.json", "w") as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)

        with open(path / "summary.json", "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        if self.strategy_snapshots:
            latest = self.strategy_snapshots[-1]
            with open(path / "strategy_latest.json", "w") as f:
                json.dump(latest.to_dict(), f, indent=2)

        with open(path / "metrics_series.json", "w") as f:
            json.dump(self.get_metrics_series(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> MetricsCollector:
        """Load metrics from saved files."""
        collector = cls()
        metrics_file = path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            collector.metrics = [TrainingMetrics.from_dict(d) for d in data]
        return collector

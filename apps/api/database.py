"""SQLite database layer for run metadata and metrics.

Uses aiosqlite for async access. Schema is designed to be
straightforward to migrate to PostgreSQL later.
"""

from __future__ import annotations
import json
import aiosqlite
from pathlib import Path
from typing import Any

DB_PATH = Path("data/range.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    game_type TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    config TEXT NOT NULL,  -- JSON
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    current_iteration INTEGER DEFAULT 0,
    total_iterations INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id),
    iteration INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    n_info_sets INTEGER,
    total_regret REAL,
    avg_regret REAL,
    max_regret REAL,
    exploitability_proxy REAL,
    iteration_time_ms REAL,
    ev_estimates TEXT,  -- JSON array
    UNIQUE(run_id, iteration)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON training_metrics(run_id);
"""


async def init_db(db_path: Path | None = None) -> None:
    """Initialize the database schema."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(path)) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def get_db(db_path: Path | None = None) -> aiosqlite.Connection:
    """Get a database connection."""
    path = db_path or DB_PATH
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    return db


async def save_run(db: aiosqlite.Connection, run: dict) -> None:
    """Insert or update a training run."""
    await db.execute(
        """INSERT OR REPLACE INTO training_runs
        (id, name, game_type, algorithm, config, status, created_at, started_at,
         completed_at, current_iteration, total_iterations, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run["id"], run["name"], run["config"]["game_type"],
            run["config"]["algorithm"], json.dumps(run["config"]),
            run["status"], run["created_at"], run.get("started_at"),
            run.get("completed_at"), run.get("current_iteration", 0),
            run["config"].get("n_iterations", 0), run.get("error"),
        ),
    )
    await db.commit()


async def save_metrics_batch(db: aiosqlite.Connection, run_id: str, metrics: list[dict]) -> None:
    """Save a batch of metrics."""
    for m in metrics:
        await db.execute(
            """INSERT OR REPLACE INTO training_metrics
            (run_id, iteration, timestamp, n_info_sets, total_regret, avg_regret,
             max_regret, exploitability_proxy, iteration_time_ms, ev_estimates)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, m["iteration"], m["timestamp"], m["n_info_sets"],
                m["total_regret"], m["avg_regret"], m["max_regret"],
                m["exploitability_proxy"], m["iteration_time_ms"],
                json.dumps(m.get("ev_estimates", [])),
            ),
        )
    await db.commit()

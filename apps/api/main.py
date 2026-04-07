"""Range API — FastAPI application for the poker solver platform."""

from __future__ import annotations
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.routes import health, runs, training, analysis, play
from apps.api.database import init_db
from packages.solver.trainer import TrainingOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    await init_db()
    app.state.orchestrator = TrainingOrchestrator(data_dir=PROJECT_ROOT / "data")
    yield


app = FastAPI(
    title="Range",
    description="Poker solver and analysis platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3333", "http://127.0.0.1:3000", "http://127.0.0.1:3333"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(play.router, prefix="/api/play", tags=["play"])


@app.get("/api/slumbot-log")
async def get_slumbot_log(mode: str = "training"):
    """Serve Slumbot benchmark hand logs."""
    import json
    LOG_MAP = {
        "rl_v2": "slumbot_rl_v2_log.json",
        "rl": "slumbot_rl_log.json",
        "training": "slumbot_training_log.json",
        "heuristic": "slumbot_log.json",
    }
    filename = LOG_MAP.get(mode, f"slumbot_{mode}_log.json")
    log_path = PROJECT_ROOT / "data" / filename
    if not log_path.exists():
        return {"error": f"No {mode} log found.", "hands": []}
    with open(log_path) as f:
        return json.load(f)


@app.get("/api/slumbot-logs")
async def list_slumbot_logs():
    """List available Slumbot session logs."""
    import json
    logs = []
    # Auto-discover all slumbot log files
    data_dir = PROJECT_ROOT / "data"
    log_files = sorted(data_dir.glob("slumbot_*_log.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in log_files:
        name = path.stem.replace("slumbot_", "").replace("_log", "")
        # Generate readable label from filename
        label = name.replace("_", " ").replace("ql ", "Q-Learn ").replace("ppo ", "PPO ").replace("training", "Training")
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            hands = d.get("hands", [])
            n = len(hands)
            total = sum(h.get("winnings_bb", 0) for h in hands)
            logs.append({
                "id": name,
                "label": label,
                "hands": n,
                "total_bb": round(total, 1),
                "bb_per_100": round(total / n * 100, 1) if n > 0 else 0,
                "timestamp": d.get("config", {}).get("timestamp", ""),
            })
    return {"logs": logs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)

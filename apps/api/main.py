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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)

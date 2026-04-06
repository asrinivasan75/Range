"""Training control endpoints — games, configurations, progress."""

from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/games")
async def list_games(request: Request):
    """List available game types for training."""
    orchestrator = request.app.state.orchestrator
    return {"games": orchestrator.get_available_games()}


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    """Cancel a running training job."""
    orchestrator = request.app.state.orchestrator
    success = orchestrator.cancel_run(run_id)
    return {"cancelled": success}


@router.get("/{run_id}/progress")
async def get_progress(run_id: str, request: Request):
    """Get real-time progress for a training run."""
    orchestrator = request.app.state.orchestrator
    run = orchestrator.get_run(run_id)
    if not run:
        return {"error": "not_found"}

    metrics = orchestrator.get_run_metrics(run_id)
    return {
        "run_id": run_id,
        "status": run["status"],
        "current_iteration": run.get("current_iteration", 0),
        "total_iterations": run["config"]["n_iterations"],
        "progress_pct": (
            run.get("current_iteration", 0) / max(run["config"]["n_iterations"], 1) * 100
        ),
        "latest_metrics": metrics.get("summary") if metrics else None,
    }

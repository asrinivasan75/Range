"""Training runs CRUD endpoints."""

from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from packages.solver.trainer import TrainingConfig

router = APIRouter()


class CreateRunRequest(BaseModel):
    name: str = ""
    game_type: str = "simplified_holdem"
    algorithm: str = "mccfr"
    n_iterations: int = 10000
    preflop_buckets: int = 8
    flop_buckets: int = 8
    max_raises: int = 2


@router.get("")
async def list_runs(request: Request):
    """List all training runs."""
    orchestrator = request.app.state.orchestrator
    return {"runs": orchestrator.list_runs()}


@router.post("")
async def create_run(request: Request, body: CreateRunRequest):
    """Create and start a new training run."""
    orchestrator = request.app.state.orchestrator
    config = TrainingConfig(
        game_type=body.game_type,
        algorithm=body.algorithm,
        n_iterations=body.n_iterations,
        preflop_buckets=body.preflop_buckets,
        flop_buckets=body.flop_buckets,
        max_raises=body.max_raises,
    )
    run = orchestrator.start_training(config=config, name=body.name)
    return {"run": run.to_dict()}


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request):
    """Get details for a specific run."""
    orchestrator = request.app.state.orchestrator
    run = orchestrator.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": run}


@router.get("/{run_id}/metrics")
async def get_run_metrics(run_id: str, request: Request):
    """Get metrics for a run."""
    orchestrator = request.app.state.orchestrator
    metrics = orchestrator.get_run_metrics(run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return metrics


@router.get("/{run_id}/strategy")
async def get_run_strategy(run_id: str, request: Request):
    """Get strategy data for a run."""
    orchestrator = request.app.state.orchestrator
    strategy = orchestrator.get_run_strategy(run_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"strategy": strategy}


@router.delete("/{run_id}")
async def delete_run(run_id: str, request: Request):
    """Delete a training run."""
    orchestrator = request.app.state.orchestrator
    success = orchestrator.delete_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted"}

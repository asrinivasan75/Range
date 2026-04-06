"""Analysis endpoints — equity calculation, hand strength, strategy lookup."""

from __future__ import annotations
import sys
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from packages.poker.card import Card
from packages.poker.evaluator import (
    hand_strength_monte_carlo,
    evaluate_hand,
    chen_formula,
    preflop_hand_bucket,
)
from packages.poker.hand import HandCategory

router = APIRouter()


class EquityRequest(BaseModel):
    hole_cards: list[str]  # e.g. ["Ah", "Ks"]
    board: list[str] = []  # e.g. ["Td", "7c", "2h"]
    n_simulations: int = 5000


class HandStrengthRequest(BaseModel):
    hole_cards: list[str]
    board: list[str] = []


@router.post("/equity")
async def compute_equity(body: EquityRequest):
    """Compute hand equity via Monte Carlo simulation."""
    try:
        hole = [Card.from_str(c) for c in body.hole_cards]
        board = [Card.from_str(c) for c in body.board]
        equity = hand_strength_monte_carlo(hole, board, n_simulations=body.n_simulations)
        chen = chen_formula(hole[0], hole[1]) if len(hole) == 2 else None
        bucket = preflop_hand_bucket(hole[0], hole[1]) if len(hole) == 2 and not board else None

        result = {
            "equity": round(equity, 4),
            "equity_pct": round(equity * 100, 2),
            "hole_cards": body.hole_cards,
            "board": body.board,
            "n_simulations": body.n_simulations,
        }
        if chen is not None:
            result["chen_score"] = round(chen, 1)
        if bucket is not None:
            result["preflop_bucket"] = bucket
        return result

    except Exception as e:
        return {"error": str(e)}


@router.post("/hand-strength")
async def hand_strength(body: HandStrengthRequest):
    """Evaluate hand strength (if 5+ cards available)."""
    try:
        hole = [Card.from_str(c) for c in body.hole_cards]
        board = [Card.from_str(c) for c in body.board]

        result = {"hole_cards": body.hole_cards, "board": body.board}

        if len(hole) + len(board) >= 5:
            category, kickers = evaluate_hand(hole, board)
            result["hand_category"] = str(HandCategory(category))
            result["hand_rank"] = category.value
            result["kickers"] = list(kickers)

        if len(hole) == 2:
            result["chen_score"] = round(chen_formula(hole[0], hole[1]), 1)
            result["preflop_bucket"] = preflop_hand_bucket(hole[0], hole[1])

        # Monte Carlo equity
        equity = hand_strength_monte_carlo(hole, board, n_simulations=2000)
        result["equity"] = round(equity, 4)
        result["equity_pct"] = round(equity * 100, 2)

        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/preflop-chart")
async def preflop_chart():
    """Get the full preflop hand strength chart."""
    from packages.solver.abstractions import get_preflop_buckets

    buckets = get_preflop_buckets(8)

    # Build a 13x13 grid (rows = high card, cols = low card)
    from packages.poker.card import Rank
    ranks = list(reversed(list(Rank)))  # A, K, Q, ..., 2

    grid = []
    for i, r1 in enumerate(ranks):
        row = []
        for j, r2 in enumerate(ranks):
            if i == j:
                # Pair
                label = f"{r1}{r2}"
            elif i < j:
                # Suited (above diagonal)
                label = f"{r1}{r2}s"
            else:
                # Off-suit (below diagonal)
                label = f"{r2}{r1}o"
            bucket = buckets.get(label, 0)
            row.append({"hand": label, "bucket": bucket})
        grid.append(row)

    return {
        "grid": grid,
        "n_buckets": 8,
        "ranks": [str(r) for r in ranks],
    }

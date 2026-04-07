"""Play endpoints — interactive poker against the trained bot.

Uses the BettingEngine state machine for all game logic.
Server stores hand state keyed by hand_id — frontend sends only actions.
"""

from __future__ import annotations
import uuid
import json
import time
import random
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from packages.poker.card import Card
from packages.poker.deck import Deck
from packages.poker.evaluator import evaluate_hand, hand_strength_monte_carlo, chen_formula
from packages.poker.hand import HandCategory
from packages.poker.game_state import Street
from packages.poker.betting_engine import BettingEngine, BettingState, ActionKind, LegalAction
from packages.poker.sizing import (
    compute_sizing_candidates, select_bot_sizing, SizingCandidate, streets_remaining,
)
from packages.solver.abstractions import get_hand_bucket_preflop, compute_flop_bucket
from packages.solver.holdem_simplified import ACTION_NAMES, BET_FRACTIONS
from packages.solver.rl_agent import (
    QLearningAgent, extract_features as rl_extract_features,
    get_legal_mask as rl_get_legal_mask,
)
from packages.solver.neural_agent import (
    PPOAgent, extract_features as ppo_extract_features,
    get_legal_mask as ppo_get_legal_mask,
)
from packages.solver.range_estimator import RangeEstimator

_range_estimator = RangeEstimator()

router = APIRouter()

# Server-side hand storage
_active_hands: dict[str, dict] = {}

# Session logging for play vs bot
_play_sessions: dict[str, dict] = {}  # session_id -> {hands: [], ...}

# Cached Q-learning agents
_ql_agents: dict[str, QLearningAgent] = {}


def _get_ql_agent(weights_path: str) -> QLearningAgent | None:
    """Load or get cached Q-learning agent."""
    if weights_path not in _ql_agents:
        agent = QLearningAgent()
        if agent.load(weights_path):
            _ql_agents[weights_path] = agent
        else:
            return None
    return _ql_agents[weights_path]


_ppo_agents: dict[str, PPOAgent] = {}

def _get_ppo_agent(weights_path: str) -> PPOAgent | None:
    """Load or get cached PPO agent."""
    if weights_path not in _ppo_agents:
        agent = PPOAgent()
        if agent.load(weights_path):
            _ppo_agents[weights_path] = agent
        else:
            return None
    return _ppo_agents[weights_path]


# ── Request models ───────────────────────────────────────────

class NewHandRequest(BaseModel):
    run_id: str = ""
    starting_stack: float = 100.0
    bot_type: str = "heuristic"  # "heuristic" | "ql" | "ppo"
    weights_path: str = ""       # path to QL/PPO weights file
    session_id: str = ""         # empty = auto-create new session

class PlayerActionRequest(BaseModel):
    hand_id: str
    action_type: str   # "fold" | "check" | "call" | "bet" | "raise" | "all_in"
    amount: float = 0  # required for bet/raise

class AdvisorRequest(BaseModel):
    hand_id: str


# ── Helpers ──────────────────────────────────────────────────

ACTION_KIND_MAP = {
    "fold": ActionKind.FOLD,
    "check": ActionKind.CHECK,
    "call": ActionKind.CALL,
    "bet": ActionKind.BET,
    "raise": ActionKind.RAISE,
    "all_in": ActionKind.ALL_IN,
}

E = BettingEngine


def _format_legal_actions(actions: list[LegalAction]) -> list[dict]:
    """Convert LegalAction objects to JSON-friendly dicts for the frontend."""
    result = []
    for a in actions:
        d: dict = {"type": a.kind.name.lower(), "label": a.describe()}
        if a.kind == ActionKind.CALL:
            d["amount"] = round(a.amount, 1)
        elif a.kind in (ActionKind.BET, ActionKind.RAISE):
            d["min_amount"] = round(a.min_amount, 1)
            d["max_amount"] = round(a.max_amount, 1)
        elif a.kind == ActionKind.ALL_IN:
            d["amount"] = round(a.amount, 1)
        result.append(d)
    return result


def _format_sizing(candidates: list[SizingCandidate]) -> list[dict]:
    return [
        {"name": c.name, "amount": round(c.amount, 1), "fraction": round(c.fraction, 2), "rationale": c.rationale}
        for c in candidates
    ]


def _board_for_street(street: Street, hand_data: dict) -> list[Card]:
    cards = []
    if street.value >= Street.FLOP.value:
        cards.extend(hand_data["flop"])
    if street.value >= Street.TURN.value:
        cards.append(hand_data["turn"])
    if street.value >= Street.RIVER.value:
        cards.append(hand_data["river"])
    return cards


def _build_response(hand_id: str, hand_data: dict, bot_action_info: dict | None = None) -> dict:
    """Build the standard response from current hand state."""
    state: BettingState = hand_data["state"]
    player_idx = hand_data["player_idx"]
    bot_idx = hand_data["bot_idx"]

    legal = E.get_legal_actions(state) if not state.is_hand_complete and state.current_player_idx == player_idx else []
    sizing = compute_sizing_candidates(state, legal) if legal else []

    # Determine board to show
    board: list[str] = []
    if state.street.value >= Street.FLOP.value or state.is_hand_complete:
        board = [c.ascii for c in hand_data["flop"]]
    if state.street.value >= Street.TURN.value or (state.is_hand_complete and state.street.value >= Street.TURN.value):
        board.append(hand_data["turn"].ascii)
    if state.street.value >= Street.RIVER.value or (state.is_hand_complete and state.street.value >= Street.RIVER.value):
        board.append(hand_data["river"].ascii)

    # Show bot cards whenever hand is over (including folds — let player see what bot had)
    reveal = state.is_hand_complete
    bot_cards = [c.ascii for c in hand_data["bot_cards"]] if reveal else ["??", "??"]

    # Action log
    log = [rec.describe(("You", "Bot") if player_idx == 0 else ("Bot", "You")) for rec in state.action_log]

    # Winner message
    message = ""
    winner = None
    if state.is_hand_complete:
        non_folded = state.non_folded_indices
        if len(non_folded) == 1:
            winner = "player" if non_folded[0] == player_idx else "bot"
            message = "You win!" if winner == "player" else "Bot wins."
        else:
            # Showdown
            board_cards = _board_for_street(state.street, hand_data)
            if len(hand_data["player_cards"]) + len(board_cards) >= 5:
                p_hand = evaluate_hand(list(hand_data["player_cards"]), board_cards)
                b_hand = evaluate_hand(list(hand_data["bot_cards"]), board_cards)
                p_cat = HandCategory(p_hand[0])
                b_cat = HandCategory(b_hand[0])
                if p_hand > b_hand:
                    winner, message = "player", f"You win! {p_cat} beats {b_cat}."
                elif b_hand > p_hand:
                    winner, message = "bot", f"Bot wins. {b_cat} beats {p_cat}."
                else:
                    winner, message = "tie", f"Split pot. Both have {p_cat}."
            else:
                winner, message = "tie", "Showdown (insufficient board cards)."

    resp = {
        "hand_id": hand_id,
        "street": state.street.name.lower(),
        "pot": round(state.pot, 1),
        "player_stack": round(state.players[player_idx].stack, 1),
        "bot_stack": round(state.players[bot_idx].stack, 1),
        "player_bet": round(state.players[player_idx].bet_this_street, 1),
        "bot_bet": round(state.players[bot_idx].bet_this_street, 1),
        "player_cards": [c.ascii for c in hand_data["player_cards"]],
        "bot_cards": bot_cards,
        "board": board,
        "legal_actions": _format_legal_actions(legal),
        "sizing_suggestions": _format_sizing(sizing),
        "is_player_turn": not state.is_hand_complete and state.current_player_idx == player_idx,
        "is_terminal": state.is_hand_complete,
        "winner": winner,
        "message": message,
        "bot_action": bot_action_info,
        "action_log": log,
        "session_id": hand_data.get("session_id", ""),
    }

    # Log completed hands to session
    if state.is_hand_complete and hand_data.get("session_id"):
        sid = hand_data["session_id"]
        if sid in _play_sessions:
            hand_log = {
                "hand_number": len(_play_sessions[sid]["hands"]) + 1,
                "our_cards": [c.ascii for c in hand_data["player_cards"]],
                "bot_cards": [c.ascii for c in hand_data["bot_cards"]],
                "board_flop": [c.ascii for c in hand_data["flop"]],
                "board_turn": hand_data["turn"].ascii,
                "board_river": hand_data["river"].ascii,
                "our_position": "BTN" if state.button_seat == player_idx else "BB",
                "winner": winner,
                "winnings_bb": round(state.players[player_idx].total_invested - state.players[bot_idx].total_invested, 1) if winner == "player" else round(-(state.players[player_idx].total_invested), 1) if winner == "bot" else 0,
                "actions": log,
                "final_pot": round(state.pot, 1),
            }
            # Compute actual winnings from pot
            if winner == "player":
                hand_log["winnings_bb"] = round((state.pot - state.players[player_idx].total_invested) / 2.0, 1)
            elif winner == "bot":
                hand_log["winnings_bb"] = round(-state.players[player_idx].total_invested / 2.0, 1)
            else:
                hand_log["winnings_bb"] = 0

            _play_sessions[sid]["hands"].append(hand_log)

            # Save session to disk
            _save_play_session(sid)

    return resp


def _save_play_session(session_id: str):
    """Save play session to disk for the replayer."""
    if session_id not in _play_sessions:
        return
    session = _play_sessions[session_id]
    hands = session["hands"]
    n = len(hands)
    total_bb = sum(h.get("winnings_bb", 0) for h in hands)
    running = []
    cum = 0
    for h in hands:
        cum += h.get("winnings_bb", 0)
        running.append(round(cum, 1))

    data = {
        "summary": {
            "hands_played": n,
            "total_bb": round(total_bb, 1),
            "bb_per_100": round(total_bb / max(n, 1) * 100, 1),
            "win_count": sum(1 for h in hands if h.get("winnings_bb", 0) > 0),
            "loss_count": sum(1 for h in hands if h.get("winnings_bb", 0) < 0),
            "win_pct": round(sum(1 for h in hands if h.get("winnings_bb", 0) > 0) / max(n, 1) * 100, 1),
            "biggest_win": round(max((h.get("winnings_bb", 0) for h in hands), default=0), 1),
            "biggest_loss": round(min((h.get("winnings_bb", 0) for h in hands), default=0), 1),
            "running_total": running,
        },
        "hands": hands,
        "config": {
            "opponent": f"Bot ({session.get('bot_type', 'heuristic')})",
            "blinds": "1/2",
            "stack_bb": 100,
            "timestamp": session.get("started", ""),
        },
    }
    path = Path("data") / f"slumbot_play_{session_id}_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _get_bot_action(
    hand_data: dict,
    state: BettingState,
    request_app,
) -> tuple[ActionKind, float, dict | None]:
    """Select the bot's action.

    Priority: Q-learning agent (if specified) > CFR strategy > equity fallback.
    Returns (action_kind, amount, display_info_or_None).
    """
    bot_cards = list(hand_data["bot_cards"])
    board = _board_for_street(state.street, hand_data)
    legal = E.get_legal_actions(state)

    if not legal:
        return ActionKind.CHECK, 0, None

    legal_kinds = {a.kind for a in legal}

    # ── Try PPO neural network agent if specified ──────────────
    weights_path = hand_data.get("weights_path", "")
    if weights_path and hand_data.get("bot_type") == "ppo":
        ppo = _get_ppo_agent(weights_path)
        if ppo:
            result = _ppo_agent_action(ppo, bot_cards, board, state, legal)
            if result:
                return result

    # ── Try Q-learning agent if specified ────────────────────
    if weights_path and hand_data.get("bot_type") == "ql":
        agent = _get_ql_agent(weights_path)
        if agent:
            result = _ql_agent_action(agent, bot_cards, board, state, legal)
            if result:
                return result

    # ── Try trained strategy lookup ──────────────────────────
    orchestrator = request_app.state.orchestrator
    strategy = orchestrator.get_run_strategy(hand_data.get("run_id", ""))

    if strategy:
        result = _try_strategy_lookup(strategy, bot_cards, board, state, legal)
        if result:
            return result

    # ── Fallback: equity-based with proper pot odds ──────────
    return _equity_fallback(bot_cards, board, state, legal, legal_kinds)


def _ppo_agent_action(
    agent: PPOAgent,
    bot_cards: list[Card],
    board: list[Card],
    state: BettingState,
    legal: list[LegalAction],
) -> tuple[ActionKind, float, dict | None] | None:
    """Use PPO neural network agent to pick an action with range-adjusted equity."""
    bot_idx = 1
    player_idx = 0
    pot = int(state.pot)
    to_call = int(state.to_call)
    invested = int(state.players[bot_idx].total_invested)
    remaining = int(state.players[bot_idx].stack)
    is_btn = (state.button_seat == bot_idx)
    n_bets = sum(1 for r in state.action_log if r.street == state.street
                 and r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN))

    # Build opponent action history for range estimation
    opp_actions = []
    for rec in state.action_log:
        if rec.player_idx == player_idx and rec.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.CALL, ActionKind.CHECK, ActionKind.ALL_IN):
            opp_actions.append({
                "type": "raise" if rec.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN) else
                        "call" if rec.kind == ActionKind.CALL else "check",
                "street": rec.street.value,
                "amount": rec.amount,
                "pot": pot,
            })

    # Compute range-adjusted equity
    range_eq = None
    if opp_actions:
        try:
            range_eq = _range_estimator.compute_equity_vs_opponent(
                bot_cards, board, opp_actions, state.street.value,
                position="oop" if is_btn else "ip", n_samples=150,
            )
        except:
            pass

    features = ppo_extract_features(
        bot_cards, board, pot, to_call, invested, remaining,
        state.street.value, is_btn, n_bets,
        range_adjusted_equity=range_eq,
    )
    legal_mask = ppo_get_legal_mask(to_call, remaining)

    # Get action probabilities for display
    probs = agent.get_action_probs(features, legal_mask)
    action_idx, _, _ = agent.choose_action(features, legal_mask)

    from packages.solver.neural_agent import ACTIONS as PPO_ACTIONS
    prob_dict = {PPO_ACTIONS[i]: f"{probs[i]*100:.1f}%" for i in range(len(PPO_ACTIONS)) if legal_mask[i] > 0}

    action_name = PPO_ACTIONS[action_idx]
    legal_kinds = {a.kind for a in legal}

    if action_name == "fold" and ActionKind.FOLD in legal_kinds:
        return ActionKind.FOLD, 0, {"label": "Folds", "source": "ppo_agent", "probabilities": prob_dict}
    if action_name == "check" and ActionKind.CHECK in legal_kinds:
        return ActionKind.CHECK, 0, {"label": "Checks", "source": "ppo_agent", "probabilities": prob_dict}
    if action_name == "call":
        call = next((a for a in legal if a.kind == ActionKind.CALL), None)
        if call:
            return ActionKind.CALL, call.amount, {"label": f"Calls {call.amount:.1f}", "source": "ppo_agent", "probabilities": prob_dict}
    if action_name == "all_in":
        if ActionKind.RAISE in legal_kinds:
            raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
            return ActionKind.RAISE, raise_a.max_amount, {"label": f"All-in {raise_a.max_amount:.1f}", "source": "ppo_agent", "probabilities": prob_dict}
        if ActionKind.BET in legal_kinds:
            bet_a = next(a for a in legal if a.kind == ActionKind.BET)
            return ActionKind.BET, bet_a.max_amount, {"label": f"All-in {bet_a.max_amount:.1f}", "source": "ppo_agent", "probabilities": prob_dict}

    # Bet/raise sizing
    frac = {"bet_small": 0.33, "bet_medium": 0.67, "bet_large": 1.0}.get(action_name, 0.5)
    if ActionKind.BET in legal_kinds:
        bet = next(a for a in legal if a.kind == ActionKind.BET)
        amount = max(bet.min_amount, min(state.pot * frac, bet.max_amount))
        return ActionKind.BET, round(amount, 1), {"label": f"Bets {amount:.1f}", "source": "ppo_agent", "probabilities": prob_dict}
    if ActionKind.RAISE in legal_kinds:
        raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
        amount = max(raise_a.min_amount, min(state.highest_bet + state.pot * frac, raise_a.max_amount))
        return ActionKind.RAISE, round(amount, 1), {"label": f"Raises to {amount:.1f}", "source": "ppo_agent", "probabilities": prob_dict}

    return None


def _ql_agent_action(
    agent: QLearningAgent,
    bot_cards: list[Card],
    board: list[Card],
    state: BettingState,
    legal: list[LegalAction],
) -> tuple[ActionKind, float, dict | None] | None:
    """Use Q-learning agent to pick an action with range-adjusted equity."""
    bot_idx = 1
    player_idx = 0
    pot = int(state.pot)
    to_call = int(state.to_call)
    invested = int(state.players[bot_idx].total_invested)
    remaining = int(state.players[bot_idx].stack)
    is_btn = (state.button_seat == bot_idx)
    n_bets = sum(1 for r in state.action_log if r.street == state.street and r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN))

    features = rl_extract_features(
        bot_cards, board, pot, to_call,
        invested, remaining, state.street.value, is_btn, n_bets,
    )

    # Override equity with range-adjusted version
    opp_actions = [
        {"type": "raise" if r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN) else
                 "call" if r.kind == ActionKind.CALL else "check",
         "street": r.street.value, "amount": r.amount, "pot": pot}
        for r in state.action_log if r.player_idx == player_idx
        and r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.CALL, ActionKind.CHECK, ActionKind.ALL_IN)
    ]
    if opp_actions:
        try:
            range_eq = _range_estimator.compute_equity_vs_opponent(
                bot_cards, board, opp_actions, state.street.value,
                position="oop" if is_btn else "ip", n_samples=100,
            )
            features[0] = range_eq  # replace raw equity with range-adjusted
        except:
            pass

    legal_mask = rl_get_legal_mask(to_call, remaining)
    action_idx = agent.choose_action(features, legal_mask)

    # Get probabilities for display
    from packages.solver.rl_agent import ACTIONS
    probs = agent.get_q_values(features)
    probs_masked = {ACTIONS[i]: round(float(probs[i]), 2) for i in range(len(ACTIONS)) if legal_mask[i] > 0}

    # Map RL action to BettingEngine action
    rl_action = ["fold", "check", "call", "bet_small", "bet_medium", "bet_large"][action_idx]
    legal_kinds = {a.kind for a in legal}

    if rl_action == "fold" and ActionKind.FOLD in legal_kinds:
        return ActionKind.FOLD, 0, {"label": "Folds", "source": "ql_agent", "q_values": probs_masked}

    if rl_action == "check" and ActionKind.CHECK in legal_kinds:
        return ActionKind.CHECK, 0, {"label": "Checks", "source": "ql_agent", "q_values": probs_masked}

    if rl_action == "call":
        call = next((a for a in legal if a.kind == ActionKind.CALL), None)
        if call:
            return ActionKind.CALL, call.amount, {"label": f"Calls {call.amount:.1f}", "source": "ql_agent", "q_values": probs_masked}

    # Bet/raise sizing
    if rl_action in ("bet_small", "bet_medium", "bet_large"):
        frac = {"bet_small": 0.33, "bet_medium": 0.67, "bet_large": 1.0}[rl_action]

        if ActionKind.BET in legal_kinds:
            bet = next(a for a in legal if a.kind == ActionKind.BET)
            amount = max(bet.min_amount, min(state.pot * frac, bet.max_amount))
            return ActionKind.BET, round(amount, 1), {"label": f"Bets {amount:.1f}", "source": "ql_agent", "q_values": probs_masked}

        if ActionKind.RAISE in legal_kinds:
            raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
            amount = max(raise_a.min_amount, min(state.highest_bet + state.pot * frac, raise_a.max_amount))
            return ActionKind.RAISE, round(amount, 1), {"label": f"Raises to {amount:.1f}", "source": "ql_agent", "q_values": probs_masked}

    return None  # fallback to other methods


def _try_strategy_lookup(
    strategy: dict,
    bot_cards: list[Card],
    board: list[Card],
    state: BettingState,
    legal: list[LegalAction],
) -> tuple[ActionKind, float, dict | None] | None:
    """Attempt to look up trained strategy. Returns None if no match."""
    from packages.solver.holdem_full import BET_FRACTIONS as FULL_FRACS, ACTION_NAMES as FULL_NAMES

    pre_bucket = get_hand_bucket_preflop(bot_cards[0], bot_cards[1], n_buckets=5)

    # Build info set key matching full-street solver format
    # Format: P<b>F<b>T<b>R<b>|<preflop_acts>/<flop_acts>/<turn_acts>/<river_acts>
    bucket_str = f"P{pre_bucket}"
    if state.street.value >= Street.FLOP.value and len(board) >= 3:
        flop_bucket = compute_flop_bucket(bot_cards, board[:3], n_buckets=5, n_simulations=150)
        bucket_str += f"F{flop_bucket}"
    if state.street.value >= Street.TURN.value and len(board) >= 4:
        turn_bucket = compute_flop_bucket(bot_cards, board[:4], n_buckets=5, n_simulations=150)
        bucket_str += f"T{turn_bucket}"
    if state.street.value >= Street.RIVER.value and len(board) >= 5:
        river_bucket = compute_flop_bucket(bot_cards, board[:5], n_buckets=5, n_simulations=150)
        bucket_str += f"R{river_bucket}"

    # Build action history from the action log
    action_str = _build_action_string_from_log(state)
    info_key = f"{bucket_str}|{action_str}"

    entry = strategy.get(info_key)
    if not entry:
        return None

    strat_probs = entry["strategy"]

    # Determine what actions the solver trained with
    # The solver has: no bet -> [k, t, p] or [k, s, p]; facing bet -> [f, c] or [f, c, r]
    solver_actions = _infer_solver_actions(state, action_str)

    if len(strat_probs) != len(solver_actions):
        return None  # mismatch

    # Sample from the strategy distribution
    idx = random.choices(range(len(solver_actions)), weights=strat_probs)[0]
    chosen_code = solver_actions[idx]

    # Map solver action to engine action + amount
    kind, amount = _map_solver_action_to_engine(chosen_code, state, legal)
    if kind is None:
        return None

    # Build display info
    prob_dict = {FULL_NAMES.get(a, a): round(p * 100, 1) for a, p in zip(solver_actions, strat_probs)}
    info = {"label": _action_to_label(kind, amount), "probabilities": prob_dict, "source": "trained"}
    return kind, amount, info


def _build_action_string_from_log(state: BettingState) -> str:
    """Reconstruct the action string from the BettingState action log for strategy lookup."""
    # Group actions by street, skip blind posts
    streets: list[list[str]] = [[] for _ in range(4)]
    for rec in state.action_log:
        if rec.street.value < 4:
            # Map ActionKind to solver action character
            char = _action_kind_to_solver_char(rec, state)
            if char:
                streets[rec.street.value].append(char)

    # Join with /
    parts = []
    for s in range(state.street.value + 1):
        parts.append("".join(streets[s]))
    return "/".join(parts)


def _action_kind_to_solver_char(rec, state: BettingState) -> str | None:
    """Map an ActionRecord to a solver action character."""
    from packages.poker.betting_engine import ActionKind as AK
    if rec.kind == AK.FOLD:
        return "f"
    if rec.kind == AK.CHECK:
        return "k"
    if rec.kind == AK.CALL:
        return "c"
    if rec.kind in (AK.BET, AK.RAISE, AK.ALL_IN):
        # Blind posts are recorded as BET — skip them (first 2 in preflop)
        # Check if this is a blind post (amount <= big_blind and it's one of the first two log entries)
        if rec.street == Street.PREFLOP and rec.amount <= state.big_blind:
            # Count how many actions before this one on preflop
            preflop_count = sum(1 for r in state.action_log if r.street == Street.PREFLOP and id(r) < id(rec))
            if preflop_count < 2:
                return None  # skip blind post

        if rec.kind == AK.RAISE:
            return "r"
        # For bets, map to closest solver action
        # This is approximate — the solver uses t(1/3), s(2/3), p(pot)
        return "t"  # default to small bet; the exact mapping doesn't need to be perfect
                     # since we're looking up the INFO SET, not reconstructing the tree
    return None


def _infer_solver_actions(state: BettingState, action_str: str) -> list[str]:
    """Determine what actions the solver would offer at this decision point."""
    current_street_actions = action_str.split("/")[-1] if "/" in action_str else action_str
    n_agg = sum(1 for a in current_street_actions if a in "tspr")
    has_bet = n_agg > 0
    if not has_bet:
        return ["k", "t", "p"]  # check, 1/3, pot
    elif n_agg >= 1:  # max_raises=1 in the full solver
        return ["f", "c"]
    else:
        return ["f", "c", "r"]


def _map_solver_action_to_engine(
    solver_action: str,
    state: BettingState,
    legal: list[LegalAction],
) -> tuple[ActionKind | None, float]:
    """Map a solver action code to a concrete (ActionKind, amount)."""
    legal_kinds = {a.kind for a in legal}

    if solver_action == "f":
        return (ActionKind.FOLD, 0) if ActionKind.FOLD in legal_kinds else (None, 0)
    if solver_action == "k":
        return (ActionKind.CHECK, 0) if ActionKind.CHECK in legal_kinds else (None, 0)
    if solver_action == "c":
        call = next((a for a in legal if a.kind == ActionKind.CALL), None)
        return (ActionKind.CALL, call.amount) if call else (None, 0)

    # Bet/raise with sizing
    frac_map = {"t": 1/3, "s": 2/3, "p": 1.0}
    frac = frac_map.get(solver_action, 0.5)

    if ActionKind.BET in legal_kinds:
        bet = next(a for a in legal if a.kind == ActionKind.BET)
        amount = max(bet.min_amount, min(state.pot * frac, bet.max_amount))
        return ActionKind.BET, round(amount, 1)
    if ActionKind.RAISE in legal_kinds:
        raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
        amount = max(raise_a.min_amount, min(state.highest_bet + state.pot * frac, raise_a.max_amount))
        return ActionKind.RAISE, round(amount, 1)

    if solver_action == "r" and ActionKind.RAISE in legal_kinds:
        raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
        return ActionKind.RAISE, round(raise_a.min_amount, 1)

    return None, 0


def _action_to_label(kind: ActionKind, amount: float) -> str:
    if kind == ActionKind.FOLD:
        return "Folds"
    if kind == ActionKind.CHECK:
        return "Checks"
    if kind == ActionKind.CALL:
        return f"Calls {amount:.1f}"
    if kind == ActionKind.BET:
        return f"Bets {amount:.1f}"
    if kind == ActionKind.RAISE:
        return f"Raises to {amount:.1f}"
    if kind == ActionKind.ALL_IN:
        return f"All-in {amount:.1f}"
    return str(kind)


def _equity_fallback(
    bot_cards: list[Card],
    board: list[Card],
    state: BettingState,
    legal: list[LegalAction],
    legal_kinds: set[ActionKind],
) -> tuple[ActionKind, float, dict | None]:
    """Equity-based fallback with proper pot odds logic. Used when no trained strategy matches."""
    equity = hand_strength_monte_carlo(bot_cards, board, n_simulations=500)
    spr = E.spr(state)

    if ActionKind.CHECK in legal_kinds:
        if equity > 0.65 or (equity > 0.4 and random.random() < 0.3):
            sizing = select_bot_sizing(state, equity, legal)
            if sizing:
                bet_action = next((a for a in legal if a.kind == ActionKind.BET), None)
                if bet_action:
                    amount = max(bet_action.min_amount, min(sizing.amount, bet_action.max_amount))
                    return ActionKind.BET, round(amount, 1), {"label": f"Bets {amount:.1f}", "source": "heuristic"}
        return ActionKind.CHECK, 0, {"label": "Checks", "source": "heuristic"}

    elif ActionKind.FOLD in legal_kinds:
        call_action = next((a for a in legal if a.kind == ActionKind.CALL), None)
        call_amount = call_action.amount if call_action else state.to_call
        pot_odds = call_amount / (state.pot + call_amount) if (state.pot + call_amount) > 0 else 0.5

        # Check hand strength on river
        has_made_hand = False
        if len(board) >= 3:
            hand_result = evaluate_hand(bot_cards, board)
            has_made_hand = hand_result[0].value >= 1

        # Discount equity for opponent aggression
        discount = {Street.PREFLOP: 0.85, Street.FLOP: 0.75, Street.TURN: 0.65, Street.RIVER: 0.55}
        adj_equity = equity * discount.get(state.street, 0.7)
        bet_ratio = call_amount / state.pot if state.pot > 0 else 1.0

        should_fold = False
        if state.street == Street.RIVER and not has_made_hand:
            should_fold = pot_odds > 0.15 or bet_ratio > 0.25
        elif adj_equity < pot_odds:
            should_fold = True
        elif adj_equity < 0.25 and bet_ratio > 0.5:
            should_fold = True

        if should_fold:
            return ActionKind.FOLD, 0, {"label": "Folds", "source": "heuristic"}

        if adj_equity > 0.7 and ActionKind.RAISE in legal_kinds:
            sizing = select_bot_sizing(state, equity, legal)
            if sizing:
                raise_a = next((a for a in legal if a.kind == ActionKind.RAISE), None)
                if raise_a:
                    amount = max(raise_a.min_amount, min(sizing.amount, raise_a.max_amount))
                    return ActionKind.RAISE, round(amount, 1), {"label": f"Raises to {amount:.1f}", "source": "heuristic"}

        if call_action:
            return ActionKind.CALL, call_action.amount, {"label": f"Calls {call_action.amount:.1f}", "source": "heuristic"}

        return ActionKind.FOLD, 0, {"label": "Folds", "source": "heuristic"}

    first = legal[0]
    return first.kind, first.amount, {"label": first.describe(), "source": "heuristic"}


def _run_bot_turns(hand_id: str, hand_data: dict, request_app) -> dict | None:
    """If it's the bot's turn, execute bot actions (including across street transitions).

    Returns the last bot_action_info, or None if bot didn't act.
    """
    last_bot_info = None
    max_iterations = 10  # safety limit

    for _ in range(max_iterations):
        state: BettingState = hand_data["state"]

        if state.is_hand_complete:
            break

        # Check if street is complete — advance if so
        if E.is_street_complete(state):
            state = E.advance_street(state)
            hand_data["state"] = state
            if state.is_hand_complete:
                break
            continue

        # Is it the bot's turn?
        if state.current_player_idx != hand_data["bot_idx"]:
            break  # player's turn

        # Bot acts
        kind, amount, info = _get_bot_action(hand_data, state, request_app)
        state = E.apply_action(state, kind, amount)
        hand_data["state"] = state
        last_bot_info = info

    return last_bot_info


# ── Endpoints ────────────────────────────────────────────────

@router.post("/new-hand")
async def new_hand(body: NewHandRequest, request: Request):
    """Deal a new hand and return initial state."""
    hand_id = str(uuid.uuid4())[:8]

    deck = Deck()
    player_cards = deck.deal(2)
    bot_cards = deck.deal(2)
    flop = deck.deal(3)
    turn_card = deck.deal_one()
    river_card = deck.deal_one()

    # Player is always seat 0, bot is seat 1
    # Button alternates — for now, player is button (acts first preflop)
    button_seat = 0
    stack = body.starting_stack

    state = E.create_hand(
        button_seat=button_seat,
        stacks=(stack, stack),
        sb=1.0,
        bb=2.0,
    )

    # Session management
    session_id = body.session_id
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    if session_id not in _play_sessions:
        _play_sessions[session_id] = {
            "id": session_id,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type": body.bot_type,
            "hands": [],
        }

    hand_data = {
        "state": state,
        "player_idx": 0,
        "bot_idx": 1,
        "player_cards": player_cards,
        "bot_cards": bot_cards,
        "flop": flop,
        "turn": turn_card,
        "river": river_card,
        "run_id": body.run_id,
        "bot_type": body.bot_type,
        "weights_path": body.weights_path,
        "session_id": session_id,
    }
    _active_hands[hand_id] = hand_data

    # If bot acts first (when player is NOT on the button), run bot turns
    bot_info = _run_bot_turns(hand_id, hand_data, request.app)

    return _build_response(hand_id, hand_data, bot_info)


@router.post("/act")
async def player_act(body: PlayerActionRequest, request: Request):
    """Process the player's action, then run bot turns."""
    hand_data = _active_hands.get(body.hand_id)
    if not hand_data:
        raise HTTPException(404, "Hand not found. Deal a new hand.")

    state: BettingState = hand_data["state"]
    player_idx = hand_data["player_idx"]

    if state.is_hand_complete:
        raise HTTPException(400, "Hand is already complete.")

    if state.current_player_idx != player_idx:
        raise HTTPException(400, "Not your turn.")

    # Map action type
    kind = ACTION_KIND_MAP.get(body.action_type)
    if not kind:
        raise HTTPException(400, f"Unknown action type: {body.action_type}")

    # Validate action is legal
    legal = E.get_legal_actions(state)
    legal_kinds = {a.kind for a in legal}
    if kind not in legal_kinds:
        raise HTTPException(400, f"Illegal action: {body.action_type}. Legal: {[a.kind.name for a in legal]}")

    # Apply player action
    amount = body.amount
    if kind == ActionKind.CALL:
        call_action = next(a for a in legal if a.kind == ActionKind.CALL)
        amount = call_action.amount
    elif kind == ActionKind.ALL_IN:
        amount = 0  # engine handles it

    state = E.apply_action(state, kind, amount)
    hand_data["state"] = state

    # Check street completion and advance
    if not state.is_hand_complete and E.is_street_complete(state):
        state = E.advance_street(state)
        hand_data["state"] = state

    # Run bot turns (may include multiple streets if bot acts first postflop)
    bot_info = _run_bot_turns(body.hand_id, hand_data, request.app)

    # After bot, check street completion again
    state = hand_data["state"]
    if not state.is_hand_complete and E.is_street_complete(state):
        state = E.advance_street(state)
        hand_data["state"] = state

    return _build_response(body.hand_id, hand_data, bot_info)


@router.post("/advisor")
async def get_advice(body: AdvisorRequest, request: Request):
    """Get solver's recommendation for the current spot."""
    hand_data = _active_hands.get(body.hand_id)
    if not hand_data:
        raise HTTPException(404, "Hand not found.")

    state: BettingState = hand_data["state"]
    player_cards = list(hand_data["player_cards"])
    board = _board_for_street(state.street, hand_data)

    equity = hand_strength_monte_carlo(player_cards, board, n_simulations=1000)
    chen = chen_formula(player_cards[0], player_cards[1])
    pre_bucket = get_hand_bucket_preflop(player_cards[0], player_cards[1], n_buckets=8)

    legal = E.get_legal_actions(state)
    sizing = compute_sizing_candidates(state, legal)

    # Select recommended action
    spr = E.spr(state)
    legal_kinds = {a.kind for a in legal}

    if ActionKind.CHECK in legal_kinds:
        if equity > 0.65:
            rec_kind = ActionKind.BET
            sizing_pick = select_bot_sizing(state, equity, legal)
            rec_label = f"Bet {sizing_pick.amount:.1f}" if sizing_pick else "Bet"
        elif equity > 0.45:
            rec_kind = ActionKind.BET
            sizing_pick = select_bot_sizing(state, equity, legal)
            rec_label = f"Bet {sizing_pick.amount:.1f}" if sizing_pick else "Bet small"
        else:
            rec_kind = ActionKind.CHECK
            rec_label = "Check"
    elif ActionKind.FOLD in legal_kinds:
        if equity > 0.6 and ActionKind.RAISE in legal_kinds:
            rec_kind = ActionKind.RAISE
            sizing_pick = select_bot_sizing(state, equity, legal)
            rec_label = f"Raise to {sizing_pick.amount:.1f}" if sizing_pick else "Raise"
        elif equity > 0.3:
            rec_kind = ActionKind.CALL
            call = next((a for a in legal if a.kind == ActionKind.CALL), None)
            rec_label = f"Call {call.amount:.1f}" if call else "Call"
        else:
            rec_kind = ActionKind.FOLD
            rec_label = "Fold"
    else:
        rec_kind = legal[0].kind if legal else ActionKind.CHECK
        rec_label = legal[0].describe() if legal else "Check"

    return {
        "equity_pct": round(equity * 100, 1),
        "chen_score": round(chen, 1),
        "preflop_bucket": pre_bucket,
        "spr": round(spr, 1),
        "effective_stack": round(E.effective_stack(state), 1),
        "recommended": rec_label,
        "recommended_type": rec_kind.name.lower(),
        "sizing_suggestions": _format_sizing(sizing),
        "legal_actions": _format_legal_actions(legal),
    }


@router.get("/bots")
async def list_bots():
    """List available bot types and weight files."""
    bots = [{"id": "heuristic", "label": "Heuristic (equity + SPR)", "type": "heuristic", "weights": ""}]

    # Find all Q-learning weight files
    for p in sorted(Path("data").glob("*.pkl")):
        name = p.stem
        if "qlearn" in name or "rl_agent" in name or "ql_v" in name:
            try:
                with open(p, "rb") as f:
                    import pickle
                    s = pickle.load(f)
                hands = s.get("hands_trained", 0)
                bots.append({
                    "id": f"ql:{p.name}",
                    "label": f"Q-Learn {name} ({hands} hands)",
                    "type": "ql",
                    "weights": str(p),
                })
            except:
                pass

    # Find PPO weight files
    for p in sorted(Path("data").glob("*ppo*.pkl")):
        name = p.stem
        try:
            with open(p, "rb") as f:
                import pickle
                s = pickle.load(f)
            hands = s.get("hands_trained", 0)
            updates = s.get("updates_done", 0)
            bots.append({
                "id": f"ppo:{p.name}",
                "label": f"PPO Neural Net {name} ({hands} hands, {updates} updates)",
                "type": "ppo",
                "weights": str(p),
            })
        except:
            pass

    return {"bots": bots}


@router.get("/sessions")
async def list_play_sessions():
    """List play sessions."""
    sessions = []
    for sid, s in _play_sessions.items():
        n = len(s["hands"])
        total = sum(h.get("winnings_bb", 0) for h in s["hands"])
        sessions.append({
            "id": sid,
            "bot_type": s.get("bot_type", "heuristic"),
            "hands": n,
            "total_bb": round(total, 1),
            "started": s.get("started", ""),
        })
    return {"sessions": sessions}

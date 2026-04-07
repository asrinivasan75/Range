"""Self-play training for PPO neural network agent.

Runs thousands of hands per minute locally — no Slumbot API needed.
Uses the BettingEngine for correct NLHE rules and plays against:
  1. Copies of itself (learns balanced play)
  2. Q-learning agents (learns to exploit their weaknesses)
  3. Random mix of opponents (generalizes)

This is the critical missing piece: the agent learns that
"bottom pair facing a huge overbet = fold" because its opponent
(itself) actually BETS with strong hands and BLUFFS with air.
"""

from __future__ import annotations
import time
import random
import pickle
import json
from pathlib import Path

import numpy as np
import torch

from packages.poker.card import Card
from packages.poker.deck import Deck
from packages.poker.evaluator import hand_strength_monte_carlo, evaluate_hand
from packages.poker.game_state import Street
from packages.poker.betting_engine import BettingEngine, BettingState, ActionKind
from packages.solver.neural_agent import (
    PPOAgent, extract_features, get_legal_mask, ppo_action_to_slumbot,
    N_ACTIONS, ACTIONS, Transition,
)
from packages.solver.rl_agent import QLearningAgent, extract_features as rl_extract_features

E = BettingEngine


def _board_cards(hand_data: dict, street: Street) -> list[Card]:
    cards = []
    if street.value >= 1:
        cards.extend(hand_data["flop"])
    if street.value >= 2:
        cards.append(hand_data["turn"])
    if street.value >= 3:
        cards.append(hand_data["river"])
    return cards


def _pick_action_ppo(
    agent: PPOAgent,
    cards: list[Card],
    board: list[Card],
    state: BettingState,
    player_idx: int,
    collect_transitions: bool = True,
) -> tuple[ActionKind, float, Transition | None]:
    """Use PPO agent to pick an action from BettingState."""
    legal = E.get_legal_actions(state)
    if not legal:
        return ActionKind.CHECK, 0, None

    pot = int(state.pot)
    to_call = int(state.to_call)
    invested = int(state.players[player_idx].total_invested)
    remaining = int(state.players[player_idx].stack)
    is_btn = (state.button_seat == player_idx)
    n_bets = sum(1 for r in state.action_log if r.street == state.street
                 and r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN))

    features = extract_features(
        cards, board, pot, to_call, invested, remaining,
        state.street.value, is_btn, n_bets,
    )
    legal_mask = get_legal_mask(to_call, remaining)

    action_idx, log_prob, value = agent.choose_action(features, legal_mask)
    transition = Transition(features, action_idx, log_prob, value, 0.0, legal_mask) if collect_transitions else None

    # Map to BettingEngine action
    action_name = ACTIONS[action_idx]
    legal_kinds = {a.kind for a in legal}

    if action_name == "fold" and ActionKind.FOLD in legal_kinds:
        return ActionKind.FOLD, 0, transition
    if action_name == "check" and ActionKind.CHECK in legal_kinds:
        return ActionKind.CHECK, 0, transition
    if action_name == "call":
        call = next((a for a in legal if a.kind == ActionKind.CALL), None)
        if call:
            return ActionKind.CALL, call.amount, transition
    if action_name == "all_in":
        ai = next((a for a in legal if a.kind == ActionKind.ALL_IN), None)
        if ai:
            return ActionKind.ALL_IN, 0, transition
        # Try as a raise to max
        raise_a = next((a for a in legal if a.kind == ActionKind.RAISE), None)
        if raise_a:
            return ActionKind.RAISE, raise_a.max_amount, transition

    # Bet/raise sizing
    frac = {"bet_small": 0.33, "bet_medium": 0.67, "bet_large": 1.0}.get(action_name, 0.5)

    if ActionKind.BET in legal_kinds:
        bet = next(a for a in legal if a.kind == ActionKind.BET)
        amount = max(bet.min_amount, min(state.pot * frac, bet.max_amount))
        return ActionKind.BET, round(amount, 1), transition
    if ActionKind.RAISE in legal_kinds:
        raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
        amount = max(raise_a.min_amount, min(state.highest_bet + state.pot * frac, raise_a.max_amount))
        return ActionKind.RAISE, round(amount, 1), transition

    # Fallback
    first = legal[0]
    return first.kind, first.amount, transition


def _pick_action_qlearn(
    agent: QLearningAgent,
    cards: list[Card],
    board: list[Card],
    state: BettingState,
    player_idx: int,
) -> tuple[ActionKind, float]:
    """Use Q-learning agent to pick an action."""
    legal = E.get_legal_actions(state)
    if not legal:
        return ActionKind.CHECK, 0

    pot = int(state.pot)
    to_call = int(state.to_call)
    invested = int(state.players[player_idx].total_invested)
    remaining = int(state.players[player_idx].stack)
    is_btn = (state.button_seat == player_idx)
    n_bets = sum(1 for r in state.action_log if r.street == state.street
                 and r.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN))

    features = rl_extract_features(
        cards, board, pot, to_call, invested, remaining,
        state.street.value, is_btn, n_bets,
    )
    from packages.solver.rl_agent import get_legal_mask as rl_mask
    mask = rl_mask(to_call, remaining)
    action_idx = agent.choose_action(features, mask)

    action_name = ["fold", "check", "call", "bet_small", "bet_medium", "bet_large"][action_idx]
    legal_kinds = {a.kind for a in legal}

    if action_name == "fold" and ActionKind.FOLD in legal_kinds:
        return ActionKind.FOLD, 0
    if action_name == "check" and ActionKind.CHECK in legal_kinds:
        return ActionKind.CHECK, 0
    if action_name == "call":
        call = next((a for a in legal if a.kind == ActionKind.CALL), None)
        if call:
            return ActionKind.CALL, call.amount

    frac = {"bet_small": 0.33, "bet_medium": 0.67, "bet_large": 1.0}.get(action_name, 0.5)
    if ActionKind.BET in legal_kinds:
        bet = next(a for a in legal if a.kind == ActionKind.BET)
        return ActionKind.BET, round(max(bet.min_amount, min(state.pot * frac, bet.max_amount)), 1)
    if ActionKind.RAISE in legal_kinds:
        raise_a = next(a for a in legal if a.kind == ActionKind.RAISE)
        return ActionKind.RAISE, round(max(raise_a.min_amount, min(state.highest_bet + state.pot * frac, raise_a.max_amount)), 1)

    return legal[0].kind, legal[0].amount


def play_hand_local(
    agent: PPOAgent,
    opponent,  # PPOAgent or QLearningAgent or "random"
    agent_seat: int = 0,
) -> tuple[float, list[Transition]]:
    """Play one hand locally. Returns (reward_bb, transitions).

    No API calls — pure local simulation using BettingEngine.
    Runs in <1ms per hand (vs ~500ms for Slumbot API).
    """
    deck = Deck()
    p0_cards = deck.deal(2)
    p1_cards = deck.deal(2)
    flop = deck.deal(3)
    turn = deck.deal_one()
    river = deck.deal_one()

    hand_data = {"flop": flop, "turn": turn, "river": river}
    cards = [p0_cards, p1_cards]

    state = E.create_hand(button_seat=0, stacks=(100.0, 100.0), sb=1.0, bb=2.0)
    transitions: list[Transition] = []

    for _ in range(50):  # safety limit
        if state.is_hand_complete:
            break

        if E.is_street_complete(state):
            state = E.advance_street(state)
            if state.is_hand_complete:
                break
            continue

        idx = state.current_player_idx
        board = _board_cards(hand_data, state.street)

        if idx == agent_seat:
            kind, amount, trans = _pick_action_ppo(agent, cards[idx], board, state, idx)
            if trans:
                transitions.append(trans)
        else:
            if isinstance(opponent, PPOAgent):
                kind, amount, _ = _pick_action_ppo(opponent, cards[idx], board, state, idx, collect_transitions=False)
            elif isinstance(opponent, QLearningAgent):
                kind, amount = _pick_action_qlearn(opponent, cards[idx], board, state, idx)
            else:
                # Random opponent
                legal = E.get_legal_actions(state)
                action = random.choice(legal)
                kind, amount = action.kind, action.amount

        try:
            state = E.apply_action(state, kind, amount)
        except (ValueError, Exception):
            break

    # Determine reward
    if not state.is_hand_complete:
        return 0, transitions

    non_folded = state.non_folded_indices
    if len(non_folded) == 1:
        winner = non_folded[0]
    else:
        board = _board_cards(hand_data, state.street)
        if len(cards[0]) + len(board) >= 5:
            h0 = evaluate_hand(list(cards[0]), board)
            h1 = evaluate_hand(list(cards[1]), board)
            winner = 0 if h0 > h1 else (1 if h1 > h0 else -1)
        else:
            winner = -1  # tie

    if winner == agent_seat:
        reward = state.players[1 - agent_seat].total_invested
    elif winner == 1 - agent_seat:
        reward = -state.players[agent_seat].total_invested
    else:
        reward = 0

    reward_bb = reward / 2.0  # normalize to bb
    return reward_bb, transitions


def train_self_play(
    n_hands: int = 50000,
    opponent_mix: dict | None = None,
    save_path: str = "data/ppo_selfplay.pkl",
    save_interval: int = 5000,
    verbose: bool = True,
) -> PPOAgent:
    """Train PPO agent via self-play + opponent pool.

    Args:
        n_hands: total hands to play
        opponent_mix: dict of {"self": weight, "qlearn_path": weight, "random": weight}
        save_path: where to save trained weights
        save_interval: save every N hands
    """
    agent = PPOAgent()

    # Try loading existing weights
    if Path(save_path).exists():
        agent.load(save_path)
        print(f"Resumed from {save_path} ({agent.hands_trained} hands)")
    else:
        print("Starting fresh PPO agent")

    # Build opponent pool
    opponents = []
    if opponent_mix is None:
        opponent_mix = {"self": 0.5, "random": 0.2}
        # Add any Q-learning agents we have
        for p in Path("data").glob("pbt_qlearn_*.pkl"):
            opponent_mix[str(p)] = 0.1

    for key, weight in opponent_mix.items():
        if key == "self":
            opponents.append(("self", weight, None))
        elif key == "random":
            opponents.append(("random", weight, "random"))
        else:
            # Load Q-learning agent
            ql = QLearningAgent()
            if ql.load(key):
                opponents.append((Path(key).stem, weight, ql))

    # Normalize weights
    total_w = sum(w for _, w, _ in opponents)
    opponents = [(n, w / total_w, o) for n, w, o in opponents]

    print(f"\nSelf-play training: {n_hands} hands")
    print(f"Opponents:")
    for name, weight, _ in opponents:
        print(f"  {name}: {weight:.0%}")
    print()

    t0 = time.time()
    total_reward = 0
    reward_history = []
    chunk_size = 1000

    for i in range(n_hands):
        # Pick opponent
        r = random.random()
        cumulative = 0
        opp = "random"
        for name, weight, obj in opponents:
            cumulative += weight
            if r < cumulative:
                if name == "self":
                    opp = agent  # play against itself
                else:
                    opp = obj
                break

        # Alternate seats
        seat = i % 2

        reward, transitions = play_hand_local(agent, opp, agent_seat=seat)
        total_reward += reward

        if transitions:
            agent.end_hand(reward, transitions)

        # Logging
        if (i + 1) % chunk_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            avg = total_reward / (i + 1)
            recent = sum(reward_history[-chunk_size:]) / min(len(reward_history[-chunk_size:]), chunk_size) if reward_history else 0
            stats = agent.get_stats()

            if verbose:
                print(f"  {i+1:>6}/{n_hands} | {rate:.0f} hands/sec | avg: {avg:+.2f}bb | "
                      f"recent {chunk_size}: {recent:+.2f}bb | updates: {stats['updates_done']} | "
                      f"lr: {stats['lr']:.6f}")

        reward_history.append(reward)

        # Save periodically
        if (i + 1) % save_interval == 0:
            agent.save(save_path)
            if verbose:
                print(f"  Saved to {save_path}")

    elapsed = time.time() - t0
    agent.save(save_path)

    print(f"\n{'='*60}")
    print(f"SELF-PLAY TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Hands:     {n_hands}")
    print(f"Time:      {elapsed:.0f}s ({n_hands/elapsed:.0f} hands/sec)")
    print(f"Avg reward: {total_reward/n_hands:+.3f} bb/hand")
    print(f"Updates:   {agent.get_stats()['updates_done']}")
    print(f"Saved:     {save_path}")

    # Show learning curve
    if len(reward_history) >= 2000:
        first_1k = sum(reward_history[:1000]) / 1000
        last_1k = sum(reward_history[-1000:]) / 1000
        print(f"\nFirst 1K avg:  {first_1k:+.3f} bb/hand")
        print(f"Last 1K avg:   {last_1k:+.3f} bb/hand")
        print(f"Improvement:   {last_1k - first_1k:+.3f} bb/hand")

    return agent


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    train_self_play(n_hands=n)

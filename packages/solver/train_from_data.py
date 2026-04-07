"""Train PPO agent from real poker hand history data (PHH format).

Parses heads-up NLHE hands from the PHH dataset and uses them to train
the PPO neural network via imitation learning + reinforcement learning.

Approach:
1. Parse hand histories into (state, action, reward) tuples
2. Use the winning player's actions as positive examples
3. Use the losing player's actions as negative examples
4. Train PPO with shaped rewards based on actual outcomes

This avoids overfitting to one opponent (Slumbot) by learning from
thousands of different human players across multiple poker sites.
"""

from __future__ import annotations
import re
import time
import random
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from packages.poker.card import Card
from packages.poker.evaluator import hand_strength_monte_carlo
from packages.solver.neural_agent import (
    PPOAgent, extract_features, get_legal_mask, N_ACTIONS, ACTIONS, Transition,
)


@dataclass
class ParsedHand:
    """A parsed hand from PHH format."""
    actions: list[dict]  # [{type, amount, player}, ...]
    board: list[str]
    hole_cards: list[list[str]]  # per player
    blinds: list[float]
    winnings: list[float]
    n_players: int


def parse_phh_file(filepath: str, max_hands: int = 50000) -> list[ParsedHand]:
    """Parse a .phhs file containing multiple hands."""
    hands = []
    current_hand = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                # New hand marker
                if current_hand and "actions" in current_hand:
                    parsed = _parse_single_hand(current_hand)
                    if parsed:
                        hands.append(parsed)
                        if len(hands) >= max_hands:
                            break
                current_hand = {}
                continue

            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                current_hand[key] = value

    # Don't forget the last hand
    if current_hand and "actions" in current_hand:
        parsed = _parse_single_hand(current_hand)
        if parsed:
            hands.append(parsed)

    return hands


def _parse_single_hand(hand_dict: dict) -> ParsedHand | None:
    """Parse a single hand from its key-value dict."""
    try:
        # Only want heads-up no-limit
        if hand_dict.get("variant") != "NT":
            return None

        # Parse actions
        actions_str = hand_dict.get("actions", "")
        if actions_str.startswith("["):
            actions_str = actions_str[1:-1]  # strip brackets
        action_parts = [a.strip().strip("'\"") for a in actions_str.split(",")]

        actions = []
        board = []
        hole_cards = [[], []]
        street = 0

        for part in action_parts:
            part = part.strip()
            if not part:
                continue

            tokens = part.split()
            if len(tokens) < 2:
                continue

            if tokens[0] == "d":
                # Deal action
                if tokens[1] == "dh":
                    # Deal hole cards
                    player_idx = int(tokens[2][1]) - 1
                    cards_str = tokens[3] if len(tokens) > 3 else "????"
                    if cards_str != "????":
                        hole_cards[player_idx] = [cards_str[i:i+2] for i in range(0, len(cards_str), 2)]
                elif tokens[1] == "db":
                    # Deal board
                    cards_str = tokens[2] if len(tokens) > 2 else ""
                    board_cards = [cards_str[i:i+2] for i in range(0, len(cards_str), 2)]
                    board.extend(board_cards)
                    street += 1
            else:
                # Player action
                player_str = tokens[0]
                player_idx = int(player_str[1]) - 1
                action_type = tokens[1]

                action = {"player": player_idx, "street": street}

                if action_type == "f":
                    action["type"] = "fold"
                elif action_type == "cc":
                    action["type"] = "call"  # check or call
                elif action_type.startswith("cbr"):
                    amount = float(tokens[2]) if len(tokens) > 2 else 0
                    action["type"] = "raise"
                    action["amount"] = amount
                elif action_type == "sm":
                    continue  # showdown/muck
                else:
                    continue

                actions.append(action)

        if not actions:
            return None

        # Parse blinds
        blinds_str = hand_dict.get("blinds_or_straddles", "[1, 2]")
        blinds = [float(x.strip()) for x in blinds_str.strip("[]").split(",")]

        # Parse winnings (if available)
        winnings = [0.0, 0.0]

        return ParsedHand(
            actions=actions,
            board=board,
            hole_cards=hole_cards,
            blinds=blinds,
            winnings=winnings,
            n_players=2,
        )
    except Exception:
        return None


def _hand_to_transitions(
    hand: ParsedHand,
    perspective: int,  # which player's perspective (0 or 1)
    reward: float,     # positive if this player won
) -> list[Transition]:
    """Convert a parsed hand into PPO transitions from one player's perspective."""
    transitions = []

    try:
        hole = [Card.from_str(c) for c in hand.hole_cards[perspective]] if hand.hole_cards[perspective] else None
        if not hole or len(hole) != 2:
            return []

        board_cards = []
        pot = sum(hand.blinds)
        invested = [hand.blinds[0], hand.blinds[1]]
        street = 0

        for action in hand.actions:
            if action["player"] != perspective:
                # Opponent action — update state but don't create transition
                if action["type"] == "raise":
                    invested[action["player"]] = action.get("amount", invested[action["player"]])
                    pot = sum(invested)
                elif action["type"] == "call":
                    invested[action["player"]] = max(invested)
                    pot = sum(invested)
                continue

            # Our action — create a transition
            if action.get("street", 0) > street:
                # New street — update board
                street = action["street"]

            board = [Card.from_str(c) for c in hand.board[:min(street * 1 + 2, len(hand.board))]]
            # Map street to board cards: 0=preflop(0), 1=flop(3), 2=turn(4), 3=river(5)
            board_idx = {0: 0, 1: 3, 2: 4, 3: 5}.get(street, 0)
            board = [Card.from_str(c) for c in hand.board[:board_idx]] if hand.board else []

            to_call = max(0, max(invested) - invested[perspective])
            remaining = 200 - invested[perspective]  # assume 200bb effective

            features = extract_features(
                hole, board, int(pot), int(to_call), int(invested[perspective]),
                int(remaining), street, perspective == 0, 0,
            )

            legal_mask = get_legal_mask(int(to_call), int(remaining))

            # Map the actual action to our action index
            action_idx = _map_action_to_idx(action, pot)
            if action_idx is None:
                continue

            transitions.append(Transition(
                features=features,
                action_idx=action_idx,
                log_prob=0.0,  # placeholder — will be computed by PPO
                value=0.0,
                reward=0.0,  # set later
                legal_mask=legal_mask,
            ))

            # Update state
            if action["type"] == "raise":
                invested[perspective] = action.get("amount", invested[perspective])
                pot = sum(invested)
            elif action["type"] == "call":
                invested[perspective] = max(invested)
                pot = sum(invested)

    except Exception:
        return []

    # Assign discounted rewards
    n = len(transitions)
    for i, t in enumerate(transitions):
        discount = 0.99 ** (n - 1 - i)
        t.reward = reward * discount / 100.0

    return transitions


def _map_action_to_idx(action: dict, pot: float) -> int | None:
    """Map a real action to our action space index."""
    if action["type"] == "fold":
        return 0  # fold
    if action["type"] == "call":
        return 2  # call (also covers check)

    if action["type"] == "raise":
        amount = action.get("amount", 0)
        if pot <= 0:
            return 3  # default to small bet
        ratio = amount / pot if pot > 0 else 0.5

        if ratio < 0.5:
            return 3  # bet_small
        elif ratio < 0.85:
            return 4  # bet_medium
        elif ratio < 1.5:
            return 5  # bet_large
        else:
            return 6  # all_in

    return None


def train_from_hands(
    data_dir: str = "data/phh-dataset/data/handhq",
    n_hands: int = 100000,
    save_path: str = "data/ppo_realdata.pkl",
    save_interval: int = 10000,
    verbose: bool = True,
) -> PPOAgent:
    """Train PPO agent from real hand history data."""

    agent = PPOAgent()
    if Path(save_path).exists():
        agent.load(save_path)
        print(f"Resumed from {save_path} ({agent.hands_trained} hands)")
    else:
        print("Starting fresh PPO agent for real data training")

    # Find all heads-up hand files
    hu_files = list(Path(data_dir).glob("*/2/*.phhs"))
    random.shuffle(hu_files)
    print(f"Found {len(hu_files)} heads-up hand files")

    t0 = time.time()
    total_hands = 0
    total_transitions = 0
    hands_won = 0

    for file_idx, filepath in enumerate(hu_files):
        if total_hands >= n_hands:
            break

        try:
            hands = parse_phh_file(str(filepath), max_hands=n_hands - total_hands)
        except Exception:
            continue

        for hand in hands:
            if total_hands >= n_hands:
                break

            # Determine winner — player who didn't fold, or by cards
            winner = -1
            for action in hand.actions:
                if action["type"] == "fold":
                    winner = 1 - action["player"]
                    break

            if winner == -1:
                # Went to showdown — skip if we don't have both hole cards
                if not hand.hole_cards[0] or not hand.hole_cards[1]:
                    continue
                winner = 0  # default

            # Train from the winner's perspective (positive reward)
            # and loser's perspective (negative reward)
            for player in range(2):
                if not hand.hole_cards[player]:
                    continue

                reward = 1.0 if player == winner else -1.0
                transitions = _hand_to_transitions(hand, player, reward)

                if transitions:
                    agent.end_hand(reward, transitions)
                    total_transitions += len(transitions)

            total_hands += 1
            if winner >= 0:
                hands_won += 1

            if verbose and total_hands % 5000 == 0:
                elapsed = time.time() - t0
                rate = total_hands / elapsed
                stats = agent.get_stats()
                print(f"  {total_hands:>7}/{n_hands} | {rate:.0f} hands/sec | "
                      f"transitions: {total_transitions} | updates: {stats['updates_done']} | "
                      f"file {file_idx+1}/{len(hu_files)}")

            if total_hands % save_interval == 0:
                agent.save(save_path)
                if verbose:
                    print(f"  Saved to {save_path}")

    elapsed = time.time() - t0
    agent.save(save_path)

    print(f"\n{'='*60}")
    print(f"REAL DATA TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Hands:        {total_hands}")
    print(f"Transitions:  {total_transitions}")
    print(f"Time:         {elapsed:.0f}s ({total_hands/max(elapsed,1):.0f} hands/sec)")
    print(f"PPO updates:  {agent.get_stats()['updates_done']}")
    print(f"Saved:        {save_path}")

    return agent


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    train_from_hands(n_hands=n)

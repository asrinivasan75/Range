"""Opponent Range Estimator — estimates what hands the opponent likely has
based on their actions, without needing to see their actual cards.

Two approaches combined:
1. GTO-based priors: theoretical hand ranges for common action sequences
   (e.g., "3-bet shove preflop" = AA, KK, QQ, AKs roughly)
2. Bayesian updating from showdown data: when we DO see cards at showdown,
   update our beliefs about what actions correlate with what hands.

The key insight: we don't need to see opponent cards in the dataset.
We can use the ACTIONS THEY TOOK + the BOARD to infer likely ranges,
then compute equity against those ranges instead of random.
"""

from __future__ import annotations
import math
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from packages.poker.card import Card, Rank
from packages.poker.evaluator import hand_strength_monte_carlo
from packages.poker.deck import FULL_DECK


# 169 canonical starting hands
def _build_canonical_hands() -> list[str]:
    """Build list of 169 canonical starting hands: AA, AKs, AKo, ..., 32o."""
    hands = []
    ranks = list(reversed(list(Rank)))  # A, K, Q, ..., 2
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                hands.append(f"{r1}{r2}")    # pair
            elif i < j:
                hands.append(f"{r1}{r2}s")   # suited (above diagonal)
            else:
                hands.append(f"{r2}{r1}o")   # offsuit (below diagonal)
    return hands


CANONICAL_HANDS = _build_canonical_hands()
HAND_TO_IDX = {h: i for i, h in enumerate(CANONICAL_HANDS)}
N_HANDS = len(CANONICAL_HANDS)  # 169


# ── GTO-based prior ranges ──────────────────────────────────

def _hand_percentile(hand: str) -> float:
    """Rough percentile ranking of a starting hand (0=worst, 1=best)."""
    idx = HAND_TO_IDX.get(hand, 84)
    # The canonical list is roughly ordered by strength
    # but not perfectly — use a simplified ranking
    rank_map = {
        "AA": 1.0, "KK": 0.99, "QQ": 0.98, "JJ": 0.96, "AKs": 0.95,
        "TT": 0.93, "AKo": 0.92, "AQs": 0.91, "99": 0.89, "AJs": 0.87,
        "AQo": 0.86, "KQs": 0.85, "88": 0.83, "ATs": 0.82, "KJs": 0.81,
        "AJo": 0.80, "KQo": 0.79, "77": 0.77, "A9s": 0.76, "KTs": 0.75,
        "ATo": 0.74, "QJs": 0.73, "66": 0.71, "KJo": 0.70, "A8s": 0.69,
        "QTs": 0.68, "K9s": 0.67, "55": 0.65, "A7s": 0.64, "JTs": 0.63,
    }
    if hand in rank_map:
        return rank_map[hand]
    # Estimate based on high card
    high = hand[0]
    high_val = {"A": 0.6, "K": 0.5, "Q": 0.4, "J": 0.35, "T": 0.3,
                "9": 0.25, "8": 0.2, "7": 0.15, "6": 0.1, "5": 0.08,
                "4": 0.06, "3": 0.04, "2": 0.02}.get(high, 0.1)
    suited_bonus = 0.05 if hand.endswith("s") else 0
    pair_bonus = 0.15 if len(hand) == 2 else 0
    return min(high_val + suited_bonus + pair_bonus, 0.95)


def gto_range_for_action(
    action_type: str,
    street: int,
    bet_size_ratio: float = 0.5,
    position: str = "ip",  # "ip" or "oop"
) -> np.ndarray:
    """Return a probability distribution over 169 hands based on GTO theory.

    This is a simplified model — real GTO ranges are board-dependent.
    """
    probs = np.zeros(N_HANDS, dtype=np.float64)

    for i, hand in enumerate(CANONICAL_HANDS):
        pct = _hand_percentile(hand)

        if street == 0:  # preflop
            if action_type == "open_raise":
                # ~40-60% of hands depending on position
                threshold = 0.45 if position == "ip" else 0.55
                probs[i] = 1.0 if pct > threshold else 0.1

            elif action_type == "3bet":
                # ~10-15% of hands: premiums + some bluffs
                probs[i] = 1.0 if pct > 0.85 else (0.3 if pct > 0.6 else 0.05)

            elif action_type == "4bet":
                # ~5-8%: strong premiums + rare bluffs
                probs[i] = 1.0 if pct > 0.92 else (0.15 if pct > 0.7 else 0.02)

            elif action_type == "5bet_shove":
                # ~3-4%: AA, KK, QQ, AKs mainly
                probs[i] = 1.0 if pct > 0.95 else (0.1 if pct > 0.85 else 0.01)

            elif action_type == "call":
                # Wide calling range
                probs[i] = 1.0 if pct > 0.3 else 0.2

            elif action_type == "limp":
                # Weaker hands that don't raise
                probs[i] = 0.5 if pct < 0.6 else 0.1

        else:  # postflop
            if action_type == "bet_small":
                # Wide range — value + draws + bluffs
                probs[i] = 0.6 if pct > 0.3 else 0.3

            elif action_type == "bet_large":
                # Polarized — strong hands + bluffs
                probs[i] = 1.0 if pct > 0.8 else (0.4 if pct < 0.2 else 0.15)

            elif action_type == "bet_overbet":
                # Very polarized — nuts or air
                probs[i] = 1.0 if pct > 0.9 else (0.5 if pct < 0.1 else 0.05)

            elif action_type == "check_raise":
                # Strong hands + semi-bluffs
                probs[i] = 1.0 if pct > 0.75 else (0.3 if pct < 0.25 else 0.1)

            elif action_type == "call":
                # Medium strength — not raising, not folding
                probs[i] = 1.0 if 0.3 < pct < 0.8 else 0.2

            elif action_type == "check":
                # Weak or trapping
                probs[i] = 0.6 if pct < 0.5 else 0.3

    # Normalize
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = np.ones(N_HANDS) / N_HANDS

    return probs


# ── Action sequence to range ─────────────────────────────────

def classify_action_sequence(
    actions: list[dict],
    street: int,
) -> str:
    """Classify an action sequence into a range category."""
    if not actions:
        return "unknown"

    # Count raises on current street
    street_actions = [a for a in actions if a.get("street", 0) == street]
    n_raises = sum(1 for a in street_actions if a.get("type") in ("raise", "bet"))
    last_action = street_actions[-1] if street_actions else actions[-1]
    last_type = last_action.get("type", "unknown")

    if street == 0:  # preflop
        if n_raises >= 4:
            return "5bet_shove"
        elif n_raises >= 3:
            return "4bet"
        elif n_raises >= 2:
            return "3bet"
        elif n_raises >= 1:
            return "open_raise"
        elif last_type == "call":
            return "call"
        else:
            return "limp"
    else:
        # Postflop
        bet_size = last_action.get("amount", 0)
        pot = last_action.get("pot", 1)
        ratio = bet_size / max(pot, 1) if bet_size else 0

        if last_type in ("raise",):
            return "check_raise" if any(a.get("type") == "check" for a in street_actions[:-1]) else "bet_large"
        elif last_type == "bet":
            if ratio > 1.2:
                return "bet_overbet"
            elif ratio > 0.6:
                return "bet_large"
            else:
                return "bet_small"
        elif last_type == "call":
            return "call"
        else:
            return "check"


# ── Equity against estimated range ───────────────────────────

def equity_vs_range(
    hole_cards: list[Card],
    board: list[Card],
    opponent_range: np.ndarray,
    n_samples: int = 200,
) -> float:
    """Compute equity against an opponent's estimated range distribution.

    Instead of equity vs random, weights opponent hands by range probability.
    """
    if len(hole_cards) != 2:
        return 0.5

    dead = set(c.id for c in hole_cards) | set(c.id for c in board)
    remaining = [c for c in FULL_DECK if c.id not in dead]

    # Sample opponent hands according to range distribution
    wins = 0.0
    total = 0.0

    # Build concrete hand combos from canonical hands with probabilities
    combos = _range_to_combos(opponent_range, dead)
    if not combos:
        # Fallback to random
        return hand_strength_monte_carlo(hole_cards, board, n_simulations=n_samples)

    import random as rng
    from packages.poker.evaluator import evaluate_hand

    cards_needed = 5 - len(board)

    for _ in range(n_samples):
        # Sample opponent hand from range
        weights = [w for _, w in combos]
        total_w = sum(weights)
        if total_w <= 0:
            continue
        r = rng.random() * total_w
        cumulative = 0
        opp_cards = combos[0][0]
        for cards, w in combos:
            cumulative += w
            if r < cumulative:
                opp_cards = cards
                break

        # Check no overlap with board
        opp_dead = set(c.id for c in opp_cards)
        if opp_dead & dead:
            continue

        # Deal remaining board if needed
        avail = [c for c in remaining if c.id not in opp_dead]
        if cards_needed > 0 and len(avail) < cards_needed:
            continue
        rng.shuffle(avail)
        full_board = list(board) + avail[:cards_needed]

        if len(hole_cards) + len(full_board) < 5:
            continue

        my_hand = evaluate_hand(list(hole_cards), full_board)
        opp_hand = evaluate_hand(list(opp_cards), full_board)

        if my_hand > opp_hand:
            wins += 1.0
        elif my_hand == opp_hand:
            wins += 0.5
        total += 1.0

    return wins / max(total, 1)


def _range_to_combos(
    range_dist: np.ndarray,
    dead_ids: set,
) -> list[tuple[list[Card], float]]:
    """Convert 169-hand range distribution to concrete card combos with weights."""
    from packages.poker.card import Suit

    combos = []
    suits = list(Suit)

    for i, hand_str in enumerate(CANONICAL_HANDS):
        prob = range_dist[i]
        if prob < 0.001:
            continue

        if len(hand_str) == 2:
            # Pair: e.g., "AA" -> 6 combos
            rank = Rank.from_char(hand_str[0])
            for s1_idx in range(4):
                for s2_idx in range(s1_idx + 1, 4):
                    c1 = Card(rank, suits[s1_idx])
                    c2 = Card(rank, suits[s2_idx])
                    if c1.id not in dead_ids and c2.id not in dead_ids:
                        combos.append(([c1, c2], prob))

        elif hand_str.endswith("s"):
            # Suited: 4 combos
            r1 = Rank.from_char(hand_str[0])
            r2 = Rank.from_char(hand_str[1])
            for s in suits:
                c1 = Card(r1, s)
                c2 = Card(r2, s)
                if c1.id not in dead_ids and c2.id not in dead_ids:
                    combos.append(([c1, c2], prob))

        elif hand_str.endswith("o"):
            # Offsuit: 12 combos
            r1 = Rank.from_char(hand_str[0])
            r2 = Rank.from_char(hand_str[1])
            for s1 in suits:
                for s2 in suits:
                    if s1 != s2:
                        c1 = Card(r1, s1)
                        c2 = Card(r2, s2)
                        if c1.id not in dead_ids and c2.id not in dead_ids:
                            combos.append(([c1, c2], prob))

    return combos


# ── High-level API ───────────────────────────────────────────

class RangeEstimator:
    """Estimates opponent range and computes equity against it."""

    def __init__(self):
        self.showdown_data: dict[str, list[int]] = defaultdict(list)
        # action_category -> list of hand indices seen at showdown

    def estimate_range(
        self,
        opponent_actions: list[dict],
        street: int,
        position: str = "ip",
    ) -> np.ndarray:
        """Estimate opponent's hand range based on their actions."""
        category = classify_action_sequence(opponent_actions, street)
        return gto_range_for_action(category, street, position=position)

    def compute_equity_vs_opponent(
        self,
        hole_cards: list[Card],
        board: list[Card],
        opponent_actions: list[dict],
        street: int,
        position: str = "ip",
        n_samples: int = 200,
    ) -> float:
        """Compute equity against opponent's estimated range."""
        opp_range = self.estimate_range(opponent_actions, street, position)
        return equity_vs_range(hole_cards, board, opp_range, n_samples)

    def record_showdown(self, action_category: str, hand_idx: int):
        """Record what hand an opponent showed at showdown for learning."""
        self.showdown_data[action_category].append(hand_idx)

    def save(self, path: str = "data/range_estimator.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(self.showdown_data), f)

    def load(self, path: str = "data/range_estimator.pkl") -> bool:
        try:
            with open(path, "rb") as f:
                self.showdown_data = defaultdict(list, pickle.load(f))
            return True
        except:
            return False

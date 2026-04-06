"""Card and action abstraction for reducing the poker game tree.

Abstractions map the enormous full-game information space to a tractable
number of buckets, enabling CFR to train on simplified Hold'em in minutes.
"""

from __future__ import annotations
import numpy as np
from typing import NamedTuple

from packages.poker.card import Card, Rank, Suit
from packages.poker.deck import FULL_DECK
from packages.poker.evaluator import (
    chen_formula,
    preflop_hand_bucket,
    evaluate_hand,
    hand_strength_monte_carlo,
)


class AbstractionConfig(NamedTuple):
    """Configuration for game abstraction."""
    preflop_buckets: int = 8
    flop_buckets: int = 8
    bet_sizes: list[float] = [0.5, 1.0]  # Fraction of pot
    max_raises_per_street: int = 2


# ── Preflop abstraction ─────────────────────────────────────

# Precomputed canonical hand groups (169 distinct starting hands)
# mapped to bucket indices

def compute_preflop_buckets(n_buckets: int = 8) -> dict[str, int]:
    """Compute the preflop bucket for every canonical starting hand.

    Returns a dict mapping canonical hand string (e.g., 'AKs', 'TTo', '72o')
    to bucket index.
    """
    hand_scores: list[tuple[str, float]] = []

    for r1 in Rank:
        for r2 in Rank:
            if r1 > r2:
                # Off-suit
                c1 = Card(r1, Suit.SPADES)
                c2 = Card(r2, Suit.HEARTS)
                label = f"{r1}{r2}o"
                score = chen_formula(c1, c2)
                hand_scores.append((label, score))
                # Suited
                c2s = Card(r2, Suit.SPADES)
                label_s = f"{r1}{r2}s"
                score_s = chen_formula(c1, c2s)
                hand_scores.append((label_s, score_s))
            elif r1 == r2:
                # Pair
                c1 = Card(r1, Suit.SPADES)
                c2 = Card(r2, Suit.HEARTS)
                label = f"{r1}{r2}"
                score = chen_formula(c1, c2)
                hand_scores.append((label, score))

    # Sort by score and assign buckets
    hand_scores.sort(key=lambda x: x[1])
    bucket_size = len(hand_scores) / n_buckets

    result = {}
    for i, (label, _) in enumerate(hand_scores):
        bucket = min(int(i / bucket_size), n_buckets - 1)
        result[label] = bucket

    return result


# ── Flop abstraction ────────────────────────────────────────

def compute_flop_bucket(
    hole_cards: list[Card],
    board: list[Card],
    n_buckets: int = 8,
    n_simulations: int = 200,
) -> int:
    """Bucket a hand+board combo by estimated equity.

    Uses Monte Carlo hand strength evaluation, then maps to [0, n_buckets-1].
    """
    equity = hand_strength_monte_carlo(hole_cards, board, n_simulations=n_simulations)
    bucket = int(equity * (n_buckets - 0.01))
    return min(bucket, n_buckets - 1)


# ── Precomputed lookup tables ───────────────────────────────

_PREFLOP_BUCKET_CACHE: dict[int, dict[str, int]] = {}


def get_preflop_buckets(n_buckets: int = 8) -> dict[str, int]:
    """Get or compute preflop bucket lookup table."""
    if n_buckets not in _PREFLOP_BUCKET_CACHE:
        _PREFLOP_BUCKET_CACHE[n_buckets] = compute_preflop_buckets(n_buckets)
    return _PREFLOP_BUCKET_CACHE[n_buckets]


def canonical_hand_key(c1: Card, c2: Card) -> str:
    """Convert two hole cards to a canonical hand string."""
    hi = max(c1.rank, c2.rank)
    lo = min(c1.rank, c2.rank)
    if hi == lo:
        return f"{Rank(hi)}{Rank(lo)}"
    suffix = "s" if c1.suit == c2.suit else "o"
    return f"{Rank(hi)}{Rank(lo)}{suffix}"


def get_hand_bucket_preflop(c1: Card, c2: Card, n_buckets: int = 8) -> int:
    """Get preflop bucket for a specific hand."""
    key = canonical_hand_key(c1, c2)
    buckets = get_preflop_buckets(n_buckets)
    return buckets.get(key, 0)

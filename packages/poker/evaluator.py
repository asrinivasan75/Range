"""Hand evaluation — 5/7-card hand ranking and Monte Carlo equity."""

from __future__ import annotations
from itertools import combinations
from collections import Counter
import random

from packages.poker.card import Card, Rank, Suit
from packages.poker.hand import HandCategory
from packages.poker.deck import FULL_DECK


def evaluate_five(cards: list[Card] | tuple[Card, ...]) -> tuple[HandCategory, tuple[int, ...]]:
    """Evaluate a 5-card hand. Returns (category, kickers) for comparison.

    Kickers are ordered so that comparing tuples gives correct hand ordering.
    """
    assert len(cards) == 5
    ranks = sorted((c.rank.value for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)

    is_flush = len(set(suits)) == 1

    # Check straight (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:  # Wheel
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        if straight_high == 14 and min(ranks) == 10:
            return (HandCategory.ROYAL_FLUSH, (14,))
        return (HandCategory.STRAIGHT_FLUSH, (straight_high,))

    freq = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if freq[0][1] == 4:
        quad_rank = freq[0][0]
        kicker = freq[1][0]
        return (HandCategory.FOUR_OF_A_KIND, (quad_rank, kicker))

    if freq[0][1] == 3 and freq[1][1] == 2:
        return (HandCategory.FULL_HOUSE, (freq[0][0], freq[1][0]))

    if is_flush:
        return (HandCategory.FLUSH, tuple(ranks))

    if is_straight:
        return (HandCategory.STRAIGHT, (straight_high,))

    if freq[0][1] == 3:
        trip_rank = freq[0][0]
        kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
        return (HandCategory.THREE_OF_A_KIND, (trip_rank, *kickers))

    if freq[0][1] == 2 and freq[1][1] == 2:
        high_pair = max(freq[0][0], freq[1][0])
        low_pair = min(freq[0][0], freq[1][0])
        kicker = [r for r in ranks if r != high_pair and r != low_pair][0]
        return (HandCategory.TWO_PAIR, (high_pair, low_pair, kicker))

    if freq[0][1] == 2:
        pair_rank = freq[0][0]
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
        return (HandCategory.ONE_PAIR, (pair_rank, *kickers))

    return (HandCategory.HIGH_CARD, tuple(ranks))


def evaluate_hand(hole_cards: list[Card], board: list[Card]) -> tuple[HandCategory, tuple[int, ...]]:
    """Evaluate the best 5-card hand from hole cards + board (5-7 total cards)."""
    all_cards = list(hole_cards) + list(board)
    if len(all_cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(all_cards)}")

    best = None
    for combo in combinations(all_cards, 5):
        result = evaluate_five(list(combo))
        if best is None or result > best:
            best = result
    return best  # type: ignore


def hand_strength_monte_carlo(
    hole_cards: list[Card],
    board: list[Card],
    n_simulations: int = 1000,
    seed: int | None = None,
) -> float:
    """Estimate hand equity against a random opponent via Monte Carlo.

    Returns win probability [0, 1] (ties count as 0.5).
    """
    rng = random.Random(seed)
    dead = set(c.id for c in hole_cards) | set(c.id for c in board)
    remaining = [c for c in FULL_DECK if c.id not in dead]

    cards_needed_board = 5 - len(board)
    wins = 0.0

    for _ in range(n_simulations):
        rng.shuffle(remaining)
        idx = 0
        # Deal remaining board cards
        sim_board = list(board) + remaining[idx : idx + cards_needed_board]
        idx += cards_needed_board
        # Deal opponent hole cards
        opp_cards = remaining[idx : idx + 2]
        idx += 2

        my_hand = evaluate_hand(list(hole_cards), sim_board)
        opp_hand = evaluate_hand(list(opp_cards), sim_board)

        if my_hand > opp_hand:
            wins += 1.0
        elif my_hand == opp_hand:
            wins += 0.5

    return wins / n_simulations


# -- Preflop hand strength table (Chen formula approximation) --

def chen_formula(card1: Card, card2: Card) -> float:
    """Compute the Chen formula score for a starting hand."""
    def rank_score(rank: Rank) -> float:
        if rank == Rank.ACE:
            return 10.0
        elif rank == Rank.KING:
            return 8.0
        elif rank == Rank.QUEEN:
            return 7.0
        elif rank == Rank.JACK:
            return 6.0
        else:
            return rank.value / 2.0

    hi = max(card1.rank, card2.rank)
    lo = min(card1.rank, card2.rank)
    score = rank_score(Rank(hi))

    # Pair bonus
    if hi == lo:
        score = max(score * 2, 5.0)

    # Suited bonus
    if card1.suit == card2.suit:
        score += 2.0

    # Gap penalty
    gap = hi - lo - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0

    # Straight bonus for connected/near-connected low cards
    if gap <= 1 and hi < Rank.QUEEN.value:
        score += 1.0

    return max(score, 0.0)


def preflop_hand_bucket(card1: Card, card2: Card, n_buckets: int = 8) -> int:
    """Map a starting hand to a bucket [0, n_buckets-1] using Chen formula.

    Bucket 0 = weakest, bucket n_buckets-1 = strongest.
    """
    score = chen_formula(card1, card2)
    # Chen scores range roughly from 0 to 20
    # Normalize to [0, 1] and bucket
    normalized = min(score / 20.0, 1.0)
    bucket = int(normalized * (n_buckets - 0.01))
    return min(bucket, n_buckets - 1)

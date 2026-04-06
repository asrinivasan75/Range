"""Deck management — shuffle, deal, remove dead cards."""

from __future__ import annotations
import random
from typing import Sequence

from packages.poker.card import Card, Rank, Suit

# Full 52-card deck as a precomputed list
FULL_DECK: list[Card] = [Card(r, s) for s in Suit for r in Rank]


class Deck:
    """A mutable deck that supports dealing and dead-card removal."""

    __slots__ = ("_cards", "_rng")

    def __init__(
        self,
        exclude: Sequence[Card] | None = None,
        seed: int | None = None,
    ) -> None:
        excluded_ids = {c.id for c in exclude} if exclude else set()
        self._cards = [c for c in FULL_DECK if c.id not in excluded_ids]
        self._rng = random.Random(seed)
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self._cards)} remain")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        return len(self._cards)

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def peek(self, n: int = 1) -> list[Card]:
        return self._cards[:n]

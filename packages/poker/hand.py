"""Hand representation and categorization."""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass

from packages.poker.card import Card


class HandCategory(IntEnum):
    """Hand rankings from weakest to strongest."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class Hand:
    """A player's hole cards."""
    cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        if len(self.cards) not in (2, 4):  # Hold'em or Omaha
            raise ValueError(f"Hand must have 2 or 4 cards, got {len(self.cards)}")

    @classmethod
    def from_str(cls, s: str) -> Hand:
        """Parse 'AhKs' into a Hand."""
        if len(s) % 2 != 0:
            raise ValueError(f"Invalid hand string: {s}")
        cards = tuple(Card.from_str(s[i : i + 2]) for i in range(0, len(s), 2))
        return cls(cards)

    @property
    def is_pocket_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    @property
    def is_suited(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].suit == self.cards[1].suit

    @property
    def high_rank(self):
        return max(c.rank for c in self.cards)

    @property
    def low_rank(self):
        return min(c.rank for c in self.cards)

    @property
    def gap(self) -> int:
        if len(self.cards) != 2:
            return 0
        return abs(self.cards[0].rank - self.cards[1].rank) - 1

    def canonical_str(self) -> str:
        """'AKs', 'TTo', '72o' etc."""
        if len(self.cards) != 2:
            return "".join(str(c) for c in self.cards)
        hi, lo = max(self.cards[0].rank, self.cards[1].rank), min(self.cards[0].rank, self.cards[1].rank)
        suffix = "s" if self.is_suited else ("" if self.is_pocket_pair else "o")
        return f"{hi}{lo}{suffix}"

    def __str__(self) -> str:
        return "".join(str(c) for c in self.cards)

    def __repr__(self) -> str:
        return f"Hand({self})"

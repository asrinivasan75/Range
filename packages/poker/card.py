"""Card, Rank, Suit primitives."""

from __future__ import annotations
from enum import IntEnum
from functools import total_ordering


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        return ["♣", "♦", "♥", "♠"][self.value]

    @classmethod
    def from_char(cls, c: str) -> Suit:
        return {
            "c": cls.CLUBS, "d": cls.DIAMONDS, "h": cls.HEARTS, "s": cls.SPADES,
            "♣": cls.CLUBS, "♦": cls.DIAMONDS, "♥": cls.HEARTS, "♠": cls.SPADES,
        }[c.lower() if len(c) == 1 and c.isascii() else c]


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        if self.value <= 9:
            return str(self.value)
        return {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]

    @classmethod
    def from_char(cls, c: str) -> Rank:
        mapping = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
            "T": 10, "t": 10, "J": 11, "j": 11, "Q": 12, "q": 12,
            "K": 13, "k": 13, "A": 14, "a": 14,
        }
        return cls(mapping[c])


@total_ordering
class Card:
    """A single playing card."""

    __slots__ = ("rank", "suit", "_id")

    def __init__(self, rank: Rank | int, suit: Suit | int) -> None:
        self.rank = Rank(rank) if isinstance(rank, int) else rank
        self.suit = Suit(suit) if isinstance(suit, int) else suit
        self._id: int = self.suit.value * 13 + (self.rank.value - 2)

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse 'Ah', 'Td', '2c' etc."""
        return cls(Rank.from_char(s[0]), Suit.from_char(s[1]))

    @property
    def id(self) -> int:
        """Unique integer 0-51."""
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._id == other._id

    def __lt__(self, other: Card) -> bool:
        return self.rank < other.rank or (self.rank == other.rank and self.suit < other.suit)

    def __hash__(self) -> int:
        return self._id

    @property
    def ascii(self) -> str:
        """ASCII-safe string like 'Ah', 'Td'."""
        suit_char = ["c", "d", "h", "s"][self.suit.value]
        return f"{self.rank}{suit_char}"

    def __repr__(self) -> str:
        return self.ascii

    def __str__(self) -> str:
        return self.ascii

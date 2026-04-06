"""Betting actions for poker game trees."""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass


class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Action:
    """A concrete action with optional sizing."""

    type: ActionType
    amount: float = 0.0

    def __str__(self) -> str:
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            return f"{self.type}:{self.amount:.0f}"
        return str(self.type)

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> Action:
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls, amount: float) -> Action:
        return cls(ActionType.CALL, amount)

    @classmethod
    def bet(cls, amount: float) -> Action:
        return cls(ActionType.BET, amount)

    @classmethod
    def raise_to(cls, amount: float) -> Action:
        return cls(ActionType.RAISE, amount)

    @classmethod
    def all_in(cls, amount: float) -> Action:
        return cls(ActionType.ALL_IN, amount)

    @property
    def is_aggressive(self) -> bool:
        return self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)

    @property
    def is_passive(self) -> bool:
        return self.type in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD)

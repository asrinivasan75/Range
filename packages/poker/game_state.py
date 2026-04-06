"""Game state representation for poker."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from packages.poker.card import Card
from packages.poker.actions import Action, ActionType


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class PlayerState:
    """State of a single player."""
    seat: int
    stack: float
    hole_cards: list[Card] = field(default_factory=list)
    is_folded: bool = False
    is_all_in: bool = False
    bet_this_street: float = 0.0
    total_invested: float = 0.0

    @property
    def is_active(self) -> bool:
        return not self.is_folded and not self.is_all_in


@dataclass
class GameState:
    """Full game state for a hand of poker."""
    players: list[PlayerState]
    board: list[Card] = field(default_factory=list)
    pot: float = 0.0
    street: Street = Street.PREFLOP
    current_player: int = 0
    action_history: list[list[Action]] = field(default_factory=lambda: [[] for _ in range(4)])
    small_blind: float = 1.0
    big_blind: float = 2.0
    is_terminal: bool = False
    winner: Optional[int] = None

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def active_players(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if p.is_active]

    @property
    def non_folded_players(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if not p.is_folded]

    @property
    def current_street_actions(self) -> list[Action]:
        return self.action_history[self.street.value]

    @property
    def to_call(self) -> float:
        """Amount the current player needs to call."""
        max_bet = max(p.bet_this_street for p in self.players)
        return max_bet - self.players[self.current_player].bet_this_street

    def get_legal_actions(self, bet_sizes: list[float] | None = None) -> list[Action]:
        """Get legal actions for the current player."""
        if self.is_terminal:
            return []

        player = self.players[self.current_player]
        if not player.is_active:
            return []

        actions: list[Action] = []
        to_call = self.to_call
        max_bet = max(p.bet_this_street for p in self.players)

        if to_call > 0:
            actions.append(Action.fold())
            if player.stack >= to_call:
                actions.append(Action.call(min(to_call, player.stack)))
            # Raise options
            min_raise = max_bet + self.big_blind
            if player.stack > to_call:
                if bet_sizes:
                    for size in bet_sizes:
                        raise_amount = self.pot + to_call + (self.pot + to_call) * size
                        if raise_amount <= player.stack and raise_amount >= min_raise:
                            actions.append(Action.raise_to(raise_amount))
                if player.stack > to_call:
                    actions.append(Action.all_in(player.stack))
        else:
            actions.append(Action.check())
            # Bet options
            if bet_sizes:
                for size in bet_sizes:
                    bet_amount = self.pot * size
                    if 0 < bet_amount <= player.stack and bet_amount >= self.big_blind:
                        actions.append(Action.bet(bet_amount))
            if player.stack > 0:
                actions.append(Action.all_in(player.stack))

        return actions

    def apply_action(self, action: Action) -> None:
        """Apply an action to mutate this game state."""
        player = self.players[self.current_player]
        self.action_history[self.street.value].append(action)

        if action.type == ActionType.FOLD:
            player.is_folded = True
            # Check if only one player remains
            if len(self.non_folded_players) == 1:
                self.is_terminal = True
                self.winner = self.non_folded_players[0]
                return

        elif action.type == ActionType.CHECK:
            pass

        elif action.type == ActionType.CALL:
            amount = min(action.amount, player.stack)
            player.stack -= amount
            player.bet_this_street += amount
            player.total_invested += amount
            self.pot += amount
            if player.stack == 0:
                player.is_all_in = True

        elif action.type in (ActionType.BET, ActionType.RAISE):
            amount = min(action.amount, player.stack)
            cost = amount - player.bet_this_street
            player.stack -= cost
            self.pot += cost
            player.total_invested += cost
            player.bet_this_street = amount
            if player.stack == 0:
                player.is_all_in = True

        elif action.type == ActionType.ALL_IN:
            amount = player.stack
            player.stack = 0
            player.total_invested += amount
            player.bet_this_street += amount
            self.pot += amount
            player.is_all_in = True

        # Advance to next active player
        self._advance_player()

    def _advance_player(self) -> None:
        """Move to the next player or next street."""
        active = self.active_players
        if len(active) <= 1 or self.is_terminal:
            if not self.is_terminal and len(self.non_folded_players) == 1:
                self.is_terminal = True
                self.winner = self.non_folded_players[0]
            return

        # Check if the street betting round is complete
        next_seat = (self.current_player + 1) % self.n_players
        while next_seat != self.current_player:
            if self.players[next_seat].is_active:
                break
            next_seat = (next_seat + 1) % self.n_players

        if next_seat == self.current_player:
            # Only one active player left
            return

        # Check if action is on the last player and all bets are matched
        all_matched = all(
            p.bet_this_street == max(pp.bet_this_street for pp in self.players)
            or p.is_folded or p.is_all_in
            for p in self.players
        )

        street_actions = self.current_street_actions
        all_acted = len(street_actions) >= len(active)

        if all_matched and all_acted:
            self._advance_street()
        else:
            self.current_player = next_seat

    def _advance_street(self) -> None:
        """Move to the next street."""
        for p in self.players:
            p.bet_this_street = 0.0

        if self.street == Street.RIVER or self.street.value >= 3:
            self.is_terminal = True
            return

        self.street = Street(self.street.value + 1)
        # Reset to first active player after dealer
        self.current_player = 0
        while not self.players[self.current_player].is_active:
            self.current_player = (self.current_player + 1) % self.n_players

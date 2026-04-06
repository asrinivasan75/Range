"""NLHE Betting Engine — immutable state machine for No-Limit Texas Hold'em.

This is the single source of truth for betting rules in interactive play.
All state transitions return new BettingState objects (never mutate).

Key design principles:
- No raise cap in no-limit
- Correct min raise: raise_to >= current_highest_bet + last_raise_increment
- Proper action reopening: only full raises reopen action
- Heads-up position: button/SB acts first preflop, BB acts first postflop
- Stack tracking: can't bet more than your stack
- Immutable: apply_action returns a new state
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Optional

from packages.poker.game_state import Street


class ActionKind(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class PlayerChips:
    seat: int
    stack: float
    bet_this_street: float
    total_invested: float
    is_folded: bool
    is_all_in: bool

    @property
    def is_active(self) -> bool:
        return not self.is_folded and not self.is_all_in


@dataclass(frozen=True)
class LegalAction:
    kind: ActionKind
    amount: float = 0.0       # exact amount for CALL; raise-to amount for BET/RAISE/ALL_IN
    min_amount: float = 0.0   # min raise-to (only meaningful for BET/RAISE)
    max_amount: float = 0.0   # max raise-to = stack (only meaningful for BET/RAISE)
    label: str = ""

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.kind == ActionKind.FOLD:
            return "Fold"
        if self.kind == ActionKind.CHECK:
            return "Check"
        if self.kind == ActionKind.CALL:
            return f"Call {self.amount:.1f}"
        if self.kind == ActionKind.BET:
            return f"Bet [{self.min_amount:.1f} - {self.max_amount:.1f}]"
        if self.kind == ActionKind.RAISE:
            return f"Raise [{self.min_amount:.1f} - {self.max_amount:.1f}]"
        if self.kind == ActionKind.ALL_IN:
            return f"All-in {self.amount:.1f}"
        return str(self.kind)


@dataclass(frozen=True)
class ActionRecord:
    """Record of an action taken, for the action log."""
    player_idx: int
    kind: ActionKind
    amount: float
    street: Street

    def describe(self, names: tuple[str, ...] = ("Player", "Bot")) -> str:
        name = names[self.player_idx] if self.player_idx < len(names) else f"P{self.player_idx}"
        if self.kind == ActionKind.FOLD:
            return f"{name} folds"
        if self.kind == ActionKind.CHECK:
            return f"{name} checks"
        if self.kind == ActionKind.CALL:
            return f"{name} calls {self.amount:.1f}"
        if self.kind in (ActionKind.BET, ActionKind.RAISE):
            verb = "bets" if self.kind == ActionKind.BET else "raises to"
            return f"{name} {verb} {self.amount:.1f}"
        if self.kind == ActionKind.ALL_IN:
            return f"{name} all-in {self.amount:.1f}"
        return f"{name} {self.kind}"


@dataclass(frozen=True)
class BettingState:
    """Immutable snapshot of the betting state."""
    street: Street
    pot: float
    players: tuple[PlayerChips, ...]
    current_player_idx: int
    last_raise_size: float                          # increment of last aggressive action
    last_aggressor_idx: Optional[int]               # who made the last aggressive action
    players_acted_since_last_raise: frozenset[int]  # for action reopening / street-end
    is_hand_complete: bool
    button_seat: int
    small_blind: float
    big_blind: float
    action_log: tuple[ActionRecord, ...]

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def active_player_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if p.is_active]

    @property
    def non_folded_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if not p.is_folded]

    @property
    def current_player(self) -> PlayerChips:
        return self.players[self.current_player_idx]

    @property
    def highest_bet(self) -> float:
        return max(p.bet_this_street for p in self.players)

    @property
    def to_call(self) -> float:
        return self.highest_bet - self.current_player.bet_this_street


class BettingEngine:
    """Stateless engine — all methods take and return BettingState."""

    @staticmethod
    def create_hand(
        button_seat: int,
        stacks: tuple[float, ...],
        sb: float = 1.0,
        bb: float = 2.0,
    ) -> BettingState:
        """Create a new hand with blinds posted."""
        n = len(stacks)
        assert n == 2, "Currently supports heads-up only"
        assert button_seat in (0, 1)

        bb_seat = 1 - button_seat  # non-button posts BB

        players = []
        log = []
        for i in range(n):
            blind = sb if i == button_seat else bb
            posted = min(blind, stacks[i])
            players.append(PlayerChips(
                seat=i,
                stack=stacks[i] - posted,
                bet_this_street=posted,
                total_invested=posted,
                is_folded=False,
                is_all_in=(stacks[i] - posted) == 0,
            ))
            blind_name = "SB" if i == button_seat else "BB"
            log.append(ActionRecord(i, ActionKind.BET, posted, Street.PREFLOP))

        pot = sum(p.bet_this_street for p in players)

        return BettingState(
            street=Street.PREFLOP,
            pot=pot,
            players=tuple(players),
            current_player_idx=button_seat,  # SB/button acts first preflop in HU
            last_raise_size=bb,  # BB is the "opening raise" for min-raise purposes
            last_aggressor_idx=bb_seat,  # BB is the initial aggressor
            players_acted_since_last_raise=frozenset(),  # no one has acted yet
            is_hand_complete=False,
            button_seat=button_seat,
            small_blind=sb,
            big_blind=bb,
            action_log=tuple(log),
        )

    @staticmethod
    def get_legal_actions(state: BettingState) -> list[LegalAction]:
        """Get all legal actions for the current player. No raise cap."""
        if state.is_hand_complete:
            return []

        idx = state.current_player_idx
        p = state.current_player
        if not p.is_active:
            return []

        actions: list[LegalAction] = []
        to_call = state.to_call
        highest = state.highest_bet

        if to_call > 0:
            # Facing a bet/raise
            actions.append(LegalAction(ActionKind.FOLD, label="Fold"))

            call_amount = min(to_call, p.stack)
            if call_amount >= to_call:
                actions.append(LegalAction(ActionKind.CALL, amount=call_amount,
                                           label=f"Call {call_amount:.1f}"))
            elif call_amount > 0:
                # Can only call for less (all-in)
                actions.append(LegalAction(ActionKind.ALL_IN, amount=p.stack,
                                           label=f"All-in {p.stack:.1f}"))
                return actions  # can't raise if can't even call fully

            # Raise option (NO CAP) — but only if action is open for this player.
            # Action is NOT open if: player already acted since the last FULL raise
            # (i.e., a short all-in did not reopen action for them).
            action_open = idx not in state.players_acted_since_last_raise
            remaining_after_call = p.stack - to_call
            if remaining_after_call > 0 and action_open:
                min_raise_to = highest + state.last_raise_size
                max_raise_to = p.bet_this_street + p.stack  # total they can put in

                if max_raise_to >= min_raise_to:
                    actions.append(LegalAction(
                        ActionKind.RAISE,
                        amount=min_raise_to,
                        min_amount=min_raise_to,
                        max_amount=max_raise_to,
                        label=f"Raise [{min_raise_to:.1f} - {max_raise_to:.1f}]",
                    ))
                elif max_raise_to > highest:
                    # Can't make a full raise but can go all-in for more than a call
                    actions.append(LegalAction(ActionKind.ALL_IN, amount=p.stack,
                                               label=f"All-in {p.stack:.1f}"))
        else:
            # No bet to face
            actions.append(LegalAction(ActionKind.CHECK, label="Check"))

            if p.stack > 0:
                min_bet = min(state.big_blind, p.stack)
                max_bet = p.stack

                if max_bet >= min_bet:
                    actions.append(LegalAction(
                        ActionKind.BET,
                        amount=min_bet,
                        min_amount=min_bet,
                        max_amount=max_bet,
                        label=f"Bet [{min_bet:.1f} - {max_bet:.1f}]",
                    ))

        return actions

    @staticmethod
    def apply_action(
        state: BettingState,
        kind: ActionKind,
        amount: float = 0.0,
    ) -> BettingState:
        """Apply an action and return the new state. Validates legality."""
        if state.is_hand_complete:
            raise ValueError("Hand is already complete")

        p = state.current_player
        idx = state.current_player_idx
        players = list(state.players)
        log = list(state.action_log)
        pot = state.pot
        last_raise_size = state.last_raise_size
        last_aggressor = state.last_aggressor_idx
        acted = set(state.players_acted_since_last_raise)
        is_complete = False

        if kind == ActionKind.FOLD:
            players[idx] = replace(p, is_folded=True)
            log.append(ActionRecord(idx, ActionKind.FOLD, 0, state.street))
            acted.add(idx)
            # Check if hand ends
            non_folded = [i for i, pp in enumerate(players) if not pp.is_folded]
            if len(non_folded) <= 1:
                is_complete = True

        elif kind == ActionKind.CHECK:
            if state.to_call > 0:
                raise ValueError("Cannot check when facing a bet")
            log.append(ActionRecord(idx, ActionKind.CHECK, 0, state.street))
            acted.add(idx)

        elif kind == ActionKind.CALL:
            to_call = min(state.to_call, p.stack)
            if to_call <= 0:
                raise ValueError("Nothing to call")
            new_stack = p.stack - to_call
            new_bet = p.bet_this_street + to_call
            players[idx] = replace(p,
                                   stack=new_stack,
                                   bet_this_street=new_bet,
                                   total_invested=p.total_invested + to_call,
                                   is_all_in=(new_stack == 0))
            pot += to_call
            log.append(ActionRecord(idx, ActionKind.CALL, to_call, state.street))
            acted.add(idx)

        elif kind in (ActionKind.BET, ActionKind.RAISE):
            # amount is the total raise-to (not the increment)
            highest = state.highest_bet
            cost = amount - p.bet_this_street
            if cost <= 0:
                raise ValueError(f"Raise-to {amount} is not more than current bet {p.bet_this_street}")
            # Clamp to stack to handle float precision (e.g., 63.7 vs 63.6999)
            cost = min(cost, p.stack)

            raise_increment = amount - highest
            if raise_increment < 0:
                raise ValueError(f"Raise-to {amount} is less than current highest {highest}")

            new_stack = p.stack - cost
            players[idx] = replace(p,
                                   stack=new_stack,
                                   bet_this_street=amount,
                                   total_invested=p.total_invested + cost,
                                   is_all_in=(new_stack == 0))
            pot += cost

            # Update raise tracking — only a FULL raise reopens action
            if raise_increment >= last_raise_size:
                last_raise_size = raise_increment
                last_aggressor = idx
                acted = {idx}  # reset — only the raiser has acted
            else:
                # Short raise (e.g., all-in for less) — does NOT reopen
                acted.add(idx)

            log.append(ActionRecord(idx, kind, amount, state.street))

        elif kind == ActionKind.ALL_IN:
            cost = p.stack
            if cost <= 0:
                raise ValueError("No chips to go all-in with")
            new_bet = p.bet_this_street + cost
            players[idx] = replace(p,
                                   stack=0,
                                   bet_this_street=new_bet,
                                   total_invested=p.total_invested + cost,
                                   is_all_in=True)
            pot += cost

            highest = max(pp.bet_this_street for pp in players)
            raise_increment = new_bet - state.highest_bet

            # Check if this is a full raise
            if raise_increment >= last_raise_size and new_bet > state.highest_bet:
                last_raise_size = raise_increment
                last_aggressor = idx
                acted = {idx}
            else:
                acted.add(idx)

            log.append(ActionRecord(idx, ActionKind.ALL_IN, new_bet, state.street))
        else:
            raise ValueError(f"Unknown action kind: {kind}")

        # Find next player
        next_idx = BettingEngine._next_active_player(players, idx)

        # Check if street/hand is complete
        if not is_complete:
            is_complete = BettingEngine._check_hand_complete(players)

        new_state = BettingState(
            street=state.street,
            pot=pot,
            players=tuple(players),
            current_player_idx=next_idx if not is_complete else idx,
            last_raise_size=last_raise_size,
            last_aggressor_idx=last_aggressor,
            players_acted_since_last_raise=frozenset(acted),
            is_hand_complete=is_complete,
            button_seat=state.button_seat,
            small_blind=state.small_blind,
            big_blind=state.big_blind,
            action_log=tuple(log),
        )

        return new_state

    @staticmethod
    def is_street_complete(state: BettingState) -> bool:
        """Check if the current betting round is over."""
        if state.is_hand_complete:
            return True

        active = [i for i, p in enumerate(state.players) if p.is_active]
        if len(active) == 0:
            return True  # everyone all-in or folded

        # All active players must have acted since the last raise
        if not all(i in state.players_acted_since_last_raise for i in active):
            return False

        # All bets must be matched (non-folded, non-all-in players)
        highest = state.highest_bet
        for i in active:
            if state.players[i].bet_this_street != highest:
                return False

        return True

    @staticmethod
    def advance_street(state: BettingState) -> BettingState:
        """Advance to the next street. Resets per-street bets."""
        if state.street == Street.RIVER:
            return replace(state, is_hand_complete=True)

        new_street = Street(state.street.value + 1)

        # Reset per-street bets
        players = []
        for p in state.players:
            players.append(replace(p, bet_this_street=0.0))

        # Postflop: non-button acts first in heads-up
        bb_seat = 1 - state.button_seat
        first_active = bb_seat
        if not players[first_active].is_active:
            first_active = BettingEngine._next_active_player(players, first_active)

        # If only one active player or none, mark hand complete
        active_count = sum(1 for p in players if p.is_active)
        hand_complete = active_count <= 1 and not state.is_hand_complete

        return BettingState(
            street=new_street,
            pot=state.pot,
            players=tuple(players),
            current_player_idx=first_active,
            last_raise_size=state.big_blind,
            last_aggressor_idx=None,
            players_acted_since_last_raise=frozenset(),
            is_hand_complete=hand_complete or state.is_hand_complete,
            button_seat=state.button_seat,
            small_blind=state.small_blind,
            big_blind=state.big_blind,
            action_log=state.action_log,
        )

    @staticmethod
    def effective_stack(state: BettingState, p1: int = 0, p2: int = 1) -> float:
        """Min of two players' remaining stacks."""
        return min(state.players[p1].stack, state.players[p2].stack)

    @staticmethod
    def spr(state: BettingState, p1: int = 0, p2: int = 1) -> float:
        """Stack-to-pot ratio."""
        if state.pot <= 0:
            return float("inf")
        return BettingEngine.effective_stack(state, p1, p2) / state.pot

    @staticmethod
    def _next_active_player(players: list | tuple, current: int) -> int:
        """Find the next active (non-folded, non-all-in) player."""
        n = len(players)
        for offset in range(1, n + 1):
            idx = (current + offset) % n
            if players[idx].is_active:
                return idx
        return current  # no active players

    @staticmethod
    def _check_hand_complete(players: list | tuple) -> bool:
        """Hand is complete if 0 or 1 non-folded players remain."""
        non_folded = sum(1 for p in players if not p.is_folded)
        return non_folded <= 1

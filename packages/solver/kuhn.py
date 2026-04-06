"""Kuhn Poker — the simplest imperfect-information game for CFR validation.

3 cards (J, Q, K), 2 players. Each antes 1 chip, dealt 1 card each.
Actions: check (p) or bet (b).
- P1 acts: check or bet
  - P1 checks -> P2 acts: check or bet
    - P2 checks -> showdown
    - P2 bets -> P1 acts: fold or call
  - P1 bets -> P2 acts: fold or call

Nash equilibrium is known, making this ideal for testing CFR correctness.
"""

from __future__ import annotations
from itertools import permutations


class KuhnPoker:
    """Kuhn Poker game implementation for CFR training."""

    CARDS = ["J", "Q", "K"]

    def n_players(self) -> int:
        return 2

    def is_terminal(self, history: str) -> bool:
        # History format: "<card0><card1><actions>"
        # e.g., "JQ" + "pb" + "c" = "JQpbc"
        if len(history) < 2:
            return False
        actions = history[2:]
        if actions in ("pp", "bb", "pbb", "pbf", "bp", "bf"):
            return True
        # "bp" = bet, pass(fold) — wait, let me use standard: p=pass/check, b=bet, f=fold, c=call
        # Let me re-define:
        # p = check/pass, b = bet, f = fold, c = call
        return actions in ("pp", "bc", "bf", "pbc", "pbf")

    def terminal_utility(self, history: str, player: int) -> float:
        """Utility for `player` at terminal node."""
        cards = history[:2]
        actions = history[2:]
        my_card = self.CARDS.index(cards[player])
        opp_card = self.CARDS.index(cards[1 - player])

        if actions == "pp":
            # Check-check: showdown, pot = 2 (1 ante each)
            return 1.0 if my_card > opp_card else -1.0

        if actions == "bf":
            # P1 bets, P2 folds
            return 1.0 if player == 0 else -1.0

        if actions == "bc":
            # P1 bets, P2 calls: showdown, pot = 4
            return 2.0 if my_card > opp_card else -2.0

        if actions == "pbf":
            # P1 checks, P2 bets, P1 folds
            return -1.0 if player == 0 else 1.0

        if actions == "pbc":
            # P1 checks, P2 bets, P1 calls: showdown, pot = 4
            return 2.0 if my_card > opp_card else -2.0

        raise ValueError(f"Unknown terminal history: {history}")

    def is_chance(self, history: str) -> bool:
        return len(history) < 2

    def chance_actions(self, history: str) -> list[tuple[str, float]]:
        """Return possible chance outcomes with probabilities."""
        if len(history) == 0:
            # Deal first card: all 3 cards equally likely for first slot
            return [(c, 1 / 3) for c in self.CARDS]
        elif len(history) == 1:
            # Deal second card: 2 remaining cards
            remaining = [c for c in self.CARDS if c != history[0]]
            return [(c, 1 / 2) for c in remaining]
        return []

    def current_player(self, history: str) -> int:
        actions = history[2:]
        if len(actions) == 0:
            return 0
        if len(actions) == 1:
            return 1
        return 0  # Only reaches here for "pb" -> P1 acts again

    def info_set_key(self, history: str, player: int) -> str:
        """Information set: card + action sequence (opponent's card is hidden)."""
        card = history[player]
        actions = history[2:]
        return f"{card}:{actions}"

    def actions(self, history: str) -> list[str]:
        actions = history[2:]
        if len(actions) == 0:
            return ["p", "b"]  # P1: check or bet
        if actions == "p":
            return ["p", "b"]  # P2: check or bet
        if actions == "b":
            return ["f", "c"]  # P2: fold or call
        if actions == "pb":
            return ["f", "c"]  # P1: fold or call
        return []

    @staticmethod
    def known_nash_equilibrium() -> dict:
        """Return the known Nash equilibrium for Kuhn Poker.

        This is useful for validating CFR convergence.
        """
        return {
            "J:": {"check": 1 - 1/3, "bet": 1/3},  # P1 w/ J: bet 1/3 (bluff)
            "Q:": {"check": 1.0, "bet": 0.0},        # P1 w/ Q: always check
            "K:": {"check": 1 - 3*1/3, "bet": 1.0},  # P1 w/ K: always bet (value)
            "J:b": {"fold": 1.0, "call": 0.0},        # P2 w/ J vs bet: always fold
            "Q:b": {"fold": 2/3, "call": 1/3},        # P2 w/ Q vs bet: call 1/3
            "K:b": {"fold": 0.0, "call": 1.0},        # P2 w/ K vs bet: always call
            "J:p": {"check": 2/3, "bet": 1/3},        # P2 w/ J after check: bet 1/3
            "Q:p": {"check": 1.0, "bet": 0.0},        # P2 w/ Q after check: check
            "K:p": {"check": 0.0, "bet": 1.0},        # P2 w/ K after check: always bet
            "J:pb": {"fold": 1.0, "call": 0.0},       # P1 w/ J vs check-bet: fold
            "Q:pb": {"fold": 2/3, "call": 1/3},       # P1 w/ Q vs check-bet: call 1/3
            "K:pb": {"fold": 0.0, "call": 1.0},       # P1 w/ K vs check-bet: always call
            "game_value_p1": -1/18,
        }

"""Full-street Simplified Hold'em for CFR training.

Covers ALL 4 streets (preflop, flop, turn, river) with coarse abstraction.
Designed so that every decision the bot makes comes from trained strategy.

History encoding (flat string for CFR concatenation):
  <pre_b0><pre_b1><preflop_actions>.<flop_b0><flop_b1><flop_actions>.<turn_b0><turn_b1><turn_actions>.<river_b0><river_b1><river_actions>

  Each street: 2 bucket chars + action chars, separated by "."
  Bucket chars: '0'-'9' (single digit per player)

Action characters:
  k = check
  t = bet 1/3 pot
  s = bet 2/3 pot
  p = bet pot
  f = fold
  c = call
  r = raise (pot-sized)
"""

from __future__ import annotations
from dataclasses import dataclass

ACTION_NAMES = {
    "f": "Fold", "k": "Check", "c": "Call",
    "t": "Bet 1/3", "s": "Bet 2/3", "p": "Bet Pot",
    "r": "Raise",
}

BET_FRACTIONS = {"t": 1/3, "s": 2/3, "p": 1.0, "r": 1.0}

STREET_NAMES = ["preflop", "flop", "turn", "river"]


@dataclass
class FullHoldemConfig:
    preflop_buckets: int = 5
    flop_buckets: int = 5
    turn_buckets: int = 5
    river_buckets: int = 5
    max_raises_per_street: int = 2
    starting_pot: float = 3.0


class FullStreetHoldem:
    """4-street Hold'em with coarse abstraction for CFR training."""

    def __init__(self, config: FullHoldemConfig | None = None) -> None:
        self.config = config or FullHoldemConfig()
        self._buckets = [
            self.config.preflop_buckets,
            self.config.flop_buckets,
            self.config.turn_buckets,
            self.config.river_buckets,
        ]
        self._max_raises = self.config.max_raises_per_street
        for b in self._buckets:
            assert b <= 10, "Max 10 buckets per street (single digit encoding)"

    def n_players(self) -> int:
        return 2

    def _parse(self, history: str) -> dict:
        """Parse flat history into structured components."""
        r = {
            "buckets": [[None, None] for _ in range(4)],  # [street][player]
            "actions": ["", "", "", ""],  # per street
            "current_street": 0,
            "phase": "deal_0_0",  # deal_<street>_<player> or "street_<n>" or "showdown"
        }

        if not history:
            return r

        # Split by "." to get street segments
        segments = history.split(".")

        for street_idx, seg in enumerate(segments):
            if not seg:
                # Empty segment — need to deal this street
                r["current_street"] = street_idx
                r["phase"] = f"deal_{street_idx}_0"
                return r

            # First char: P0 bucket
            r["buckets"][street_idx][0] = int(seg[0])
            if len(seg) == 1:
                r["current_street"] = street_idx
                r["phase"] = f"deal_{street_idx}_1"
                return r

            # Second char: P1 bucket
            r["buckets"][street_idx][1] = int(seg[1])
            r["actions"][street_idx] = seg[2:]
            r["current_street"] = street_idx

        # Determine phase
        last_street = len(segments) - 1
        last_actions = r["actions"][last_street]

        if self._street_done(last_actions):
            if "f" in last_actions:
                r["phase"] = "terminal_fold"
            elif last_street == 3:
                r["phase"] = "showdown"
            else:
                # Need to advance to next street
                next_s = last_street + 1
                r["current_street"] = next_s
                r["phase"] = f"deal_{next_s}_0"
        else:
            r["phase"] = f"street_{last_street}"

        return r

    def _street_done(self, actions: str) -> bool:
        if not actions:
            return False
        if "f" in actions:
            return True
        if len(actions) >= 2:
            if actions == "kk":
                return True
            if actions[-1] == "c":
                return True
        return False

    def is_terminal(self, history: str) -> bool:
        p = self._parse(history)
        return p["phase"] in ("terminal_fold", "showdown")

    def terminal_utility(self, history: str, player: int) -> float:
        p = self._parse(history)
        pot = self._compute_pot(p)

        if p["phase"] == "terminal_fold":
            folder = self._find_folder(p)
            return -(pot / 2) if player == folder else (pot / 2)

        # Showdown — compare best buckets (later streets override earlier)
        s0 = self._best_strength(p, 0)
        s1 = self._best_strength(p, 1)
        if s0 > s1:
            return pot / 2 if player == 0 else -(pot / 2)
        elif s0 < s1:
            return -(pot / 2) if player == 0 else pot / 2
        return 0.0

    def _best_strength(self, parsed: dict, player: int) -> int:
        """Get the latest available bucket for a player (later streets are more accurate)."""
        for street in range(3, -1, -1):
            if parsed["buckets"][street][player] is not None:
                return parsed["buckets"][street][player]
        return 0

    def _find_folder(self, parsed: dict) -> int:
        for street in range(4):
            for i, a in enumerate(parsed["actions"][street]):
                if a == "f":
                    return i % 2
        return 0

    def _compute_pot(self, parsed: dict) -> float:
        pot = self.config.starting_pot
        for street in range(4):
            acts = parsed["actions"][street]
            if not acts:
                break
            street_pot = pot
            bets = [0.0, 0.0]
            for i, a in enumerate(acts):
                who = i % 2
                if a in BET_FRACTIONS:
                    bets[who] = max(bets) + street_pot * BET_FRACTIONS[a]
                elif a == "c":
                    bets[who] = max(bets)
            pot = street_pot + sum(bets)
        return pot

    def is_chance(self, history: str) -> bool:
        p = self._parse(history)
        return p["phase"].startswith("deal_")

    def chance_actions(self, history: str) -> list[tuple[str, float]]:
        p = self._parse(history)
        phase = p["phase"]
        if not phase.startswith("deal_"):
            return []

        parts = phase.split("_")
        street_idx = int(parts[1])
        player_idx = int(parts[2])
        n_buckets = self._buckets[street_idx]
        prob = 1.0 / n_buckets

        if player_idx == 0 and street_idx > 0:
            # Need "." separator before first bucket of new street
            if not history.endswith("."):
                return [("." + str(b), prob) for b in range(n_buckets)]
        return [(str(b), prob) for b in range(n_buckets)]

    def current_player(self, history: str) -> int:
        p = self._parse(history)
        phase = p["phase"]
        if phase.startswith("street_"):
            street = int(phase.split("_")[1])
            return len(p["actions"][street]) % 2
        return 0

    def info_set_key(self, history: str, player: int) -> str:
        """Info set: player's own buckets + full action history."""
        p = self._parse(history)

        # Build bucket part — only include this player's buckets
        bucket_parts = []
        for street in range(4):
            b = p["buckets"][street][player]
            if b is not None:
                bucket_parts.append(f"{'PFTR'[street]}{b}")

        # Build action part — full action history (opponent's actions visible)
        action_parts = []
        for street in range(4):
            if p["actions"][street] or p["buckets"][street][0] is not None:
                action_parts.append(p["actions"][street])

        return "".join(bucket_parts) + "|" + "/".join(action_parts)

    def actions(self, history: str) -> list[str]:
        p = self._parse(history)
        phase = p["phase"]
        if not phase.startswith("street_"):
            return []
        street = int(phase.split("_")[1])
        return self._legal_actions(p["actions"][street])

    def _legal_actions(self, street_actions: str) -> list[str]:
        n_aggressive = sum(1 for a in street_actions if a in "tspr")
        has_bet = n_aggressive > 0
        at_cap = n_aggressive >= self._max_raises

        if not has_bet:
            return ["k", "t", "s", "p"]  # check, 1/3, 2/3, pot
        elif at_cap:
            return ["f", "c"]
        else:
            return ["f", "c", "r"]  # fold, call, raise

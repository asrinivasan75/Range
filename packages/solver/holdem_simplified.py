"""Simplified Texas Hold'em for CFR training.

2-player heads-up Hold'em with:
- Preflop + Flop (2 streets for solver training)
- Coarse hand abstraction (configurable buckets, max 10)
- 3 bet sizes: 1/3 pot, 1/2 pot, 4/3 pot
- Raise option when facing a bet (pot-sized raise)
- Max raises per street configurable

History encoding (flat string, compatible with CFR concatenation):
  Position 0:   P0 preflop bucket (char '0'-'9')
  Position 1:   P1 preflop bucket (char '0'-'9')
  Position 2+:  Preflop actions
  '.' separator when preflop ends
  After '.':    P0 flop bucket, P1 flop bucket (2 chars)
  Then:         Flop actions

Action characters:
  k = check
  t = bet 1/3 pot
  h = bet 1/2 pot
  o = bet 4/3 pot (overbet)
  f = fold
  c = call
  r = raise (pot-sized)
"""

from __future__ import annotations
from dataclasses import dataclass


ACTION_NAMES = {
    "f": "Fold", "k": "Check", "c": "Call",
    "q": "Bet 1/4", "t": "Bet 1/3", "h": "Bet 1/2",
    "s": "Bet 2/3", "p": "Bet Pot", "o": "Bet 1.5x",
    "r": "Raise",
}

BET_FRACTIONS = {"q": 1/4, "t": 1/3, "h": 1/2, "s": 2/3, "p": 1.0, "o": 1.5, "r": 1.0}


@dataclass
class SimplifiedHoldemConfig:
    preflop_buckets: int = 8
    flop_buckets: int = 8
    max_raises_per_street: int = 2
    starting_pot: float = 3.0
    starting_stack: float = 100.0


class SimplifiedHoldem:
    """Simplified Hold'em with flat history encoding for CFR compatibility."""

    def __init__(self, config: SimplifiedHoldemConfig | None = None) -> None:
        self.config = config or SimplifiedHoldemConfig()
        assert self.config.preflop_buckets <= 10
        assert self.config.flop_buckets <= 10
        self._n_pre = self.config.preflop_buckets
        self._n_flop = self.config.flop_buckets
        self._max_raises = self.config.max_raises_per_street

    def n_players(self) -> int:
        return 2

    def _parse(self, history: str) -> dict:
        r = {
            "pre_b0": None, "pre_b1": None,
            "flop_b0": None, "flop_b1": None,
            "pre_actions": "", "flop_actions": "",
            "phase": "deal_pre_0",
        }
        n = len(history)
        if n == 0:
            return r

        r["pre_b0"] = int(history[0])
        if n == 1:
            r["phase"] = "deal_pre_1"
            return r

        r["pre_b1"] = int(history[1])
        dot_pos = history.find(".", 2)

        if dot_pos == -1:
            r["pre_actions"] = history[2:]
            r["phase"] = "deal_flop_0" if self._street_done(r["pre_actions"]) else "preflop"
            return r

        r["pre_actions"] = history[2:dot_pos]
        flop_part = history[dot_pos + 1:]

        if len(flop_part) == 0:
            r["phase"] = "deal_flop_0"
            return r
        r["flop_b0"] = int(flop_part[0])
        if len(flop_part) == 1:
            r["phase"] = "deal_flop_1"
            return r
        r["flop_b1"] = int(flop_part[1])
        r["flop_actions"] = flop_part[2:]
        r["phase"] = "showdown" if self._street_done(r["flop_actions"]) else "flop"
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

    def _has_fold(self, parsed: dict) -> bool:
        return "f" in parsed["pre_actions"] or "f" in parsed["flop_actions"]

    def _folder(self, parsed: dict) -> int:
        for street_key in ["pre_actions", "flop_actions"]:
            for i, a in enumerate(parsed[street_key]):
                if a == "f":
                    return i % 2
        return 0

    def is_terminal(self, history: str) -> bool:
        p = self._parse(history)
        return self._has_fold(p) or p["phase"] == "showdown"

    def terminal_utility(self, history: str, player: int) -> float:
        p = self._parse(history)
        pot = self._pot(p)
        if self._has_fold(p):
            folder = self._folder(p)
            return -(pot / 2) if player == folder else (pot / 2)
        s0 = p["flop_b0"] if p["flop_b0"] is not None else p["pre_b0"]
        s1 = p["flop_b1"] if p["flop_b1"] is not None else p["pre_b1"]
        if s0 > s1:
            return pot / 2 if player == 0 else -(pot / 2)
        elif s0 < s1:
            return -(pot / 2) if player == 0 else pot / 2
        return 0.0

    def _pot(self, p: dict) -> float:
        pot = self.config.starting_pot
        for acts in [p["pre_actions"], p["flop_actions"]]:
            street_pot = pot
            bets = [0.0, 0.0]
            for i, a in enumerate(acts):
                who = i % 2
                if a in BET_FRACTIONS:
                    frac = BET_FRACTIONS[a]
                    bets[who] = max(bets) + street_pot * frac
                elif a == "c":
                    bets[who] = max(bets)
            pot = street_pot + sum(bets)
        return pot

    def is_chance(self, history: str) -> bool:
        return self._parse(history)["phase"].startswith("deal_")

    def chance_actions(self, history: str) -> list[tuple[str, float]]:
        p = self._parse(history)
        phase = p["phase"]
        if phase == "deal_pre_0":
            return [(str(i), 1.0 / self._n_pre) for i in range(self._n_pre)]
        if phase == "deal_pre_1":
            return [(str(i), 1.0 / self._n_pre) for i in range(self._n_pre)]
        if phase == "deal_flop_0":
            if "." not in history:
                return [("." + str(i), 1.0 / self._n_flop) for i in range(self._n_flop)]
            return [(str(i), 1.0 / self._n_flop) for i in range(self._n_flop)]
        if phase == "deal_flop_1":
            return [(str(i), 1.0 / self._n_flop) for i in range(self._n_flop)]
        return []

    def current_player(self, history: str) -> int:
        p = self._parse(history)
        if p["phase"] == "preflop":
            return len(p["pre_actions"]) % 2
        if p["phase"] == "flop":
            return len(p["flop_actions"]) % 2
        return 0

    def info_set_key(self, history: str, player: int) -> str:
        p = self._parse(history)
        pre_bucket = p["pre_b0"] if player == 0 else p["pre_b1"]
        flop_bucket = p["flop_b0"] if player == 0 else p["flop_b1"]
        if p["phase"] in ("preflop", "deal_flop_0"):
            return f"P{pre_bucket}|{p['pre_actions']}"
        if p["phase"] in ("flop", "showdown", "deal_flop_1"):
            flop_str = f"F{flop_bucket}" if flop_bucket is not None else ""
            return f"P{pre_bucket}{flop_str}|{p['pre_actions']}/{p['flop_actions']}"
        return f"P{pre_bucket}|"

    def actions(self, history: str) -> list[str]:
        p = self._parse(history)
        if p["phase"] == "preflop":
            return self._legal_actions(p["pre_actions"])
        if p["phase"] == "flop":
            return self._legal_actions(p["flop_actions"])
        return []

    def _legal_actions(self, street_actions: str) -> list[str]:
        n_aggressive = sum(1 for a in street_actions if a in "qthspor")
        has_bet = n_aggressive > 0
        at_cap = n_aggressive >= self._max_raises

        if not has_bet:
            return ["k", "q", "t", "h", "s", "p", "o"]  # check + 6 bet sizes
        elif at_cap:
            return ["f", "c"]  # fold, call
        else:
            return ["f", "c", "r"]  # fold, call, raise

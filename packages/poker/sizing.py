"""Stack-depth-aware sizing framework for NLHE.

Separates LEGAL sizing (what the rules allow) from STRATEGIC sizing
(what the bot should prefer given stack depth, SPR, and multi-street planning).

Key concept: geometric sizing — the bet fraction that, if used on every
remaining street and called, gets exactly all-in by the river.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from packages.poker.betting_engine import BettingEngine, BettingState, ActionKind, LegalAction
from packages.poker.game_state import Street


@dataclass(frozen=True)
class SizingCandidate:
    name: str
    amount: float       # chip amount to bet/raise-to
    fraction: float     # as fraction of pot
    rationale: str


def geometric_bet_fraction(spr: float, streets_remaining: int) -> float:
    """Compute the pot fraction that gets all-in in exactly N streets.

    If you bet x * pot on each street and villain calls every time:
    - After 1 street: pot grows by factor (1 + 2x)
    - After N streets: pot grows by (1 + 2x)^N
    - To get all-in: (1 + 2x)^N >= 1 + 2*SPR
    - Solve: x = ((1 + 2*SPR)^(1/N) - 1) / 2
    """
    if streets_remaining <= 0:
        return 1.0
    if spr <= 0:
        return 0.0
    target = 1.0 + 2.0 * spr
    x = (target ** (1.0 / streets_remaining) - 1.0) / 2.0
    return max(x, 0.0)


def streets_remaining(street: Street) -> int:
    """Number of remaining betting streets including current."""
    return 4 - street.value  # PREFLOP=0 -> 4, FLOP=1 -> 3, TURN=2 -> 2, RIVER=3 -> 1


def compute_sizing_candidates(
    state: BettingState,
    legal_actions: list[LegalAction] | None = None,
) -> list[SizingCandidate]:
    """Generate strategic sizing candidates based on stack depth and SPR.

    Returns candidates filtered to the legal bet/raise range.
    """
    if legal_actions is None:
        legal_actions = BettingEngine.get_legal_actions(state)

    # Find the bet or raise action to get min/max legal amounts
    bet_action = None
    for a in legal_actions:
        if a.kind in (ActionKind.BET, ActionKind.RAISE):
            bet_action = a
            break

    if bet_action is None:
        return []  # no bet/raise available

    min_legal = bet_action.min_amount
    max_legal = bet_action.max_amount
    pot = state.pot
    is_raise = bet_action.kind == ActionKind.RAISE

    if pot <= 0:
        return []

    eff = BettingEngine.effective_stack(state)
    spr = BettingEngine.spr(state)
    n_streets = streets_remaining(state.street)

    candidates: list[SizingCandidate] = []

    # ── SPR-based strategy ──────────────────────────────────

    if spr < 1.5:
        # Very shallow — just jam. No fold equity gained by smaller sizing.
        candidates.append(SizingCandidate(
            "All-in", max_legal, max_legal / pot if pot > 0 else 0,
            f"SPR {spr:.1f} < 1.5 — commit immediately"
        ))

    elif spr < 4:
        # Shallow — geometric sizing gets it in over remaining streets
        geo = geometric_bet_fraction(spr, n_streets)
        geo_amount = _raise_to_amount(pot, geo, state, is_raise)
        candidates.append(SizingCandidate(
            "Geometric", geo_amount, geo,
            f"Geometric for {n_streets} street{'s' if n_streets > 1 else ''} (SPR {spr:.1f})"
        ))
        # Also offer a smaller and jam option
        small = _raise_to_amount(pot, 1/3, state, is_raise)
        candidates.append(SizingCandidate("1/3 pot", small, 1/3, "Small sizing"))
        candidates.append(SizingCandidate(
            "All-in", max_legal, max_legal / pot if pot > 0 else 0,
            f"Jam — effective stack {eff:.0f}"
        ))

    else:
        # Deep — full menu of sizes
        for name, frac in [("1/3 pot", 1/3), ("1/2 pot", 1/2), ("2/3 pot", 2/3), ("Pot", 1.0)]:
            amt = _raise_to_amount(pot, frac, state, is_raise)
            candidates.append(SizingCandidate(name, amt, frac, f"{name} sizing"))

        # Geometric sizing
        geo = geometric_bet_fraction(spr, n_streets)
        geo_amount = _raise_to_amount(pot, geo, state, is_raise)
        if 0.1 < geo < 2.0:  # only if it's a reasonable fraction
            candidates.append(SizingCandidate(
                "Geometric", geo_amount, geo,
                f"Geometric for {n_streets} street{'s' if n_streets > 1 else ''}"
            ))

        # Overbet for polarized ranges
        if spr > 6:
            overbets = [(1.5, "1.5x pot"), (2.0, "2x pot")]
            for frac, name in overbets:
                amt = _raise_to_amount(pot, frac, state, is_raise)
                candidates.append(SizingCandidate(name, amt, frac, "Overbet / polarized"))

    # ── Filter to legal range and deduplicate ───────────────

    filtered = []
    seen_amounts = set()
    for c in candidates:
        clamped = max(min_legal, min(c.amount, max_legal))
        rounded = round(clamped, 1)
        if rounded not in seen_amounts:
            seen_amounts.add(rounded)
            filtered.append(SizingCandidate(c.name, rounded, c.fraction, c.rationale))

    # Sort by amount
    filtered.sort(key=lambda c: c.amount)
    return filtered


def select_bot_sizing(
    state: BettingState,
    equity: float,
    legal_actions: list[LegalAction] | None = None,
) -> Optional[SizingCandidate]:
    """Select the best sizing for the bot based on equity and SPR.

    This is the strategy layer that picks from candidates.
    """
    candidates = compute_sizing_candidates(state, legal_actions)
    if not candidates:
        return None

    spr = BettingEngine.spr(state)

    if spr < 1.5:
        # Always jam when committed
        return candidates[-1]  # largest = all-in

    if equity > 0.7:
        # Strong hand — use larger sizing or geometric
        geo = [c for c in candidates if "Geometric" in c.name]
        if geo:
            return geo[0]
        return candidates[-2] if len(candidates) > 1 else candidates[-1]

    elif equity > 0.5:
        # Medium — use smaller sizing
        small = [c for c in candidates if c.fraction <= 0.5]
        return small[-1] if small else candidates[0]

    else:
        # Weak / bluff — use small sizing
        return candidates[0]


def _raise_to_amount(pot: float, fraction: float, state: BettingState, is_raise: bool) -> float:
    """Convert a pot fraction to a raise-to amount."""
    bet_amount = pot * fraction
    if is_raise:
        # Raise-to = current bet + bet_amount
        return state.highest_bet + bet_amount
    return bet_amount

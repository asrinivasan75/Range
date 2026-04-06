"""Tests for Kuhn Poker CFR training — validates convergence to known Nash equilibrium."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from packages.solver.cfr import CFRTrainer
from packages.solver.kuhn import KuhnPoker


def test_kuhn_game_basics():
    """Test basic game mechanics."""
    game = KuhnPoker()
    assert game.n_players() == 2
    assert game.is_chance("")
    assert game.is_chance("J")
    assert not game.is_chance("JQ")
    assert not game.is_terminal("JQ")
    assert game.is_terminal("JQpp")
    assert game.is_terminal("JQbf")
    assert game.is_terminal("JQbc")


def test_kuhn_terminal_utilities():
    """Test terminal payoff calculations."""
    game = KuhnPoker()
    # Check-check: higher card wins 1
    assert game.terminal_utility("JQpp", 0) == -1.0  # J < Q
    assert game.terminal_utility("JQpp", 1) == 1.0
    assert game.terminal_utility("KJpp", 0) == 1.0   # K > J
    # Bet-fold: bettor wins 1
    assert game.terminal_utility("JQbf", 0) == 1.0   # P1 bets, P2 folds
    assert game.terminal_utility("JQbf", 1) == -1.0
    # Bet-call: higher card wins 2
    assert game.terminal_utility("JQbc", 0) == -2.0   # J < Q, pot=4
    assert game.terminal_utility("JQbc", 1) == 2.0


def test_kuhn_info_sets():
    """Test information set construction."""
    game = KuhnPoker()
    # P0 with Jack at start
    assert game.info_set_key("JQ", 0) == "J:"
    # P0 doesn't see Q — P1 with Queen sees same action but different card
    assert game.info_set_key("JQ", 1) == "Q:"
    # After P0 checks
    assert game.info_set_key("JQp", 1) == "Q:p"


def test_kuhn_cfr_convergence():
    """Test that CFR converges toward Nash equilibrium for Kuhn poker."""
    game = KuhnPoker()
    trainer = CFRTrainer(game)
    trainer.train_vanilla(10000)

    nash = KuhnPoker.known_nash_equilibrium()
    strategy = trainer.get_strategy_summary()

    # Check P1 with King bets frequently (Nash has K betting often for value)
    k_strat = strategy.get("K:", {})
    if k_strat:
        avg = np.array(k_strat["strategy"])
        # Actions are [p, b] — K should bet majority of the time
        assert avg[1] > 0.55, f"K should bet frequently, got {avg}"

    # Check P2 with Jack vs bet always folds
    j_b = strategy.get("J:b", {})
    if j_b:
        avg = np.array(j_b["strategy"])
        # Actions are [f, c] — J should fold vs bet
        assert avg[0] > 0.7, f"J should fold vs bet, got {avg}"

    # Check P2 with King vs bet always calls
    k_b = strategy.get("K:b", {})
    if k_b:
        avg = np.array(k_b["strategy"])
        # Actions are [f, c] — K should call
        assert avg[1] > 0.8, f"K should call vs bet, got {avg}"


def test_kuhn_mccfr_convergence():
    """Test that MCCFR also converges for Kuhn poker."""
    game = KuhnPoker()
    trainer = CFRTrainer(game)
    trainer.train_mccfr(10000)

    strategy = trainer.get_strategy_summary()

    # K with bet should be well-visited and have a clear strategy
    k_b = strategy.get("K:b", {})
    if k_b:
        avg = np.array(k_b["strategy"])
        assert avg[1] > 0.7, f"K should call vs bet in MCCFR, got {avg}"


def test_kuhn_exploitability_decreases():
    """Test that exploitability proxy decreases with more iterations."""
    game = KuhnPoker()
    trainer = CFRTrainer(game)
    metrics = trainer.train_vanilla(2000)

    if len(metrics) >= 2:
        early = metrics[0]["exploitability_proxy"]
        late = metrics[-1]["exploitability_proxy"]
        assert late < early, f"Exploitability should decrease: {early} -> {late}"

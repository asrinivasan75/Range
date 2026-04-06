"""Counterfactual Regret Minimization — vanilla CFR and external-sampling MCCFR.

This module provides the core CFR algorithm used across all game implementations.
Each game must implement the GameInterface protocol.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
import time


class GameInterface(Protocol):
    """Protocol that games must implement for CFR training."""

    def is_terminal(self, history: str) -> bool: ...
    def terminal_utility(self, history: str, player: int) -> float: ...
    def is_chance(self, history: str) -> bool: ...
    def chance_actions(self, history: str) -> list[tuple[str, float]]: ...
    def current_player(self, history: str) -> int: ...
    def info_set_key(self, history: str, player: int) -> str: ...
    def actions(self, history: str) -> list[str]: ...
    def n_players(self) -> int: ...


@dataclass
class InfoSetData:
    """Stores regrets and strategy sums for an information set."""
    n_actions: int
    cumulative_regrets: np.ndarray = field(init=False)
    strategy_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cumulative_regrets = np.zeros(self.n_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(self.n_actions, dtype=np.float64)

    def current_strategy(self) -> np.ndarray:
        """Compute current strategy via regret matching."""
        positive_regrets = np.maximum(self.cumulative_regrets, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.ones(self.n_actions) / self.n_actions

    def average_strategy(self) -> np.ndarray:
        """Compute the average strategy (the convergent Nash approximation)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.n_actions) / self.n_actions


class CFRTrainer:
    """Vanilla CFR and External-Sampling MCCFR trainer."""

    def __init__(self, game: GameInterface) -> None:
        self.game = game
        self.info_sets: dict[str, InfoSetData] = {}
        self.iterations_done: int = 0
        self._metrics_history: list[dict] = []

    def _get_or_create_info_set(self, key: str, n_actions: int) -> InfoSetData:
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetData(n_actions)
        return self.info_sets[key]

    # ── Vanilla CFR ──────────────────────────────────────────

    def cfr(self, history: str, reach_probs: np.ndarray) -> np.ndarray:
        """Run one pass of vanilla CFR. Returns utility for each player."""
        n = self.game.n_players()

        if self.game.is_terminal(history):
            return np.array([self.game.terminal_utility(history, p) for p in range(n)])

        if self.game.is_chance(history):
            util = np.zeros(n)
            for action, prob in self.game.chance_actions(history):
                util += prob * self.cfr(history + action, reach_probs * prob)
            return util

        player = self.game.current_player(history)
        actions = self.game.actions(history)
        info_key = self.game.info_set_key(history, player)
        info_set = self._get_or_create_info_set(info_key, len(actions))

        strategy = info_set.current_strategy()
        action_utilities = np.zeros((len(actions), n))

        for i, action in enumerate(actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_utilities[i] = self.cfr(history + action, new_reach)

        node_utility = np.zeros(n)
        for i in range(len(actions)):
            node_utility += strategy[i] * action_utilities[i]

        # Update regrets
        opp_reach = np.prod(reach_probs[:player]) * np.prod(reach_probs[player + 1:])
        for i in range(len(actions)):
            regret = action_utilities[i][player] - node_utility[player]
            info_set.cumulative_regrets[i] += opp_reach * regret

        # Accumulate strategy
        info_set.strategy_sum += reach_probs[player] * strategy

        return node_utility

    # ── External Sampling MCCFR ──────────────────────────────

    def external_sampling_mccfr(self, history: str, update_player: int) -> float:
        """One traversal of external-sampling MCCFR. Returns utility for update_player."""
        if self.game.is_terminal(history):
            return self.game.terminal_utility(history, update_player)

        if self.game.is_chance(history):
            chance_actions = self.game.chance_actions(history)
            # Sample one chance outcome
            probs = [p for _, p in chance_actions]
            idx = np.random.choice(len(chance_actions), p=probs)
            action, _ = chance_actions[idx]
            return self.external_sampling_mccfr(history + action, update_player)

        player = self.game.current_player(history)
        actions = self.game.actions(history)
        info_key = self.game.info_set_key(history, player)
        info_set = self._get_or_create_info_set(info_key, len(actions))

        strategy = info_set.current_strategy()

        if player == update_player:
            # Traverse all actions
            utilities = np.zeros(len(actions))
            for i, action in enumerate(actions):
                utilities[i] = self.external_sampling_mccfr(
                    history + action, update_player
                )

            node_utility = np.dot(strategy, utilities)

            # Update regrets
            for i in range(len(actions)):
                info_set.cumulative_regrets[i] += utilities[i] - node_utility

            # Accumulate strategy
            info_set.strategy_sum += strategy

            return node_utility
        else:
            # Sample according to opponent strategy
            action_idx = np.random.choice(len(actions), p=strategy)
            return self.external_sampling_mccfr(
                history + actions[action_idx], update_player
            )

    # ── Training loops ───────────────────────────────────────

    def train_vanilla(
        self,
        n_iterations: int,
        callback: callable | None = None,
    ) -> list[dict]:
        """Run vanilla CFR for n_iterations."""
        n = self.game.n_players()
        metrics = []

        for i in range(n_iterations):
            t0 = time.time()
            reach_probs = np.ones(n)
            utilities = self.cfr("", reach_probs)
            elapsed = time.time() - t0
            self.iterations_done += 1

            if (i + 1) % max(1, n_iterations // 100) == 0 or i == n_iterations - 1:
                m = self._compute_metrics(elapsed, utilities)
                metrics.append(m)
                self._metrics_history.append(m)
                if callback:
                    callback(m)

        return metrics

    def train_mccfr(
        self,
        n_iterations: int,
        callback: callable | None = None,
    ) -> list[dict]:
        """Run external-sampling MCCFR for n_iterations."""
        n = self.game.n_players()
        metrics = []

        for i in range(n_iterations):
            t0 = time.time()
            for p in range(n):
                self.external_sampling_mccfr("", p)
            elapsed = time.time() - t0
            self.iterations_done += 1

            if (i + 1) % max(1, n_iterations // 100) == 0 or i == n_iterations - 1:
                m = self._compute_metrics(elapsed, np.zeros(n))
                metrics.append(m)
                self._metrics_history.append(m)
                if callback:
                    callback(m)

        return metrics

    def _compute_metrics(self, iteration_time: float, utilities: np.ndarray) -> dict:
        """Compute current training metrics."""
        total_regret = 0.0
        max_regret = 0.0
        for info_set in self.info_sets.values():
            abs_regret = np.sum(np.maximum(info_set.cumulative_regrets, 0))
            total_regret += abs_regret
            max_regret = max(max_regret, abs_regret)

        avg_regret = total_regret / max(len(self.info_sets), 1)
        exploitability_proxy = total_regret / max(self.iterations_done, 1)

        return {
            "iteration": self.iterations_done,
            "n_info_sets": len(self.info_sets),
            "total_regret": float(total_regret),
            "avg_regret": float(avg_regret),
            "max_regret": float(max_regret),
            "exploitability_proxy": float(exploitability_proxy),
            "iteration_time_ms": iteration_time * 1000,
            "utilities": utilities.tolist() if isinstance(utilities, np.ndarray) else utilities,
        }

    def get_strategy_summary(self) -> dict[str, dict[str, float]]:
        """Get the average strategy for each information set."""
        summary = {}
        for key, info_set in self.info_sets.items():
            avg = info_set.average_strategy()
            summary[key] = {
                "strategy": avg.tolist(),
                "regret": info_set.cumulative_regrets.tolist(),
                "visits": float(info_set.strategy_sum.sum()),
            }
        return summary

    @property
    def metrics_history(self) -> list[dict]:
        return self._metrics_history

"""Reinforcement Learning agent for Slumbot play.

Uses feature-based Q-learning with experience replay.
Learns from outcomes of hands played against Slumbot.

Features: equity, pot odds, SPR, street, position, bet sizing, hand strength.
Actions: fold, check, call, bet_small (1/3), bet_medium (2/3), bet_large (pot).

This is a SEPARATE model from the CFR solver — it learns exploitative play
against a specific opponent by observing their tendencies.
"""

from __future__ import annotations
import json
import random
import math
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from packages.poker.card import Card
from packages.poker.evaluator import hand_strength_monte_carlo, evaluate_hand
from packages.poker.hand import HandCategory


# ── Action space ─────────────────────────────────────────────

ACTIONS = ["fold", "check", "call", "bet_small", "bet_medium", "bet_large"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# ── Feature extraction ───────────────────────────────────────

N_FEATURES = 12


def extract_features(
    hole_cards: list[Card],
    board: list[Card],
    pot: int,
    to_call: int,
    our_invested: int,
    our_remaining: int,
    street: int,
    is_btn: bool,
    n_bets_this_street: int,
) -> np.ndarray:
    """Extract a feature vector from the current game state.

    Returns normalized features in [0, 1] range (mostly).
    """
    # Equity (0-1)
    equity = hand_strength_monte_carlo(hole_cards, board, n_simulations=100)

    # Pot odds (0-1): what fraction of the new pot we need to invest
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    # SPR: stack-to-pot ratio (capped at 20 for normalization)
    spr = min(our_remaining / max(pot, 1), 20.0) / 20.0

    # Street (0-1): 0=preflop, 0.33=flop, 0.67=turn, 1=river
    street_norm = street / 3.0

    # Position: 1=button, 0=BB
    position = 1.0 if is_btn else 0.0

    # Bet size relative to pot (0-2+, capped)
    bet_ratio = min(to_call / max(pot - to_call, 1), 2.0) / 2.0

    # Hand strength category (0-1)
    hand_strength = 0.0
    if len(board) >= 3:
        result = evaluate_hand(hole_cards, board)
        hand_strength = min(result[0].value / 9.0, 1.0)  # 0=high card, 1=royal flush

    # Whether we have at least a pair (binary)
    has_pair = 1.0 if hand_strength >= 1/9 else 0.0

    # Commitment level: how much of our stack is already in (0-1)
    total_stack = our_invested + our_remaining
    commitment = our_invested / max(total_stack, 1)

    # Number of bets/raises on this street (0-1, capped at 4)
    aggression = min(n_bets_this_street / 4.0, 1.0)

    # Equity vs pot odds advantage (can be negative)
    equity_advantage = equity - pot_odds

    # Combined strength signal (equity * hand_strength for later streets)
    combined_strength = equity * (0.5 + 0.5 * hand_strength) if street > 0 else equity

    return np.array([
        equity,           # 0
        pot_odds,         # 1
        spr,              # 2
        street_norm,      # 3
        position,         # 4
        bet_ratio,        # 5
        hand_strength,    # 6
        has_pair,         # 7
        commitment,       # 8
        aggression,       # 9
        equity_advantage, # 10
        combined_strength,# 11
    ], dtype=np.float64)


# ── Q-Learning Agent ─────────────────────────────────────────

@dataclass
class Experience:
    """A single experience tuple for replay."""
    features: np.ndarray
    action_idx: int
    reward: float  # normalized reward (bb won/lost)
    legal_mask: np.ndarray  # which actions were legal


class QLearningAgent:
    """Simple linear Q-learning agent with experience replay.

    Uses a weight matrix W (N_FEATURES x N_ACTIONS) to compute Q-values:
      Q(s, a) = features @ W[:, a]

    Simple but effective — linear models are fast and interpretable.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount: float = 0.95,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
    ):
        # Weight matrix: features -> action values
        self.W = np.zeros((N_FEATURES, N_ACTIONS), dtype=np.float64)
        # Initialize with reasonable priors
        self._init_weights()

        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer: list[Experience] = []
        self.max_buffer = 10000
        self.batch_size = 64

        self.hands_trained = 0
        self.total_reward = 0.0

    def _init_weights(self):
        """Set initial weights encoding basic poker knowledge."""
        # fold: prefer when equity is low, pot odds are bad
        self.W[0, 0] = -2.0   # low equity -> don't fold (neg because high equity = don't fold)
        self.W[1, 0] = 1.5    # high pot odds -> fold
        self.W[7, 0] = -1.0   # has pair -> don't fold
        self.W[10, 0] = -2.0  # equity advantage -> don't fold

        # check: neutral, safe play
        self.W[0, 1] = -0.5   # low equity -> check is fine

        # call: prefer when equity > pot odds
        self.W[0, 2] = 1.0    # high equity -> call
        self.W[10, 2] = 2.0   # equity advantage -> call
        self.W[7, 2] = 0.5    # has pair -> call

        # bet_small: good with medium hands
        self.W[0, 3] = 0.8
        self.W[11, 3] = 1.0   # combined strength

        # bet_medium: good with strong hands
        self.W[0, 4] = 1.2
        self.W[6, 4] = 1.5    # hand strength
        self.W[11, 4] = 1.5

        # bet_large: very strong hands or bluffs
        self.W[0, 5] = 1.5
        self.W[6, 5] = 2.0
        self.W[2, 5] = -0.5   # low SPR -> bet large (committed)

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        return features @ self.W

    def choose_action(
        self,
        features: np.ndarray,
        legal_mask: np.ndarray,
    ) -> int:
        """Epsilon-greedy action selection."""
        q_values = self.get_q_values(features)

        # Mask illegal actions with -inf
        masked_q = np.where(legal_mask > 0, q_values, -1e9)

        if random.random() < self.epsilon:
            # Explore: random legal action
            legal_indices = np.where(legal_mask > 0)[0]
            return int(np.random.choice(legal_indices))
        else:
            # Exploit: best Q-value
            return int(np.argmax(masked_q))

    def store_experience(self, exp: Experience):
        """Add experience to replay buffer."""
        self.replay_buffer.append(exp)
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)

    def train_batch(self):
        """Train on a random batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        for exp in batch:
            q_values = self.get_q_values(exp.features)
            target = exp.reward  # single-step (reward at end of hand)

            # Update only the chosen action
            error = target - q_values[exp.action_idx]
            gradient = exp.features * error * self.lr
            self.W[:, exp.action_idx] += gradient

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_hand(self, reward_bb: float, hand_experiences: list[Experience]):
        """Called at end of each hand with the reward and all experiences from that hand."""
        self.hands_trained += 1
        self.total_reward += reward_bb

        # Assign discounted rewards to each experience
        n = len(hand_experiences)
        for i, exp in enumerate(hand_experiences):
            # Discount: actions earlier in the hand get discounted reward
            discount_factor = self.discount ** (n - 1 - i)
            exp.reward = reward_bb * discount_factor / 100.0  # normalize to [-2, 2] range roughly
            self.store_experience(exp)

        # Train every hand
        self.train_batch()

    def save(self, path: str = "data/rl_agent.pkl"):
        """Save agent weights and state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "W": self.W,
            "epsilon": self.epsilon,
            "hands_trained": self.hands_trained,
            "total_reward": self.total_reward,
            "buffer_size": len(self.replay_buffer),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str = "data/rl_agent.pkl") -> bool:
        """Load agent weights. Returns True if successful."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.W = state["W"]
            self.epsilon = state["epsilon"]
            self.hands_trained = state["hands_trained"]
            self.total_reward = state["total_reward"]
            return True
        except (FileNotFoundError, EOFError):
            return False

    def get_stats(self) -> dict:
        return {
            "hands_trained": self.hands_trained,
            "avg_reward_bb": round(self.total_reward / max(self.hands_trained, 1), 2),
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.replay_buffer),
        }


# ── Action mapping for Slumbot ───────────────────────────────

def rl_action_to_slumbot(
    action_idx: int,
    pot: int,
    to_call: int,
    our_invested: int,
    our_remaining: int,
    use_call_for_check: bool = False,
) -> str:
    """Convert RL action index to Slumbot action string."""
    action = ACTIONS[action_idx]

    if action == "fold":
        return "f"
    if action == "check":
        return "c" if use_call_for_check else "k"
    if action == "call":
        return "c"

    # Bet/raise sizing
    if action == "bet_small":
        bet = int(pot * 0.33)
    elif action == "bet_medium":
        bet = int(pot * 0.67)
    else:  # bet_large
        bet = int(pot * 1.0)

    bet = max(bet, 100)  # min bet = 1bb
    bet = min(bet, our_remaining)
    total = our_invested + bet
    return f"b{total}"


def get_legal_mask(to_call: int, our_remaining: int) -> np.ndarray:
    """Compute which actions are legal."""
    mask = np.zeros(N_ACTIONS, dtype=np.float64)

    if to_call > 0:
        # Facing a bet: can fold, call, or raise
        mask[ACTION_TO_IDX["fold"]] = 1
        if our_remaining >= to_call:
            mask[ACTION_TO_IDX["call"]] = 1
        if our_remaining > to_call:
            mask[ACTION_TO_IDX["bet_small"]] = 1
            mask[ACTION_TO_IDX["bet_medium"]] = 1
            mask[ACTION_TO_IDX["bet_large"]] = 1
    else:
        # No bet: can check or bet
        mask[ACTION_TO_IDX["check"]] = 1
        if our_remaining > 0:
            mask[ACTION_TO_IDX["bet_small"]] = 1
            mask[ACTION_TO_IDX["bet_medium"]] = 1
            mask[ACTION_TO_IDX["bet_large"]] = 1

    return mask

"""PPO Neural Network agent for poker.

Uses Proximal Policy Optimization with:
- Actor network: outputs action probability distributions (mixed strategies)
- Critic network: estimates state value for variance reduction
- Shared feature encoder with separate actor/critic heads
- Rich feature extraction (30+ features including board texture, blockers)
- Experience buffer with GAE (Generalized Advantage Estimation)

This fixes the core weakness of the linear Q-learner:
the agent LEARNS when to randomize, outputting proper mixed strategies.
"""

from __future__ import annotations
import math
import random
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from packages.poker.card import Card, Rank, Suit
from packages.poker.evaluator import hand_strength_monte_carlo, evaluate_hand, evaluate_five
from packages.poker.hand import HandCategory


# ── Action space ─────────────────────────────────────────────

ACTIONS = ["fold", "check", "call", "bet_small", "bet_medium", "bet_large", "all_in"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# ── Feature extraction (30 features) ────────────────────────

N_FEATURES = 30


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
    n_raises_total: int = 0,
    range_adjusted_equity: float | None = None,
) -> np.ndarray:
    """Extract a rich feature vector from the current game state.

    30 features capturing equity, pot geometry, board texture,
    hand category, position, and betting dynamics.
    If range_adjusted_equity is provided, uses it instead of raw equity.
    """
    total_stack = our_invested + our_remaining
    eff_stack = our_remaining

    # ── Core poker features ──────────────────────────────────
    # 0: Equity — range-adjusted if available, otherwise vs random
    if range_adjusted_equity is not None:
        equity = range_adjusted_equity
    else:
        equity = hand_strength_monte_carlo(hole_cards, board, n_simulations=30)

    # 1: Pot odds (0-1)
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    # 2: Equity advantage over pot odds (can be negative)
    equity_edge = equity - pot_odds

    # 3: SPR (stack-to-pot ratio, capped and normalized)
    spr = min(eff_stack / max(pot, 1), 20.0) / 20.0

    # 4: Street (0-1 normalized)
    street_norm = street / 3.0

    # 5: Position (1=IP/button, 0=OOP/BB)
    position = 1.0 if is_btn else 0.0

    # 6: Bet ratio facing (opponent bet / pot, capped)
    bet_ratio = min(to_call / max(pot - to_call, 1), 3.0) / 3.0

    # 7: Commitment (fraction of starting stack invested)
    commitment = our_invested / max(total_stack, 1)

    # 8: Aggression on current street (normalized)
    street_aggression = min(n_bets_this_street / 4.0, 1.0)

    # 9: Total aggression across hand
    total_aggression = min(n_raises_total / 8.0, 1.0)

    # ── Hand strength features ───────────────────────────────
    hand_cat = 0.0
    has_pair = 0.0
    has_two_pair_plus = 0.0
    has_strong_hand = 0.0  # trips or better
    has_top_pair = 0.0

    if len(board) >= 3:
        result = evaluate_hand(hole_cards, board)
        cat_val = result[0].value
        hand_cat = cat_val / 9.0
        has_pair = 1.0 if cat_val >= 1 else 0.0
        has_two_pair_plus = 1.0 if cat_val >= 2 else 0.0
        has_strong_hand = 1.0 if cat_val >= 3 else 0.0

        # Top pair detection: does one of our cards match the highest board card?
        board_ranks = sorted([c.rank.value for c in board], reverse=True)
        our_ranks = [c.rank.value for c in hole_cards]
        if board_ranks and any(r == board_ranks[0] for r in our_ranks):
            has_top_pair = 1.0 if cat_val >= 1 else 0.0

    # Preflop hand strength (always available)
    # 15: Pocket pair
    is_pocket_pair = 1.0 if hole_cards[0].rank == hole_cards[1].rank else 0.0
    # 16: Suited
    is_suited = 1.0 if hole_cards[0].suit == hole_cards[1].suit else 0.0
    # 17: High card rank (A=1, 2=0, normalized)
    high_rank = max(hole_cards[0].rank.value, hole_cards[1].rank.value)
    high_rank_norm = (high_rank - 2) / 12.0
    # 18: Card gap (connectedness)
    gap = abs(hole_cards[0].rank.value - hole_cards[1].rank.value)
    connectedness = max(1.0 - gap / 12.0, 0.0)

    # ── Board texture features (postflop only) ──────────────
    board_wet = 0.0  # flush/straight potential
    board_paired = 0.0
    board_high = 0.0
    n_overcards = 0.0
    flush_possible = 0.0
    straight_possible = 0.0

    if len(board) >= 3:
        board_ranks = [c.rank.value for c in board]
        board_suits = [c.suit.value for c in board]

        # Board highness
        board_high = max(board_ranks) / 14.0

        # Board paired
        from collections import Counter
        rank_counts = Counter(board_ranks)
        board_paired = 1.0 if max(rank_counts.values()) >= 2 else 0.0

        # Flush possible (3+ of same suit on board)
        suit_counts = Counter(board_suits)
        flush_possible = 1.0 if max(suit_counts.values()) >= 3 else 0.0

        # Straight possible (3+ cards within 5 ranks)
        sorted_ranks = sorted(set(board_ranks))
        for i in range(len(sorted_ranks)):
            window = [r for r in sorted_ranks if sorted_ranks[i] <= r <= sorted_ranks[i] + 4]
            if len(window) >= 3:
                straight_possible = 1.0
                break

        board_wet = (flush_possible + straight_possible) / 2.0

        # Overcards to our hand
        our_high = max(hole_cards[0].rank.value, hole_cards[1].rank.value)
        n_overcards = sum(1 for r in board_ranks if r > our_high) / len(board_ranks)

    # ── Blocker features ─────────────────────────────────────
    # Do we block strong hands the opponent could have?
    has_ace = 1.0 if any(c.rank == Rank.ACE for c in hole_cards) else 0.0
    has_king = 1.0 if any(c.rank == Rank.KING for c in hole_cards) else 0.0

    # Flush blocker (we hold a card of the most common board suit)
    flush_blocker = 0.0
    if len(board) >= 3:
        suit_counts = Counter(c.suit.value for c in board)
        most_common_suit = suit_counts.most_common(1)[0][0]
        if any(c.suit.value == most_common_suit for c in hole_cards):
            flush_blocker = 1.0

    # ── Composite signals ────────────────────────────────────
    # Combined strength (equity weighted by hand category)
    combined_strength = equity * (0.5 + 0.5 * hand_cat) if street > 0 else equity

    # Bluff candidate (low equity but has blockers)
    bluff_signal = (1.0 - equity) * (has_ace * 0.5 + flush_blocker * 0.5)

    return np.array([
        equity,             # 0
        pot_odds,           # 1
        equity_edge,        # 2
        spr,                # 3
        street_norm,        # 4
        position,           # 5
        bet_ratio,          # 6
        commitment,         # 7
        street_aggression,  # 8
        total_aggression,   # 9
        hand_cat,           # 10
        has_pair,           # 11
        has_two_pair_plus,  # 12
        has_strong_hand,    # 13
        has_top_pair,       # 14
        is_pocket_pair,     # 15
        is_suited,          # 16
        high_rank_norm,     # 17
        connectedness,      # 18
        board_wet,          # 19
        board_paired,       # 20
        board_high,         # 21
        n_overcards,        # 22
        flush_possible,     # 23
        straight_possible,  # 24
        has_ace,            # 25
        flush_blocker,      # 26
        combined_strength,  # 27
        bluff_signal,       # 28
        has_king,           # 29
    ], dtype=np.float32)


# ── Neural Network Architecture ─────────────────────────────

class PokerNetwork(nn.Module):
    """Shared encoder with actor (policy) and critic (value) heads.

    Architecture:
      Input (30) -> Encoder (256 -> 256 -> 128, residual) ->
        Actor head (128 -> 64 -> 7 actions, softmax)
        Critic head (128 -> 64 -> 1 value)
    """

    def __init__(self, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS):
        super().__init__()

        # Shared feature encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.res_block1 = ResidualBlock(256, 256)
        self.res_block2 = ResidualBlock(256, 128)

        # Actor head (policy — outputs action probabilities)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_actions),
        )

        # Critic head (value — estimates expected return)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Initialize with small weights for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor output layer: smaller init for more uniform initial policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic output: standard init
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        """Forward pass. Returns (action_logits, state_value).

        Args:
            x: feature tensor (batch_size, n_features)
            legal_mask: boolean mask (batch_size, n_actions), True = legal
        """
        h = self.encoder(x)
        h = self.res_block1(h)
        h = self.res_block2(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        # Mask illegal actions with large negative value
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e8)

        return logits, value

    def get_policy(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        """Get action probabilities."""
        logits, value = self.forward(x, legal_mask)
        probs = F.softmax(logits, dim=-1)
        return probs, value


class ResidualBlock(nn.Module):
    """Residual block with optional dimension change."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + self.shortcut(x))


# ── PPO Agent ────────────────────────────────────────────────

@dataclass
class Transition:
    """Single step of experience."""
    features: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float
    legal_mask: np.ndarray


class PPOAgent:
    """Proximal Policy Optimization agent for poker.

    Key improvements over linear Q-learning:
    - Outputs probability distributions (mixed strategies)
    - Neural network captures complex card interactions
    - PPO clipping prevents catastrophic policy updates
    - GAE for variance-reduced advantage estimation
    - Entropy bonus encourages exploration / mixed strategies
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        buffer_size: int = 1024,
    ):
        self.device = torch.device("cpu")  # CPU is fine for this scale
        self.network = PokerNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.buffer: list[Transition] = []
        self.hands_trained: int = 0
        self.total_reward: float = 0.0
        self.updates_done: int = 0

    @torch.no_grad()
    def choose_action(
        self,
        features: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[int, float, float]:
        """Choose action from the policy. Returns (action_idx, log_prob, value)."""
        self.network.eval()
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        mask = torch.BoolTensor(legal_mask.astype(bool)).unsqueeze(0).to(self.device)

        probs, value = self.network.get_policy(x, mask)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_action_probs(
        self,
        features: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """Get full action probability distribution (for display/logging)."""
        self.network.eval()
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        mask = torch.BoolTensor(legal_mask.astype(bool)).unsqueeze(0).to(self.device)
        probs, _ = self.network.get_policy(x, mask)
        return probs.squeeze(0).cpu().numpy()

    def store_transition(self, t: Transition):
        self.buffer.append(t)

    def end_hand(self, reward_bb: float, hand_transitions: list[Transition]):
        """Called at end of each hand. Assigns rewards and triggers PPO update if buffer full."""
        self.hands_trained += 1
        self.total_reward += reward_bb

        n = len(hand_transitions)
        for i, t in enumerate(hand_transitions):
            # Discount: later actions get full reward, earlier ones discounted
            discount = self.gamma ** (n - 1 - i)
            t.reward = reward_bb * discount / 100.0  # normalize
            self.buffer.append(t)

        # Update when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._ppo_update()
            self.buffer = []

    def _ppo_update(self):
        """Run PPO update on the buffer."""
        if len(self.buffer) < self.batch_size:
            return

        self.network.train()

        # Prepare tensors
        features = torch.FloatTensor(np.array([t.features for t in self.buffer])).to(self.device)
        actions = torch.LongTensor([t.action_idx for t in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t.log_prob for t in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in self.buffer]).to(self.device)
        old_values = torch.FloatTensor([t.value for t in self.buffer]).to(self.device)
        masks = torch.BoolTensor(np.array([t.legal_mask.astype(bool) for t in self.buffer])).to(self.device)

        # Compute advantages with GAE
        advantages = self._compute_gae(rewards, old_values)
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        n = len(self.buffer)
        for _ in range(self.epochs_per_update):
            # Mini-batch training
            indices = list(range(n))
            random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                b_features = features[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_masks = masks[batch_idx]

                # Forward pass
                logits, values = self.network(b_features, b_masks)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, b_returns)

                # Entropy bonus (encourages mixed strategies)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.scheduler.step()
        self.updates_done += 1

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = torch.zeros(n)
        last_gae = 0

        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * last_gae

        return advantages

    def save(self, path: str = "data/ppo_agent.pkl"):
        """Save full agent state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hands_trained": self.hands_trained,
            "total_reward": self.total_reward,
            "updates_done": self.updates_done,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str = "data/ppo_agent.pkl") -> bool:
        """Load agent state."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.network.load_state_dict(state["network_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.hands_trained = state["hands_trained"]
            self.total_reward = state["total_reward"]
            self.updates_done = state.get("updates_done", 0)
            return True
        except (FileNotFoundError, EOFError, KeyError):
            return False

    def get_stats(self) -> dict:
        return {
            "hands_trained": self.hands_trained,
            "avg_reward_bb": round(self.total_reward / max(self.hands_trained, 1), 2),
            "updates_done": self.updates_done,
            "buffer_size": len(self.buffer),
            "lr": self.optimizer.param_groups[0]["lr"],
        }


# ── Action mapping for Slumbot ───────────────────────────────

def ppo_action_to_slumbot(
    action_idx: int,
    pot: int,
    to_call: int,
    our_invested: int,
    our_remaining: int,
    use_call_for_check: bool = False,
) -> str:
    """Convert PPO action index to Slumbot action string."""
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
    elif action == "bet_large":
        bet = int(pot * 1.0)
    elif action == "all_in":
        bet = our_remaining
    else:
        bet = int(pot * 0.5)

    bet = max(bet, 100)  # min bet = 1bb
    bet = min(bet, our_remaining)
    total = our_invested + bet
    return f"b{total}"


def get_legal_mask(to_call: int, our_remaining: int) -> np.ndarray:
    """Compute which actions are legal."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)

    if to_call > 0:
        mask[ACTION_TO_IDX["fold"]] = 1
        if our_remaining >= to_call:
            mask[ACTION_TO_IDX["call"]] = 1
        if our_remaining > to_call:
            mask[ACTION_TO_IDX["bet_small"]] = 1
            mask[ACTION_TO_IDX["bet_medium"]] = 1
            mask[ACTION_TO_IDX["bet_large"]] = 1
        if our_remaining > 0:
            mask[ACTION_TO_IDX["all_in"]] = 1
    else:
        mask[ACTION_TO_IDX["check"]] = 1
        if our_remaining > 0:
            mask[ACTION_TO_IDX["bet_small"]] = 1
            mask[ACTION_TO_IDX["bet_medium"]] = 1
            mask[ACTION_TO_IDX["bet_large"]] = 1
            mask[ACTION_TO_IDX["all_in"]] = 1

    return mask

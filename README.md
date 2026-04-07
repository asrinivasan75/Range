# Range

**Poker solver and analysis platform for No-Limit Texas Hold'em.**

Range combines game-theoretic solvers (CFR/MCCFR), reinforcement learning (Q-learning + PPO neural networks), population-based training, and opponent range estimation to build poker agents that compete against world-class bots. It includes a premium frontend with a cinematic landing page, training dashboard, interactive Play vs Bot mode, and hand replayer.

## Slumbot Leaderboard

Our Q-learning agent **ranks #3 worldwide** on the [Slumbot](https://slumbot.com) leaderboard (min. 5,000 hands):

| Metric | Score | Rank |
|--------|-------|------|
| **Earnings** | +96.34 BB/100 | **#3** |
| **Baseline** (luck-adjusted) | +38.62 BB/100 | **#5** |

Slumbot is a near-Nash HUNL poker bot that won the 2018 Annual Computer Poker Competition. Our agent beats it using a lightweight Q-learning model (72 parameters) trained via Population-Based Training against Slumbot's API.

---

## Architecture

```
Range/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   └── routes/
│   │       ├── play.py         # Play vs Bot (BettingEngine + RL agents)
│   │       ├── runs.py         # Training run CRUD
│   │       ├── analysis.py     # Equity calculator, preflop chart
│   │       └── training.py     # Training control
│   └── web/                    # Next.js frontend
│       └── src/app/dashboard/
│           ├── play/           # Interactive Play vs Bot
│           ├── replayer/       # Hand history replayer
│           ├── runs/           # Training runs + metrics charts
│           ├── explorer/       # Equity calculator + preflop chart
│           ├── strategy/       # Strategy browser
│           └── architecture/   # How It Works flowcharts
├── packages/
│   ├── poker/
│   │   ├── betting_engine.py   # Immutable NLHE state machine (correct rules)
│   │   ├── sizing.py           # SPR-aware geometric sizing
│   │   ├── evaluator.py        # 5/7-card hand evaluation + Monte Carlo equity
│   │   ├── card.py             # Card, Rank, Suit primitives
│   │   ├── deck.py             # Deck management
│   │   ├── hand.py             # Hand representation
│   │   ├── actions.py          # Action types
│   │   └── game_state.py       # Game state (legacy)
│   └── solver/
│       ├── cfr.py              # Vanilla CFR + External-Sampling MCCFR
│       ├── rl_agent.py         # Q-Learning agent (12 features, 72 params)
│       ├── neural_agent.py     # PPO neural network (30 features, ~100K params)
│       ├── range_estimator.py  # Opponent range estimation (GTO priors)
│       ├── self_play.py        # Local self-play training
│       ├── slumbot.py          # Slumbot API integration + benchmarking
│       ├── pbt.py              # Population-Based Training
│       ├── train_from_data.py  # Train from real hand history data (PHH)
│       ├── holdem_full.py      # 4-street Hold'em for CFR
│       ├── holdem_simplified.py# 2-street Hold'em for CFR
│       ├── kuhn.py             # Kuhn Poker (CFR validation)
│       ├── abstractions.py     # Card/action abstraction
│       ├── metrics.py          # Training metrics
│       └── trainer.py          # Training orchestrator
├── tests/                      # 61 tests (evaluator, CFR, betting engine)
├── scripts/                    # Setup, run, train, seed, slumbot
└── data/                       # Weights, logs, checkpoints, hand histories
```

## AI Systems

### 1. CFR Solver (Game-Theoretic)
Counterfactual Regret Minimization finds Nash equilibrium strategies through self-play iteration. Implemented for Kuhn Poker (validation), Simplified Hold'em (2 streets), and Full Hold'em (4 streets). Produces balanced strategies that can't be exploited.

### 2. Q-Learning Agent (Reinforcement Learning)
Linear function approximation with 12 features and 72 parameters. Trained against Slumbot via Population-Based Training. Learns exploitative strategies — finds specific weaknesses in the opponent. Currently ranks #3 on Slumbot leaderboard.

**Features:** equity, pot odds, SPR, street, position, bet ratio, hand category, has pair, commitment, aggression, equity advantage, combined strength.

**Learning:** Online stochastic gradient descent on `(reward - Q_prediction)²`. After each hand, the weights shift so that profitable actions become more likely in similar states.

### 3. PPO Neural Network (Deep RL)
Actor-critic architecture with ~100K parameters. 30 features including board texture, blockers, flush/straight potential, overcards, and bluff signals. Outputs probability distributions over actions (mixed strategies) instead of deterministic choices.

**Architecture:** 30 → 256 → 256 → 128 (shared encoder with residual blocks) → Actor head (7 action probabilities) + Critic head (state value).

**Training:** Self-play against itself + Q-learning opponents + random agents. 500K+ hands at 28-40 hands/sec locally. PPO clipping prevents catastrophic weight updates. Entropy bonus maintains mixed strategies.

### 4. Population-Based Training
Runs multiple agents in parallel against Slumbot. Every 2,000 hands, evaluates performance. Bottom performers get replaced with perturbed clones of the top 2 agents. Adaptive threshold decays from 60 → 25 bb/100 over training. Weight snapshots saved at every replacement for rollback.

### 5. Opponent Range Estimation
Instead of computing equity against a random hand, estimates what the opponent likely holds based on their action sequence. Maps action patterns to GTO-based hand range distributions (e.g., "5-bet shove" → AA/KK/QQ 65%). Computes equity against the estimated range.

**Impact:** Q6s vs random = 52% equity (calls all-in). Q6s vs 5-bet range = 32% equity (correctly folds).

### 6. Betting Engine
Immutable state machine with correct NLHE rules:
- No raise cap (unlimited re-raises)
- Correct min-raise tracking (last raise increment)
- Action reopening (short all-in doesn't reopen)
- Stack tracking and all-in detection
- Heads-up position (button first preflop, BB first postflop)
- SPR-aware geometric sizing

40 unit tests covering legal actions, min raise, street completion, position, stacks, and immutability.

## Stack

| Layer | Technology |
|-------|-----------|
| **Solver/RL** | Python, NumPy, PyTorch, Numba |
| **API** | FastAPI, Pydantic, aiosqlite |
| **Frontend** | Next.js 14, TypeScript, Tailwind, Framer Motion, Recharts |
| **Data** | SQLite, JSON, pickle (weights) |

## Quick Start

```bash
# Setup
./scripts/setup.sh

# Generate demo training data
./scripts/seed.sh

# Start API (terminal 1)
./scripts/api.sh

# Start frontend (terminal 2)
./scripts/web.sh

# Open http://localhost:3000
```

## Training Commands

```bash
# CFR training
./scripts/train.sh simplified_holdem 10000 mccfr
./scripts/train.sh kuhn 5000 cfr

# Slumbot benchmark (register at slumbot.com first)
./scripts/slumbot.sh <username> <password> 1000
./scripts/slumbot.sh <username> <password> 1000 --rl  # with RL agent

# Self-play PPO training (local, no API needed)
PYTHONPATH=. python -c "from packages.solver.self_play import train_self_play; train_self_play(n_hands=50000)"

# Population-Based Training
PYTHONPATH=. python packages/solver/pbt.py 50000

# Train from real hand data (PHH dataset)
PYTHONPATH=. python packages/solver/train_from_data.py 100000
```

## Frontend Pages

| Page | Description |
|------|-------------|
| **Landing** | Cinematic hero, leaderboard, game theory explainer, architecture, features |
| **Play vs Bot** | Interactive heads-up poker with bot selection (Q-learning, PPO, heuristic), raise slider, session logging |
| **Hand Replayer** | Review Slumbot sessions and play sessions with visual card playback, profit curves, filtering |
| **Training Runs** | Create/monitor CFR training, convergence charts, strategy explorer |
| **Explorer** | Equity calculator, preflop hand strength chart |
| **How It Works** | Interactive flowcharts of Q-learning, PPO, range estimation, reward calculation |

## Testing

```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v  # 61 tests
```

## Key Results

| System | Opponent | Hands | Result |
|--------|----------|-------|--------|
| Q-Learning (PBT) | Slumbot | 5,000+ | **+96.34 BB/100** (#3 leaderboard) |
| Q-Learning (frozen) | Slumbot | 1,000 | +26.7 BB/100 |
| PPO self-play | Self + Q-learning pool | 500,000 | +0.67 bb/hand (last 1K) |
| PPO real data | 1.17M human hands (PHH) | 1,168,726 | 190 updates |

## How Learning Works

**Q-Learning (72 parameters):**
```
features = [equity, pot_odds, spr, street, position, ...]  # 12 numbers
Q_values = features @ W                                      # linear dot product
action = argmax(Q_values)                                     # pick best
error = reward - Q_values[action]                             # how wrong?
W[:, action] += lr × error × features                        # gradient descent
```

**PPO Neural Network (~100K parameters):**
```
features → Encoder(256→256→128) → Actor(probabilities) + Critic(value)
action = sample(probabilities)                # mixed strategy
advantage = reward - critic_prediction        # better or worse than expected?
actor_loss = -min(ratio × adv, clip(ratio) × adv)  # PPO clipping
Update network to increase probability of high-advantage actions
```

**Range Estimation:**
```
opponent_actions → classify("5-bet shove") → GTO range(AA 16%, KK 16%, ...)
equity_vs_range(Q6s, {AA,KK,QQ}) = 32%    # vs 52% against random
Bot correctly folds instead of calling
```

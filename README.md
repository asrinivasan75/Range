# Range

**Poker solver and analysis platform for No-Limit Texas Hold'em.**

Range trains near-optimal poker strategies through Counterfactual Regret Minimization (CFR/MCCFR), with a premium frontend for exploring training results, inspecting strategies, and analyzing hand equity.

---

## Architecture

```
Range/
├── apps/
│   ├── api/            # FastAPI backend — REST endpoints, training control
│   └── web/            # Next.js frontend — landing page, dashboard, explorer
├── packages/
│   ├── poker/          # Domain modeling — cards, hands, evaluation, game state
│   └── solver/         # Training core — CFR, MCCFR, abstractions, orchestrator
├── tests/              # pytest test suite
├── scripts/            # Setup, run, train, seed scripts
├── data/               # SQLite DB, training runs, checkpoints
└── docs/               # Additional documentation
```

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Solver** | Python + NumPy | Fastest path to working trainer. NumPy for vectorized operations. Clear seams for future Rust/C++ acceleration. |
| **API** | FastAPI | Async, auto-docs, Pydantic validation. Production-grade with minimal boilerplate. |
| **Frontend** | Next.js + TypeScript + Tailwind + Framer Motion | Modern React with excellent DX. Tailwind for rapid premium UI. Framer Motion for cinematic animations. |
| **Charts** | Recharts | Clean composable chart library that works well with React. |
| **Data** | SQLite + JSON files | Simple, zero-config. Schema designed for easy Postgres migration. |

**Why Python for the solver?** Python gives the fastest path to a working, correct CFR implementation. The algorithm is straightforward to implement and debug. Performance-critical inner loops can later be accelerated with Numba (already a dependency) or replaced with Rust/C++ modules at the function level — the `GameInterface` protocol makes this a clean swap.

## Training System

Range implements a two-level training system:

### 1. Kuhn Poker (Foundation)
- 3-card game (J, Q, K), 2 players, 12 information sets
- Known Nash equilibrium for validation
- Trains in < 5 seconds
- Used to verify CFR correctness

### 2. Simplified Hold'em (Primary)
- Heads-up No-Limit Hold'em with preflop + flop
- Coarse card abstraction: 8 preflop buckets × 8 flop buckets (via Chen formula + equity estimation)
- Limited action space: fold, check/call, bet half-pot, bet pot (max 2 raises/street)
- ~2,000–5,000 information sets
- Trains in 1–3 minutes on a laptop
- Produces meaningful strategy outputs, EV estimates, and convergence metrics

**Documented simplifications:** Full NLHE has ~10^160 game states. This simplified version reduces that to ~5K info sets through hand bucketing and action abstraction. The architecture is designed so that increasing granularity (more buckets, more streets, more bet sizes) is a configuration change, not a rewrite.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- pnpm or npm

### Setup
```bash
./scripts/setup.sh
```

### Generate Demo Data
```bash
./scripts/seed.sh
```
This trains Kuhn Poker and Simplified Hold'em, producing data for the dashboard.

### Start the API
```bash
./scripts/api.sh
# → http://localhost:8000 (API docs at /docs)
```

### Start the Frontend
```bash
./scripts/web.sh
# → http://localhost:3000
```

### Train from CLI
```bash
# Simplified Hold'em, 10K iterations, MCCFR
./scripts/train.sh simplified_holdem 10000 mccfr

# Kuhn Poker, 5K iterations, vanilla CFR
./scripts/train.sh kuhn 5000 cfr
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/runs` | List all training runs |
| POST | `/api/runs` | Start a new training run |
| GET | `/api/runs/{id}` | Get run details |
| GET | `/api/runs/{id}/metrics` | Get training metrics |
| GET | `/api/runs/{id}/strategy` | Get strategy data |
| DELETE | `/api/runs/{id}` | Delete a run |
| GET | `/api/training/games` | List available game types |
| GET | `/api/training/{id}/progress` | Get training progress |
| POST | `/api/analysis/equity` | Compute hand equity (Monte Carlo) |
| POST | `/api/analysis/hand-strength` | Evaluate hand strength |
| GET | `/api/analysis/preflop-chart` | Get preflop bucket chart |

## Testing

```bash
# Activate venv
source .venv/bin/activate

# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test suites
PYTHONPATH=. pytest tests/solver/test_kuhn.py -v    # CFR convergence validation
PYTHONPATH=. pytest tests/poker/test_evaluator.py -v # Hand evaluation
```

## What's Real Now vs Future Work

### Implemented (v0.1)
- [x] Full poker domain model (cards, hands, evaluation, game state)
- [x] Vanilla CFR and External-Sampling MCCFR
- [x] Kuhn Poker with Nash equilibrium validation
- [x] Simplified Hold'em with preflop + flop abstraction
- [x] Training orchestrator with persistence and background execution
- [x] REST API with all CRUD + analysis endpoints
- [x] Premium landing page with cinematic scrollytelling
- [x] Dashboard with training runs, metrics charts, strategy explorer
- [x] Hand equity calculator with Monte Carlo simulation
- [x] Preflop hand strength chart
- [x] SQLite persistence layer

### Future Work (v0.2+)
- [ ] Turn + River streets for Hold'em
- [ ] Finer-grained abstractions (20+ buckets, k-means clustering)
- [ ] Multiple bet sizes per action node
- [ ] Discount CFR / Linear CFR weighting
- [ ] Real-time subgame solving
- [ ] Rust/C++ acceleration for CFR traversal inner loop
- [ ] WebSocket streaming for live training progress
- [ ] Multi-table tournament (MTT) ICM integration
- [ ] Opponent modeling / exploitation mode
- [ ] Export strategies to GTO trainer format
- [ ] Docker deployment configuration
- [ ] PostgreSQL migration for production

## Project Philosophy

Range prioritizes:
1. **Correctness first** — validated against known equilibria before scaling
2. **Clean extensibility** — modular game interface, clear abstraction boundaries
3. **Practical output** — real training that produces real strategies, not architecture docs
4. **Premium experience** — the frontend should make game theory feel exciting

The goal: a believable path from toy poker → full-scale NLHE solver, with a product experience worth using along the way.

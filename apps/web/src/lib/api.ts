const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchAPI<{ status: string }>("/api/health"),

  // Runs
  listRuns: () => fetchAPI<{ runs: Run[] }>("/api/runs"),
  getRun: (id: string) => fetchAPI<{ run: Run }>(`/api/runs/${id}`),
  createRun: (config: CreateRunConfig) =>
    fetchAPI<{ run: Run }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  deleteRun: (id: string) =>
    fetchAPI<{ status: string }>(`/api/runs/${id}`, { method: "DELETE" }),
  getRunMetrics: (id: string) =>
    fetchAPI<RunMetrics>(`/api/runs/${id}/metrics`),
  getRunStrategy: (id: string) =>
    fetchAPI<{ strategy: Record<string, StrategyEntry> }>(
      `/api/runs/${id}/strategy`
    ),

  // Training
  listGames: () => fetchAPI<{ games: GameType[] }>("/api/training/games"),
  getProgress: (runId: string) =>
    fetchAPI<RunProgress>(`/api/training/${runId}/progress`),
  cancelRun: (runId: string) =>
    fetchAPI<{ cancelled: boolean }>(`/api/training/${runId}/cancel`, {
      method: "POST",
    }),

  // Analysis
  computeEquity: (holeCards: string[], board: string[] = []) =>
    fetchAPI<EquityResult>("/api/analysis/equity", {
      method: "POST",
      body: JSON.stringify({ hole_cards: holeCards, board }),
    }),
  getPreflopChart: () =>
    fetchAPI<PreflopChart>("/api/analysis/preflop-chart"),

  // Play
  newHand: (runId: string) =>
    fetchAPI<NewHandResult>("/api/play/new-hand", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    }),
  getAdvice: (params: AdvisorParams) =>
    fetchAPI<AdvisorResult>("/api/play/advisor", {
      method: "POST",
      body: JSON.stringify(params),
    }),
  botAction: (params: BotActionParams) =>
    fetchAPI<BotActionResult>("/api/play/bot-action", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};

// Types
export interface Run {
  id: string;
  name: string;
  config: RunConfig;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  current_iteration: number;
  error: string | null;
}

export interface RunConfig {
  game_type: string;
  algorithm: string;
  n_iterations: number;
  checkpoint_interval: number;
  preflop_buckets: number;
  flop_buckets: number;
  max_raises: number;
}

export interface CreateRunConfig {
  name?: string;
  game_type?: string;
  algorithm?: string;
  n_iterations?: number;
  preflop_buckets?: number;
  flop_buckets?: number;
  max_raises?: number;
}

export interface RunMetrics {
  series: {
    iterations: number[];
    exploitability: number[];
    total_regret: number[];
    avg_regret: number[];
    n_info_sets: number[];
    iteration_time_ms: number[];
    timestamps: number[];
  };
  summary: {
    total_iterations: number;
    n_info_sets: number;
    final_exploitability: number;
    final_avg_regret: number;
    total_time_seconds: number;
    avg_iteration_ms: number;
    regret_reduction: number;
  };
}

export interface RunProgress {
  run_id: string;
  status: string;
  current_iteration: number;
  total_iterations: number;
  progress_pct: number;
  latest_metrics: Record<string, number> | null;
}

export interface StrategyEntry {
  strategy: number[];
  regret: number[];
  visits: number;
}

export interface GameType {
  id: string;
  name: string;
  description: string;
  complexity: string;
  recommended_iterations: number;
  estimated_time: string;
}

export interface EquityResult {
  equity: number;
  equity_pct: number;
  hole_cards: string[];
  board: string[];
  chen_score?: number;
  preflop_bucket?: number;
}

export interface PreflopChart {
  grid: { hand: string; bucket: number }[][];
  n_buckets: number;
  ranks: string[];
}

// Play types
export interface AdvisorParams {
  run_id: string;
  hole_cards: string[];
  board?: string[];
  action_history?: string[];
  position?: number;
}

export interface AdvisorResult {
  info_set_key: string;
  preflop_bucket: number;
  flop_bucket: number | null;
  equity: number;
  equity_pct: number;
  chen_score: number;
  action_labels: string[];
  action_codes: string[];
  strategy?: number[];
  recommended_action: string;
  recommended_code: string;
  action_probabilities: Record<string, number>;
  found: boolean;
  note?: string;
}

export interface NewHandResult {
  player_cards: string[];
  bot_cards: string[];
  board: {
    flop: string[];
    turn: string;
    river: string;
  };
  pot: number;
  player_stack: number;
  bot_stack: number;
}

export interface BotActionParams {
  run_id: string;
  bot_hole_cards: string[];
  board?: string[];
  preflop_actions?: string;
  flop_actions?: string;
  bot_position?: number;
}

export interface BotActionResult {
  action: string;
  action_label: string;
  info_set_key: string;
  strategy_probabilities: Record<string, number> | null;
}

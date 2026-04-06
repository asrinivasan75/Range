"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api, type Run, type CreateRunConfig, type GameType } from "@/lib/api";
import {
  formatNumber,
  formatTimestamp,
  statusBadgeClass,
} from "@/lib/utils";

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [games, setGames] = useState<GameType[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreate, setShowCreate] = useState(false);

  // Form state
  const [form, setForm] = useState<CreateRunConfig>({
    name: "",
    game_type: "simplified_holdem",
    algorithm: "mccfr",
    n_iterations: 10000,
    preflop_buckets: 8,
    flop_buckets: 8,
  });

  const refresh = useCallback(() => {
    api.listRuns().then((res) => {
      setRuns(res.runs);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    api.listGames().then((res) => setGames(res.games)).catch(() => {});
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleCreate = async () => {
    setCreating(true);
    try {
      await api.createRun(form);
      setShowCreate(false);
      refresh();
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Training Runs
          </h1>
          <p className="text-sm text-graphite-400 mt-1">
            Configure and monitor solver training
          </p>
        </div>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-4 py-2.5 rounded-lg bg-accent-blue text-white text-sm font-medium hover:bg-accent-blue/90 transition-colors"
        >
          + New Run
        </button>
      </div>

      {/* Create form */}
      {showCreate && (
        <div className="card border-accent-blue/20">
          <h3 className="text-sm font-semibold text-white mb-4">
            Configure Training Run
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Field label="Name (optional)">
              <input
                type="text"
                value={form.name || ""}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="my-training-run"
                className="input-field"
              />
            </Field>
            <Field label="Game Type">
              <select
                value={form.game_type}
                onChange={(e) => setForm({ ...form, game_type: e.target.value })}
                className="input-field"
              >
                <option value="kuhn">Kuhn Poker</option>
                <option value="simplified_holdem">Simplified Hold&apos;em</option>
              </select>
            </Field>
            <Field label="Algorithm">
              <select
                value={form.algorithm}
                onChange={(e) => setForm({ ...form, algorithm: e.target.value })}
                className="input-field"
              >
                <option value="mccfr">MCCFR (External Sampling)</option>
                <option value="cfr">Vanilla CFR</option>
              </select>
            </Field>
            <Field label="Iterations">
              <input
                type="number"
                value={form.n_iterations}
                onChange={(e) =>
                  setForm({ ...form, n_iterations: parseInt(e.target.value) || 1000 })
                }
                className="input-field"
              />
            </Field>
            {form.game_type === "simplified_holdem" && (
              <>
                <Field label="Preflop Buckets">
                  <input
                    type="number"
                    value={form.preflop_buckets}
                    onChange={(e) =>
                      setForm({ ...form, preflop_buckets: parseInt(e.target.value) || 8 })
                    }
                    min={2}
                    max={20}
                    className="input-field"
                  />
                </Field>
                <Field label="Flop Buckets">
                  <input
                    type="number"
                    value={form.flop_buckets}
                    onChange={(e) =>
                      setForm({ ...form, flop_buckets: parseInt(e.target.value) || 8 })
                    }
                    min={2}
                    max={20}
                    className="input-field"
                  />
                </Field>
              </>
            )}
          </div>

          {/* Game info */}
          {games.length > 0 && (
            <div className="mt-4 p-3 rounded-lg bg-graphite-800/30 border border-graphite-700/30">
              {games
                .filter((g) => g.id === form.game_type)
                .map((g) => (
                  <div key={g.id} className="text-xs text-graphite-400">
                    <span className="text-graphite-300 font-medium">{g.name}:</span>{" "}
                    {g.description}
                    <span className="text-graphite-500 ml-2">
                      Est. {g.estimated_time}
                    </span>
                  </div>
                ))}
            </div>
          )}

          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={() => setShowCreate(false)}
              className="px-4 py-2 text-sm text-graphite-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleCreate}
              disabled={creating}
              className="px-6 py-2 rounded-lg bg-accent-blue text-white text-sm font-medium hover:bg-accent-blue/90 transition-colors disabled:opacity-50"
            >
              {creating ? "Starting..." : "Start Training"}
            </button>
          </div>
        </div>
      )}

      {/* Runs table */}
      <div className="card">
        {loading ? (
          <div className="text-sm text-graphite-500 py-8 text-center">
            Loading runs...
          </div>
        ) : runs.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-4xl mb-4 opacity-20">◈</div>
            <p className="text-graphite-400 mb-2">No training runs yet</p>
            <p className="text-xs text-graphite-500">
              Click &quot;New Run&quot; to start training a solver
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-xs text-graphite-500 uppercase tracking-wider border-b border-graphite-800/50">
                  <th className="text-left py-3 px-4 font-medium">Run</th>
                  <th className="text-left py-3 px-4 font-medium">Game</th>
                  <th className="text-left py-3 px-4 font-medium">Status</th>
                  <th className="text-right py-3 px-4 font-medium">
                    Progress
                  </th>
                  <th className="text-right py-3 px-4 font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.id}
                    className="border-b border-graphite-800/30 hover:bg-graphite-800/20 transition-colors"
                  >
                    <td className="py-3 px-4">
                      <Link
                        href={`/dashboard/runs/${run.id}`}
                        className="font-mono text-sm text-accent-blue hover:text-accent-cyan transition-colors"
                      >
                        {run.name || run.id}
                      </Link>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-xs text-graphite-400">
                        {run.config.game_type === "simplified_holdem"
                          ? "Hold'em"
                          : "Kuhn"}
                      </span>
                      <span className="text-xs text-graphite-600 ml-1">
                        · {run.config.algorithm.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={statusBadgeClass(run.status)}>
                        {run.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-24 h-1.5 rounded-full bg-graphite-800 overflow-hidden">
                          <div
                            className="h-full bg-accent-blue rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.min(
                                (run.current_iteration /
                                  run.config.n_iterations) *
                                  100,
                                100
                              )}%`,
                            }}
                          />
                        </div>
                        <span className="font-mono text-xs text-graphite-400 w-12 text-right">
                          {Math.round(
                            (run.current_iteration / run.config.n_iterations) *
                              100
                          )}
                          %
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right text-xs text-graphite-500">
                      {run.created_at ? formatTimestamp(run.created_at) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-xs text-graphite-400 mb-1.5 font-medium">
        {label}
      </label>
      {children}
      <style jsx global>{`
        .input-field {
          width: 100%;
          padding: 0.5rem 0.75rem;
          border-radius: 0.5rem;
          border: 1px solid rgba(66, 66, 75, 0.5);
          background: rgba(58, 58, 65, 0.3);
          color: white;
          font-size: 0.875rem;
          outline: none;
          transition: border-color 0.2s;
        }
        .input-field:focus {
          border-color: rgba(59, 130, 246, 0.5);
        }
        .input-field option {
          background: #18181b;
          color: white;
        }
      `}</style>
    </div>
  );
}

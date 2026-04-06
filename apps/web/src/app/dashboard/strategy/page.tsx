"use client";

import { useState, useEffect } from "react";
import { api, type Run } from "@/lib/api";
import { StrategyTable } from "@/components/dashboard/strategy-table";
import { statusBadgeClass } from "@/lib/utils";

export default function StrategyPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listRuns()
      .then((res) => {
        const completed = res.runs.filter((r) => r.status === "completed");
        setRuns(completed);
        if (completed.length > 0 && !selectedRunId) {
          setSelectedRunId(completed[0].id);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    setStrategy(null);
    api
      .getRunStrategy(selectedRunId)
      .then((res) => setStrategy(res.strategy))
      .catch(() => {});
  }, [selectedRunId]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Strategy Explorer
        </h1>
        <p className="text-sm text-graphite-400 mt-1">
          Browse computed strategies from completed training runs
        </p>
      </div>

      {/* Run selector */}
      <div className="card">
        <label className="text-xs text-graphite-400 block mb-2">
          Select Training Run
        </label>
        {loading ? (
          <div className="text-sm text-graphite-500">Loading runs...</div>
        ) : runs.length === 0 ? (
          <div className="text-sm text-graphite-500">
            No completed runs. Train a solver first.
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {runs.map((run) => (
              <button
                key={run.id}
                onClick={() => setSelectedRunId(run.id)}
                className={`px-3 py-2 rounded-lg text-sm transition-all ${
                  selectedRunId === run.id
                    ? "bg-accent-blue/20 text-accent-blue border border-accent-blue/30"
                    : "bg-graphite-800/40 text-graphite-400 border border-graphite-800 hover:text-white"
                }`}
              >
                <span className="font-mono">{run.name || run.id}</span>
                <span className="text-xs text-graphite-500 ml-2">
                  {run.config.game_type === "simplified_holdem" ? "Hold'em" : "Kuhn"}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Strategy display */}
      {selectedRunId && strategy && (
        <StrategyTable
          strategy={strategy}
          gameType={runs.find((r) => r.id === selectedRunId)?.config.game_type || "kuhn"}
        />
      )}

      {selectedRunId && !strategy && (
        <div className="card text-center py-12 text-graphite-500 text-sm">
          Loading strategy data...
        </div>
      )}
    </div>
  );
}

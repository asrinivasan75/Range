"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, type Run } from "@/lib/api";
import { formatTimestamp, formatNumber, statusBadgeClass } from "@/lib/utils";

export default function ArtifactsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listRuns()
      .then((res) => {
        setRuns(res.runs.filter((r) => r.status === "completed"));
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Artifacts
        </h1>
        <p className="text-sm text-graphite-400 mt-1">
          Training outputs, checkpoints, and saved strategies
        </p>
      </div>

      <div className="card">
        <h2 className="text-sm font-semibold text-white mb-4">
          Completed Training Runs
        </h2>

        {loading ? (
          <div className="text-sm text-graphite-500 py-8 text-center">
            Loading...
          </div>
        ) : runs.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-graphite-400 mb-2">No completed runs yet</p>
            <Link
              href="/dashboard/runs"
              className="text-sm text-accent-blue hover:text-accent-cyan"
            >
              Start training →
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {runs.map((run) => (
              <div
                key={run.id}
                className="p-4 rounded-lg bg-graphite-800/30 border border-graphite-800/50 hover:border-graphite-700/50 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-sm text-white">
                      {run.name || run.id}
                    </span>
                    <span className={statusBadgeClass(run.status)}>
                      {run.status}
                    </span>
                  </div>
                  <span className="text-xs text-graphite-500">
                    {run.created_at ? formatTimestamp(run.created_at) : "—"}
                  </span>
                </div>

                <div className="flex items-center gap-4 text-xs text-graphite-500 mb-3">
                  <span>
                    {run.config.game_type === "simplified_holdem"
                      ? "Simplified Hold'em"
                      : "Kuhn Poker"}
                  </span>
                  <span>·</span>
                  <span>{formatNumber(run.current_iteration, 0)} iterations</span>
                  <span>·</span>
                  <span>{run.config.algorithm.toUpperCase()}</span>
                </div>

                {/* Artifact links */}
                <div className="flex flex-wrap gap-2">
                  <ArtifactBadge
                    label="Metrics"
                    href={`/dashboard/runs/${run.id}`}
                  />
                  <ArtifactBadge
                    label="Strategy"
                    href={`/dashboard/runs/${run.id}`}
                  />
                  <ArtifactBadge label="Checkpoint" tag="data/" />
                  <ArtifactBadge label="Config" tag="JSON" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* File structure info */}
      <div className="card">
        <h2 className="text-sm font-semibold text-white mb-4">
          Data Directory Structure
        </h2>
        <pre className="text-xs font-mono text-graphite-400 leading-loose">
{`data/
├── runs/
│   ├── <run_id>/
│   │   ├── run.json            # Run metadata & config
│   │   ├── metrics.json        # Full metrics history
│   │   ├── metrics_series.json # Time-series for charts
│   │   ├── summary.json        # Run summary stats
│   │   ├── strategy_latest.json# Final strategy snapshot
│   │   └── trainer_state.pkl   # Full trainer checkpoint
│   └── ...
└── range.db                    # SQLite metadata`}
        </pre>
      </div>
    </div>
  );
}

function ArtifactBadge({
  label,
  href,
  tag,
}: {
  label: string;
  href?: string;
  tag?: string;
}) {
  const content = (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-graphite-800/60 border border-graphite-700/30 text-xs text-graphite-300 hover:text-white hover:border-graphite-600 transition-colors cursor-pointer">
      <span className="w-1 h-1 rounded-full bg-accent-blue/50" />
      {label}
      {tag && <span className="text-graphite-600 text-[10px]">{tag}</span>}
    </span>
  );

  if (href) return <Link href={href}>{content}</Link>;
  return content;
}

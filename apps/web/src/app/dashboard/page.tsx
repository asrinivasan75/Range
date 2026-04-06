"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type Run, type RunMetrics } from "@/lib/api";
import { formatNumber, formatDuration, formatTimestamp, statusBadgeClass } from "@/lib/utils";

interface DashboardData {
  runs: Run[];
  loading: boolean;
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData>({ runs: [], loading: true });

  useEffect(() => {
    api
      .listRuns()
      .then((res) => setData({ runs: res.runs, loading: false }))
      .catch(() => setData((d) => ({ ...d, loading: false })));
  }, []);

  const completedRuns = data.runs.filter((r) => r.status === "completed");
  const runningRuns = data.runs.filter((r) => r.status === "running");
  const totalIterations = data.runs.reduce(
    (sum, r) => sum + r.current_iteration,
    0
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Dashboard
        </h1>
        <p className="text-sm text-graphite-400 mt-1">
          Training overview and system status
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Runs"
          value={data.runs.length.toString()}
          detail={`${completedRuns.length} completed`}
        />
        <StatCard
          label="Active"
          value={runningRuns.length.toString()}
          detail="currently training"
          accent
        />
        <StatCard
          label="Total Iterations"
          value={formatNumber(totalIterations, 0)}
          detail="across all runs"
        />
        <StatCard
          label="Games Available"
          value="2"
          detail="Kuhn + Hold'em"
        />
      </div>

      {/* Recent runs */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white">Recent Runs</h2>
          <Link
            href="/dashboard/runs"
            className="text-xs text-accent-blue hover:text-accent-cyan transition-colors"
          >
            View All →
          </Link>
        </div>

        {data.loading ? (
          <div className="text-sm text-graphite-500 py-8 text-center">
            Loading...
          </div>
        ) : data.runs.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-graphite-400 mb-4">No training runs yet</p>
            <Link
              href="/dashboard/runs"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-blue/10 text-accent-blue text-sm font-medium hover:bg-accent-blue/20 transition-colors"
            >
              Start Your First Run
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {data.runs.slice(0, 5).map((run) => (
              <Link
                key={run.id}
                href={`/dashboard/runs/${run.id}`}
                className="flex items-center gap-4 p-4 rounded-lg hover:bg-graphite-800/30 transition-colors group"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-white truncate">
                      {run.name || run.id}
                    </span>
                    <span className={statusBadgeClass(run.status)}>
                      {run.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-graphite-500">
                    <span>{run.config.game_type}</span>
                    <span>·</span>
                    <span>{run.config.algorithm.toUpperCase()}</span>
                    <span>·</span>
                    <span>
                      {formatNumber(run.current_iteration, 0)} /{" "}
                      {formatNumber(run.config.n_iterations, 0)} iters
                    </span>
                  </div>
                </div>
                <div className="text-xs text-graphite-500">
                  {run.created_at ? formatTimestamp(run.created_at) : "—"}
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Quick actions */}
      <div className="grid md:grid-cols-2 gap-4">
        <QuickAction
          title="Train Kuhn Poker"
          description="Validate CFR convergence on the simplest poker game. 12 info sets, trains in seconds."
          href="/dashboard/runs"
          tag="Foundation"
        />
        <QuickAction
          title="Train Simplified Hold'em"
          description="Preflop + flop with coarse abstraction. ~3000 info sets, trains in 1-3 minutes."
          href="/dashboard/runs"
          tag="Primary"
        />
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  detail,
  accent,
}: {
  label: string;
  value: string;
  detail: string;
  accent?: boolean;
}) {
  return (
    <div className="card">
      <div className="stat-label mb-2">{label}</div>
      <div className={`stat-value ${accent ? "text-accent-blue" : ""}`}>
        {value}
      </div>
      <div className="text-xs text-graphite-500 mt-1">{detail}</div>
    </div>
  );
}

function QuickAction({
  title,
  description,
  href,
  tag,
}: {
  title: string;
  description: string;
  href: string;
  tag: string;
}) {
  return (
    <Link href={href} className="card-interactive group">
      <div className="flex items-start justify-between mb-3">
        <h3 className="font-semibold text-white text-sm group-hover:text-accent-blue transition-colors">
          {title}
        </h3>
        <span className="badge-blue text-[10px]">{tag}</span>
      </div>
      <p className="text-xs text-graphite-400 leading-relaxed">{description}</p>
    </Link>
  );
}

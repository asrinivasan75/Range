"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, type Run, type RunMetrics, type StrategyEntry } from "@/lib/api";
import { formatNumber, formatDuration, statusBadgeClass } from "@/lib/utils";
import { MetricsCharts } from "@/components/dashboard/metrics-charts";
import { StrategyTable } from "@/components/dashboard/strategy-table";

export default function RunDetailPage() {
  const params = useParams();
  const runId = params.id as string;

  const [run, setRun] = useState<Run | null>(null);
  const [metrics, setMetrics] = useState<RunMetrics | null>(null);
  const [strategy, setStrategy] = useState<Record<string, StrategyEntry> | null>(null);
  const [activeTab, setActiveTab] = useState<"metrics" | "strategy">("metrics");

  useEffect(() => {
    const fetchAll = () => {
      api.getRun(runId).then((res) => setRun(res.run)).catch(() => {});
      api.getRunMetrics(runId).then(setMetrics).catch(() => {});
      api.getRunStrategy(runId).then((res) => setStrategy(res.strategy)).catch(() => {});
    };

    fetchAll();
    const interval = setInterval(fetchAll, 2000);
    return () => clearInterval(interval);
  }, [runId]);

  if (!run) {
    return (
      <div className="text-graphite-500 py-12 text-center">Loading run...</div>
    );
  }

  const progress = Math.round(
    (run.current_iteration / run.config.n_iterations) * 100
  );

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-graphite-500">
        <Link href="/dashboard/runs" className="hover:text-graphite-300">
          Runs
        </Link>
        <span>/</span>
        <span className="text-graphite-300 font-mono">{run.name || run.id}</span>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-white tracking-tight font-mono">
              {run.name || run.id}
            </h1>
            <span className={statusBadgeClass(run.status)}>{run.status}</span>
          </div>
          <div className="flex items-center gap-4 mt-2 text-xs text-graphite-500">
            <span>
              {run.config.game_type === "simplified_holdem"
                ? "Simplified Hold'em"
                : "Kuhn Poker"}
            </span>
            <span>·</span>
            <span>{run.config.algorithm.toUpperCase()}</span>
            <span>·</span>
            <span>{formatNumber(run.config.n_iterations, 0)} iterations</span>
            {run.config.game_type === "simplified_holdem" && (
              <>
                <span>·</span>
                <span>
                  {run.config.preflop_buckets}×{run.config.flop_buckets} buckets
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-graphite-400">Training Progress</span>
          <span className="font-mono text-sm text-white">{progress}%</span>
        </div>
        <div className="w-full h-2 rounded-full bg-graphite-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-blue to-accent-cyan transition-all duration-700"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-3 text-xs text-graphite-500">
          <span>
            {formatNumber(run.current_iteration, 0)} /{" "}
            {formatNumber(run.config.n_iterations, 0)} iterations
          </span>
          {metrics?.summary && (
            <span>
              {formatDuration(metrics.summary.total_time_seconds)} elapsed
            </span>
          )}
        </div>
      </div>

      {/* Summary stats */}
      {metrics?.summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MiniStat
            label="Info Sets"
            value={formatNumber(metrics.summary.n_info_sets, 0)}
          />
          <MiniStat
            label="Exploitability"
            value={formatNumber(metrics.summary.final_exploitability, 4)}
          />
          <MiniStat
            label="Avg Regret"
            value={formatNumber(metrics.summary.final_avg_regret, 4)}
          />
          <MiniStat
            label="Avg Iteration"
            value={`${formatNumber(metrics.summary.avg_iteration_ms, 1)}ms`}
          />
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b border-graphite-800/50 pb-px">
        <TabButton
          active={activeTab === "metrics"}
          onClick={() => setActiveTab("metrics")}
        >
          Metrics & Charts
        </TabButton>
        <TabButton
          active={activeTab === "strategy"}
          onClick={() => setActiveTab("strategy")}
        >
          Strategy Explorer
        </TabButton>
      </div>

      {/* Tab content */}
      {activeTab === "metrics" && metrics?.series && (
        <MetricsCharts series={metrics.series} />
      )}

      {activeTab === "strategy" && strategy && (
        <StrategyTable
          strategy={strategy}
          gameType={run.config.game_type}
        />
      )}

      {activeTab === "metrics" && !metrics?.series && (
        <div className="card text-center py-12 text-graphite-500 text-sm">
          {run.status === "running"
            ? "Metrics will appear as training progresses..."
            : "No metrics data available for this run."}
        </div>
      )}

      {activeTab === "strategy" && !strategy && (
        <div className="card text-center py-12 text-graphite-500 text-sm">
          {run.status === "running"
            ? "Strategy will be available after training completes..."
            : "No strategy data available for this run."}
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <div className="stat-label mb-1">{label}</div>
      <div className="font-mono text-lg font-bold text-white">{value}</div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium transition-colors relative ${
        active
          ? "text-white"
          : "text-graphite-500 hover:text-graphite-300"
      }`}
    >
      {children}
      {active && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-blue rounded-full" />
      )}
    </button>
  );
}

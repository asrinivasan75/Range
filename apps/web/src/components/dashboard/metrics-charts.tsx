"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

interface MetricsSeries {
  iterations: number[];
  exploitability: number[];
  total_regret: number[];
  avg_regret: number[];
  n_info_sets: number[];
  iteration_time_ms: number[];
  timestamps: number[];
}

interface MetricsChartsProps {
  series: MetricsSeries;
}

const chartTheme = {
  bg: "#18181b",
  grid: "#2a2a30",
  text: "#6b6b78",
  blue: "#3b82f6",
  cyan: "#06b6d4",
  emerald: "#10b981",
  amber: "#f59e0b",
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload) return null;
  return (
    <div className="glass rounded-lg px-3 py-2 shadow-lg border border-graphite-700/50">
      <div className="text-xs text-graphite-400 mb-1 font-mono">
        Iteration {label?.toLocaleString()}
      </div>
      {payload.map((entry: any) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-graphite-400">{entry.name}:</span>
          <span className="font-mono text-white">
            {typeof entry.value === "number"
              ? entry.value < 0.01
                ? entry.value.toExponential(2)
                : entry.value.toFixed(4)
              : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
};

export function MetricsCharts({ series }: MetricsChartsProps) {
  const data = useMemo(() => {
    return series.iterations.map((iter, i) => ({
      iteration: iter,
      exploitability: series.exploitability[i],
      total_regret: series.total_regret[i],
      avg_regret: series.avg_regret[i],
      n_info_sets: series.n_info_sets[i],
      iteration_time_ms: series.iteration_time_ms[i],
    }));
  }, [series]);

  if (data.length === 0) {
    return (
      <div className="card text-center py-12 text-graphite-500 text-sm">
        No metrics data yet
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Exploitability convergence */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Exploitability Convergence
            </h3>
            <p className="text-xs text-graphite-500 mt-0.5">
              Lower is better — approaches zero at Nash equilibrium
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-accent-blue" />
            <span className="text-xs text-graphite-400">Exploitability Proxy</span>
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="exploitGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={chartTheme.blue} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={chartTheme.blue} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
              <XAxis
                dataKey="iteration"
                stroke={chartTheme.text}
                fontSize={10}
                tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
              />
              <YAxis
                stroke={chartTheme.text}
                fontSize={10}
                tickFormatter={(v) => v < 0.01 ? v.toExponential(1) : v.toFixed(2)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="exploitability"
                name="Exploitability"
                stroke={chartTheme.blue}
                fill="url(#exploitGrad)"
                strokeWidth={2}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Regret metrics */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="card">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-white">
              Regret Magnitude
            </h3>
            <p className="text-xs text-graphite-500 mt-0.5">
              Total and average cumulative regret
            </p>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="iteration"
                  stroke={chartTheme.text}
                  fontSize={10}
                  tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                />
                <YAxis stroke={chartTheme.text} fontSize={10} />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="total_regret"
                  name="Total Regret"
                  stroke={chartTheme.amber}
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="avg_regret"
                  name="Avg Regret"
                  stroke={chartTheme.emerald}
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-white">
              Iteration Performance
            </h3>
            <p className="text-xs text-graphite-500 mt-0.5">
              Time per iteration (ms)
            </p>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="iteration"
                  stroke={chartTheme.text}
                  fontSize={10}
                  tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                />
                <YAxis stroke={chartTheme.text} fontSize={10} />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="iteration_time_ms"
                  name="Time (ms)"
                  stroke={chartTheme.cyan}
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

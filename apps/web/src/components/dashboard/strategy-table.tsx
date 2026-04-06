"use client";

import { useState, useMemo } from "react";
import type { StrategyEntry } from "@/lib/api";

interface StrategyTableProps {
  strategy: Record<string, StrategyEntry>;
  gameType: string;
}

const KUHN_ACTION_LABELS: Record<string, string[]> = {
  default: ["Check/Pass", "Bet"],
  facing_bet: ["Fold", "Call"],
};

const HOLDEM_ACTION_LABELS: Record<string, string> = {
  k: "Check",
  h: "Bet ½",
  b: "Bet Pot",
  f: "Fold",
  c: "Call",
};

export function StrategyTable({ strategy, gameType }: StrategyTableProps) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"key" | "visits">("visits");

  const entries = useMemo(() => {
    let items = Object.entries(strategy).map(([key, entry]) => ({
      key,
      ...entry,
    }));

    if (search) {
      items = items.filter((e) =>
        e.key.toLowerCase().includes(search.toLowerCase())
      );
    }

    if (sortBy === "visits") {
      items.sort((a, b) => b.visits - a.visits);
    } else {
      items.sort((a, b) => a.key.localeCompare(b.key));
    }

    return items;
  }, [strategy, search, sortBy]);

  const getActionLabels = (key: string): string[] => {
    if (gameType === "kuhn") {
      const actions = key.split(":")[1] || "";
      if (actions.endsWith("b")) return ["Fold", "Call"];
      return ["Check", "Bet"];
    }
    // Simplified Hold'em — determine from context
    const actionPart = key.split("|")[1] || "";
    const currentStreet = actionPart.includes("/")
      ? actionPart.split("/").pop() || ""
      : actionPart;
    const nRaises = (currentStreet.match(/[hb]/g) || []).length;
    const hasBet = nRaises > 0;
    const atCap = nRaises >= 2;

    if (!hasBet) return ["Check", "Bet ½", "Bet Pot"];
    if (atCap) return ["Fold", "Call"];
    return ["Fold", "Call", "Raise ½"];
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search info sets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full max-w-xs px-3 py-2 rounded-lg bg-graphite-900/60 border border-graphite-800/50 text-sm text-white placeholder-graphite-600 focus:border-accent-blue/50 outline-none"
          />
        </div>
        <div className="flex items-center gap-2 text-xs text-graphite-500">
          <span>Sort:</span>
          <button
            onClick={() => setSortBy("visits")}
            className={`px-2 py-1 rounded ${
              sortBy === "visits" ? "text-white bg-graphite-800" : "hover:text-white"
            }`}
          >
            Visits
          </button>
          <button
            onClick={() => setSortBy("key")}
            className={`px-2 py-1 rounded ${
              sortBy === "key" ? "text-white bg-graphite-800" : "hover:text-white"
            }`}
          >
            Name
          </button>
        </div>
        <span className="text-xs text-graphite-500">
          {entries.length} info sets
        </span>
      </div>

      {/* Strategy cards */}
      <div className="grid gap-2">
        {entries.slice(0, 50).map((entry) => {
          const labels = getActionLabels(entry.key);
          return (
            <div
              key={entry.key}
              className="card p-4 hover:bg-graphite-800/40 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <code className="text-sm font-mono text-accent-blue">
                  {entry.key}
                </code>
                <span className="text-xs text-graphite-500">
                  {Math.round(entry.visits).toLocaleString()} visits
                </span>
              </div>

              {/* Strategy bar */}
              <div className="space-y-1.5">
                {entry.strategy.map((prob, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span className="text-xs text-graphite-400 w-20 shrink-0 truncate">
                      {labels[i] || `Action ${i}`}
                    </span>
                    <div className="flex-1 h-5 rounded-md bg-graphite-800 overflow-hidden relative">
                      <div
                        className="h-full rounded-md transition-all duration-300"
                        style={{
                          width: `${Math.max(prob * 100, 0.5)}%`,
                          background: getBarColor(i, entry.strategy.length),
                        }}
                      />
                    </div>
                    <span className="font-mono text-xs text-graphite-300 w-14 text-right">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {entries.length > 50 && (
        <div className="text-center text-xs text-graphite-500 py-4">
          Showing 50 of {entries.length} info sets
        </div>
      )}
    </div>
  );
}

function getBarColor(index: number, total: number): string {
  const colors = [
    "#3b82f6", // blue
    "#10b981", // emerald
    "#f59e0b", // amber
    "#06b6d4", // cyan
    "#f43f5e", // rose
  ];
  return colors[index % colors.length];
}

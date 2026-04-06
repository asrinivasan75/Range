"use client";

import { useState, useEffect } from "react";
import { api, type EquityResult, type PreflopChart } from "@/lib/api";
import { bucketColor } from "@/lib/utils";

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = [
  { char: "s", symbol: "♠", color: "text-graphite-300" },
  { char: "h", symbol: "♥", color: "text-red-400" },
  { char: "d", symbol: "♦", color: "text-blue-400" },
  { char: "c", symbol: "♣", color: "text-green-400" },
];

export default function ExplorerPage() {
  const [card1, setCard1] = useState({ rank: "A", suit: "h" });
  const [card2, setCard2] = useState({ rank: "K", suit: "s" });
  const [board, setBoard] = useState<string[]>([]);
  const [equity, setEquity] = useState<EquityResult | null>(null);
  const [preflopChart, setPreflopChart] = useState<PreflopChart | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getPreflopChart().then(setPreflopChart).catch(() => {});
  }, []);

  const computeEquity = async () => {
    setLoading(true);
    try {
      const result = await api.computeEquity(
        [`${card1.rank}${card1.suit}`, `${card2.rank}${card2.suit}`],
        board
      );
      setEquity(result);
    } catch {
      // API may not be running
    }
    setLoading(false);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Hand Explorer
        </h1>
        <p className="text-sm text-graphite-400 mt-1">
          Analyze hand equity and preflop hand strength
        </p>
      </div>

      {/* Equity calculator */}
      <div className="card">
        <h2 className="text-sm font-semibold text-white mb-4">
          Equity Calculator
        </h2>
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="text-xs text-graphite-400 block mb-1">Card 1</label>
            <div className="flex gap-1">
              <select
                value={card1.rank}
                onChange={(e) => setCard1({ ...card1, rank: e.target.value })}
                className="px-2 py-1.5 rounded bg-graphite-800 border border-graphite-700 text-white text-sm"
              >
                {RANKS.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
              <select
                value={card1.suit}
                onChange={(e) => setCard1({ ...card1, suit: e.target.value })}
                className="px-2 py-1.5 rounded bg-graphite-800 border border-graphite-700 text-white text-sm"
              >
                {SUITS.map((s) => (
                  <option key={s.char} value={s.char}>{s.symbol}</option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <label className="text-xs text-graphite-400 block mb-1">Card 2</label>
            <div className="flex gap-1">
              <select
                value={card2.rank}
                onChange={(e) => setCard2({ ...card2, rank: e.target.value })}
                className="px-2 py-1.5 rounded bg-graphite-800 border border-graphite-700 text-white text-sm"
              >
                {RANKS.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
              <select
                value={card2.suit}
                onChange={(e) => setCard2({ ...card2, suit: e.target.value })}
                className="px-2 py-1.5 rounded bg-graphite-800 border border-graphite-700 text-white text-sm"
              >
                {SUITS.map((s) => (
                  <option key={s.char} value={s.char}>{s.symbol}</option>
                ))}
              </select>
            </div>
          </div>
          <button
            onClick={computeEquity}
            disabled={loading}
            className="px-5 py-1.5 rounded-lg bg-accent-blue text-white text-sm font-medium hover:bg-accent-blue/90 transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Calculate"}
          </button>
        </div>

        {/* Results */}
        {equity && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <ResultStat label="Equity" value={`${equity.equity_pct}%`} />
            {equity.chen_score !== undefined && (
              <ResultStat label="Chen Score" value={equity.chen_score.toFixed(1)} />
            )}
            {equity.preflop_bucket !== undefined && (
              <ResultStat label="Preflop Bucket" value={`${equity.preflop_bucket}/7`} />
            )}
            <ResultStat label="Hand" value={equity.hole_cards.join(" ")} mono />
          </div>
        )}
      </div>

      {/* Preflop chart */}
      {preflopChart && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Preflop Hand Chart
              </h2>
              <p className="text-xs text-graphite-500 mt-0.5">
                Hand strength buckets (0=weakest, 7=strongest). Suited above diagonal.
              </p>
            </div>
            <div className="flex items-center gap-1">
              {Array.from({ length: 8 }, (_, i) => (
                <div
                  key={i}
                  className="w-3 h-3 rounded-sm"
                  style={{ background: bucketColor(i) }}
                  title={`Bucket ${i}`}
                />
              ))}
            </div>
          </div>

          <div className="overflow-x-auto">
            <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `auto repeat(13, 1fr)` }}>
              {/* Header row */}
              <div />
              {RANKS.map((r) => (
                <div key={r} className="text-center text-xs font-mono text-graphite-500 pb-1 w-10">
                  {r}
                </div>
              ))}

              {/* Grid */}
              {preflopChart.grid.map((row, i) => (
                <>
                  <div key={`label-${i}`} className="text-xs font-mono text-graphite-500 pr-2 flex items-center justify-end">
                    {RANKS[i]}
                  </div>
                  {row.map((cell, j) => (
                    <div
                      key={`${i}-${j}`}
                      className="w-10 h-8 rounded-sm flex items-center justify-center text-[9px] font-mono transition-transform hover:scale-110 cursor-default"
                      style={{
                        background: bucketColor(cell.bucket),
                        opacity: 0.7 + (cell.bucket / 8) * 0.3,
                      }}
                      title={`${cell.hand}: Bucket ${cell.bucket}`}
                    >
                      <span className="text-white/80 font-medium">
                        {cell.hand.replace(/[os]$/, "")}
                      </span>
                    </div>
                  ))}
                </>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ResultStat({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="p-3 rounded-lg bg-graphite-800/40 border border-graphite-700/30">
      <div className="text-xs text-graphite-500 mb-1">{label}</div>
      <div className={`text-lg font-bold text-white ${mono ? "font-mono" : ""}`}>
        {value}
      </div>
    </div>
  );
}

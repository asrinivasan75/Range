"use client";

import { FadeUp } from "@/components/ui/motion";

const EARNINGS_BOARD = [
  { name: "monkey_test_2.5", score: 260.71, ours: false },
  { name: "pm2", score: 124.5, ours: false },
  { name: "eventreinforcev4", score: 96.34, ours: true },
  { name: "pm", score: 58.12, ours: false },
  { name: "res_v3_b8", score: 57.25, ours: false },
  { name: "EVPA1691e7-11", score: 54.32, ours: false },
  { name: "chrisal3", score: 47.88, ours: false },
  { name: "EVPA30-0.02s-46", score: 41.46, ours: false },
  { name: "fernandin", score: 40.83, ours: false },
  { name: "runnerrunner", score: 38.82, ours: false },
];

const BASELINE_BOARD = [
  { name: "pm2", score: 94.14, ours: false },
  { name: "EVPA1691e7-11", score: 58.58, ours: false },
  { name: "Baseline_Blueprint", score: 44.83, ours: false },
  { name: "unimaru", score: 41.54, ours: false },
  { name: "eventreinforcev4", score: 38.62, ours: true },
  { name: "kaike_pokerbot1", score: 35.48, ours: false },
  { name: "pm", score: 35.18, ours: false },
  { name: "DeepStack1e8iter9", score: 34.1, ours: false },
  { name: "monkey_test_2.5", score: 31.26, ours: false },
  { name: "s1060", score: 31.0, ours: false },
];

function LeaderboardTable({ title, data, subtitle }: { title: string; data: typeof EARNINGS_BOARD; subtitle: string }) {
  return (
    <div className="card">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-white">{title}</h3>
        <p className="text-xs text-graphite-500 mt-0.5">{subtitle}</p>
      </div>
      <div className="space-y-0.5">
        {data.map((entry, i) => (
          <div
            key={entry.name}
            className={`flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
              entry.ours
                ? "bg-accent-emerald/10 border border-accent-emerald/20"
                : i % 2 === 0
                ? "bg-graphite-800/20"
                : ""
            }`}
          >
            <div className="flex items-center gap-3">
              <span className={`font-mono text-xs w-6 text-right ${
                entry.ours ? "text-accent-emerald font-bold" : "text-graphite-500"
              }`}>
                #{i + 1}
              </span>
              <span className={`text-sm font-mono ${
                entry.ours ? "text-accent-emerald font-bold" : "text-graphite-300"
              }`}>
                {entry.name}
              </span>
              {entry.ours && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-emerald/20 text-accent-emerald font-medium uppercase tracking-wider">
                  Ours
                </span>
              )}
            </div>
            <span className={`font-mono text-sm font-bold ${
              entry.ours ? "text-accent-emerald" : "text-white"
            }`}>
              +{entry.score}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Leaderboard() {
  return (
    <section className="relative py-32 px-6">
      <div className="glow-line max-w-lg mx-auto mb-32" />

      <div className="max-w-5xl mx-auto">
        <FadeUp>
          <div className="text-center mb-16">
            <p className="text-xs uppercase tracking-[0.2em] text-accent-emerald font-medium mb-4">
              Benchmark Results
            </p>
            <h2 className="text-display-md font-bold text-white mb-4">
              Slumbot Leaderboard
            </h2>
            <p className="text-graphite-400 max-w-2xl mx-auto text-lg">
              Our Q-learning agent ranks <span className="text-accent-emerald font-semibold">#3 worldwide</span> against
              Slumbot — a near-Nash HUNL poker bot that won the 2018 Annual Computer Poker Competition.
            </p>
            <p className="text-graphite-500 text-sm mt-2">
              Min. 5,000 hands · BB/100 (big blinds per 100 hands)
            </p>
          </div>
        </FadeUp>

        <div className="grid md:grid-cols-2 gap-6">
          <FadeUp delay={0.1}>
            <LeaderboardTable
              title="Earnings"
              subtitle="Raw win rate (BB/100)"
              data={EARNINGS_BOARD}
            />
          </FadeUp>
          <FadeUp delay={0.2}>
            <LeaderboardTable
              title="Baseline"
              subtitle="Luck-adjusted (BB/100)"
              data={BASELINE_BOARD}
            />
          </FadeUp>
        </div>

        <FadeUp delay={0.3}>
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full border border-graphite-700/50 bg-graphite-900/50">
              <span className="w-2 h-2 rounded-full bg-accent-emerald animate-pulse" />
              <span className="text-sm text-graphite-300">
                Training in progress — PBT with 8 agents running continuously
              </span>
            </div>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

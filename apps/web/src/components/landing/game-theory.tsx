"use client";

import { FadeUp, FadeIn } from "@/components/ui/motion";
import { motion } from "framer-motion";

export function GameTheory() {
  return (
    <section className="relative py-32 px-6 overflow-hidden">
      {/* Ambient glow */}
      <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-accent-emerald/3 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-5xl mx-auto">
        <FadeUp>
          <div className="text-center mb-20">
            <p className="text-xs uppercase tracking-[0.2em] text-accent-cyan font-medium mb-4">
              Game Theory
            </p>
            <h2 className="text-display-md font-bold text-white mb-4">
              Why Solve Poker?
            </h2>
            <p className="text-graphite-400 max-w-2xl mx-auto text-lg">
              Poker is the canonical benchmark for decision-making under uncertainty
            </p>
          </div>
        </FadeUp>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Left: The Problem */}
          <FadeUp delay={0.1}>
            <div className="card h-full">
              <div className="text-xs uppercase tracking-wider text-graphite-500 mb-4 font-medium">
                The Problem
              </div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Imperfect Information
              </h3>
              <div className="space-y-4 text-sm text-graphite-400 leading-relaxed">
                <p>
                  Unlike chess or Go, poker involves <span className="text-graphite-200">hidden information</span>.
                  You cannot see your opponent&apos;s cards. Optimal play requires reasoning about
                  what they <em>could</em> hold, not just what the board shows.
                </p>
                <p>
                  This makes poker fundamentally harder than perfect-information games.
                  You need <span className="text-graphite-200">mixed strategies</span> — randomizing
                  between actions to remain unexploitable.
                </p>
              </div>

              {/* Visual: hidden cards */}
              <div className="mt-6 flex items-center justify-center gap-4">
                <div className="flex gap-1">
                  <div className="w-12 h-16 rounded-md bg-accent-blue/20 border border-accent-blue/30 flex items-center justify-center font-mono text-accent-blue text-lg">A</div>
                  <div className="w-12 h-16 rounded-md bg-accent-blue/20 border border-accent-blue/30 flex items-center justify-center font-mono text-accent-blue text-lg">K</div>
                </div>
                <span className="text-graphite-600 text-lg">vs</span>
                <div className="flex gap-1">
                  <div className="w-12 h-16 rounded-md bg-graphite-800 border border-graphite-700 flex items-center justify-center text-graphite-600 text-lg">?</div>
                  <div className="w-12 h-16 rounded-md bg-graphite-800 border border-graphite-700 flex items-center justify-center text-graphite-600 text-lg">?</div>
                </div>
              </div>
            </div>
          </FadeUp>

          {/* Right: The Solution */}
          <FadeUp delay={0.2}>
            <div className="card h-full">
              <div className="text-xs uppercase tracking-wider text-graphite-500 mb-4 font-medium">
                The Solution
              </div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Counterfactual Regret Minimization
              </h3>
              <div className="space-y-4 text-sm text-graphite-400 leading-relaxed">
                <p>
                  CFR is a self-play algorithm. At each information set, it tracks
                  <span className="text-graphite-200"> counterfactual regret</span> — how much better
                  each action would have performed compared to the chosen strategy.
                </p>
                <p>
                  Over iterations, positive regrets push the strategy toward better actions.
                  The <span className="text-graphite-200">average strategy</span> across all iterations
                  converges to an <span className="text-accent-emerald">approximate Nash equilibrium</span>.
                </p>
              </div>

              {/* Visual: convergence */}
              <div className="mt-6 p-4 rounded-lg bg-graphite-800/50 border border-graphite-700/50">
                <div className="flex items-end gap-1 h-16">
                  {[40, 32, 25, 20, 16, 13, 10, 8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.8, 2.5].map(
                    (h, i) => (
                      <motion.div
                        key={i}
                        initial={{ height: 0 }}
                        whileInView={{ height: `${h * 1.5}px` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.4, delay: i * 0.05 }}
                        className="flex-1 bg-gradient-to-t from-accent-blue/40 to-accent-blue/80 rounded-t-sm min-w-[4px]"
                      />
                    )
                  )}
                </div>
                <div className="flex justify-between mt-2">
                  <span className="text-[10px] text-graphite-500 font-mono">Iteration 0</span>
                  <span className="text-[10px] text-graphite-500 font-mono">Exploitability →</span>
                  <span className="text-[10px] text-graphite-500 font-mono">10,000</span>
                </div>
              </div>
            </div>
          </FadeUp>
        </div>

        {/* Key equation */}
        <FadeUp delay={0.3}>
          <div className="mt-8 card text-center">
            <p className="text-xs uppercase tracking-wider text-graphite-500 mb-3 font-medium">
              Core Update Rule
            </p>
            <div className="font-mono text-lg md:text-xl text-graphite-200 tracking-tight">
              <span className="text-accent-blue">R</span>
              <sup className="text-accent-cyan text-sm">T</sup>(I, a) =
              <span className="text-graphite-400"> ∑</span>
              <sub className="text-xs text-graphite-500">t=1..T</sub>{" "}
              <span className="text-accent-emerald">π</span>
              <sub className="text-xs text-graphite-500">-i</sub>(h) ·{" "}
              [<span className="text-accent-amber">u</span>
              <sub className="text-xs text-graphite-500">i</sub>(σ
              <sub className="text-xs">I→a</sub>, h) −{" "}
              <span className="text-accent-amber">u</span>
              <sub className="text-xs text-graphite-500">i</sub>(σ, h)]
            </div>
            <p className="text-xs text-graphite-500 mt-3">
              Cumulative counterfactual regret for action <em>a</em> at information set <em>I</em>
            </p>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

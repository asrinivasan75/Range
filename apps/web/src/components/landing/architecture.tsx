"use client";

import { FadeUp, FadeIn } from "@/components/ui/motion";
import { motion } from "framer-motion";

const layers = [
  {
    name: "Frontend",
    tech: "Next.js · TypeScript · Tailwind · Framer Motion",
    items: ["Landing Experience", "Training Dashboard", "Strategy Explorer", "Analysis Tools"],
    color: "border-accent-blue/30 bg-accent-blue/5",
    dotColor: "bg-accent-blue",
  },
  {
    name: "API Layer",
    tech: "FastAPI · Pydantic · Async",
    items: ["REST Endpoints", "Training Control", "Metrics Streaming", "Analysis Queries"],
    color: "border-accent-cyan/30 bg-accent-cyan/5",
    dotColor: "bg-accent-cyan",
  },
  {
    name: "Solver Engine",
    tech: "Python · NumPy · CFR/MCCFR",
    items: ["Kuhn Poker (validation)", "Simplified Hold'em", "Card Abstraction", "Action Abstraction"],
    color: "border-accent-emerald/30 bg-accent-emerald/5",
    dotColor: "bg-accent-emerald",
  },
  {
    name: "Persistence",
    tech: "SQLite · JSON · Checkpoints",
    items: ["Training Runs", "Strategy Snapshots", "Metrics History", "Configuration"],
    color: "border-accent-amber/30 bg-accent-amber/5",
    dotColor: "bg-accent-amber",
  },
];

export function Architecture() {
  return (
    <section className="relative py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeUp>
          <div className="text-center mb-20">
            <p className="text-xs uppercase tracking-[0.2em] text-accent-emerald font-medium mb-4">
              Architecture
            </p>
            <h2 className="text-display-md font-bold text-white mb-4">
              Built for Extension
            </h2>
            <p className="text-graphite-400 max-w-2xl mx-auto text-lg">
              Clean separation of concerns. Each layer is independently upgradeable.
            </p>
          </div>
        </FadeUp>

        <div className="space-y-4">
          {layers.map((layer, i) => (
            <FadeUp key={layer.name} delay={i * 0.12}>
              <div className={`rounded-xl border p-6 md:p-8 ${layer.color} transition-all duration-300 hover:scale-[1.01]`}>
                <div className="flex flex-col md:flex-row md:items-center gap-6">
                  <div className="md:w-64 shrink-0">
                    <div className="flex items-center gap-3 mb-1">
                      <div className={`w-2 h-2 rounded-full ${layer.dotColor}`} />
                      <h3 className="font-semibold text-white text-lg">
                        {layer.name}
                      </h3>
                    </div>
                    <p className="text-xs font-mono text-graphite-500 ml-5">
                      {layer.tech}
                    </p>
                  </div>

                  <div className="flex-1 grid grid-cols-2 md:grid-cols-4 gap-3">
                    {layer.items.map((item) => (
                      <div
                        key={item}
                        className="px-3 py-2 rounded-lg bg-graphite-900/60 border border-graphite-800/50 text-sm text-graphite-300 text-center"
                      >
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </FadeUp>
          ))}
        </div>

        {/* Connection lines visualization */}
        <FadeIn delay={0.6}>
          <div className="mt-16 text-center">
            <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full border border-graphite-700/50 bg-graphite-900/50">
              <span className="text-xs text-graphite-400">Future acceleration:</span>
              <span className="font-mono text-xs text-graphite-300">Rust / C++</span>
              <span className="text-xs text-graphite-500">for hot-loop CFR traversal</span>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

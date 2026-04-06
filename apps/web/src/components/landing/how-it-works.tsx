"use client";

import { FadeUp, FadeIn } from "@/components/ui/motion";

const steps = [
  {
    number: "01",
    title: "Abstract",
    subtitle: "Reduce Complexity",
    description:
      "Full No-Limit Hold'em has ~10^160 game states. We reduce this to a tractable size through card bucketing (grouping similar hands by strength) and action abstraction (limiting bet sizes to key fractions).",
    detail: "169 starting hands → 8 strategic buckets",
    color: "from-accent-blue to-accent-cyan",
  },
  {
    number: "02",
    title: "Train",
    subtitle: "Iterate Toward Equilibrium",
    description:
      "Counterfactual Regret Minimization (CFR) repeatedly plays against itself, accumulating regret for suboptimal actions. Over thousands of iterations, strategies converge toward a Nash equilibrium where no player can improve by deviating.",
    detail: "Regret → Strategy → Convergence",
    color: "from-accent-cyan to-accent-emerald",
  },
  {
    number: "03",
    title: "Analyze",
    subtitle: "Extract Insights",
    description:
      "Inspect the trained strategies: how often to bet, call, or fold at each decision point. Visualize convergence curves, exploitability metrics, and strategy distributions across information sets.",
    detail: "Strategy × Information Set → Optimal Play",
    color: "from-accent-emerald to-accent-amber",
  },
  {
    number: "04",
    title: "Iterate",
    subtitle: "Refine & Extend",
    description:
      "Increase abstraction granularity, add streets, expand bet sizing, or move to larger game variants. Each training run builds on the foundation toward increasingly accurate equilibrium approximations.",
    detail: "Foundation → Full Solver Pipeline",
    color: "from-accent-amber to-accent-rose",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="relative py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeUp>
          <div className="text-center mb-20">
            <p className="text-xs uppercase tracking-[0.2em] text-accent-blue font-medium mb-4">
              The Pipeline
            </p>
            <h2 className="text-display-md font-bold text-white mb-4">
              From Game State to Equilibrium
            </h2>
            <p className="text-graphite-400 max-w-2xl mx-auto text-lg">
              Four stages transform an intractable game into near-optimal strategy
            </p>
          </div>
        </FadeUp>

        <div className="space-y-2">
          {steps.map((step, i) => (
            <FadeUp key={step.number} delay={i * 0.1}>
              <div className="group relative glass-hover rounded-2xl p-8 md:p-10">
                {/* Step number */}
                <div className="flex items-start gap-8">
                  <div className="shrink-0">
                    <span
                      className={`font-mono text-5xl font-black bg-gradient-to-b ${step.color} bg-clip-text text-transparent opacity-30 group-hover:opacity-60 transition-opacity duration-500`}
                    >
                      {step.number}
                    </span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-3 mb-2">
                      <h3 className="text-xl font-bold text-white">
                        {step.title}
                      </h3>
                      <span className="text-sm text-graphite-500">
                        {step.subtitle}
                      </span>
                    </div>
                    <p className="text-graphite-400 leading-relaxed mb-4 max-w-2xl">
                      {step.description}
                    </p>
                    <div className="inline-block">
                      <code className="text-xs font-mono px-3 py-1.5 rounded-md bg-graphite-800/80 text-graphite-300 border border-graphite-700/50">
                        {step.detail}
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </FadeUp>
          ))}
        </div>
      </div>
    </section>
  );
}

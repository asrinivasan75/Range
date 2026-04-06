"use client";

import { FadeUp } from "@/components/ui/motion";
import Link from "next/link";

export function CTA() {
  return (
    <section className="relative py-32 px-6">
      <div className="glow-line max-w-lg mx-auto mb-32" />

      <div className="max-w-3xl mx-auto text-center">
        <FadeUp>
          <h2 className="text-display-sm font-bold text-white mb-6">
            Start Training
          </h2>
          <p className="text-graphite-400 text-lg mb-10 max-w-xl mx-auto">
            Launch a training run in seconds. Watch strategies converge in real-time.
            Explore the mathematics of optimal play.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/dashboard"
              className="px-8 py-3.5 bg-white text-graphite-950 rounded-lg font-semibold text-sm tracking-wide hover:bg-graphite-100 transition-colors duration-200"
            >
              Open Dashboard
            </Link>
            <Link
              href="/dashboard/runs"
              className="px-8 py-3.5 rounded-lg font-medium text-sm tracking-wide text-graphite-300 border border-graphite-700/50 hover:border-graphite-600 hover:text-white transition-all duration-200"
            >
              View Training Runs
            </Link>
          </div>
        </FadeUp>

        {/* Footer */}
        <FadeUp delay={0.2}>
          <div className="mt-32 pt-8 border-t border-graphite-800/50">
            <div className="flex items-center justify-center gap-6 text-xs text-graphite-500">
              <span>Range v0.1.0</span>
              <span className="w-1 h-1 rounded-full bg-graphite-700" />
              <span>Python + NumPy + FastAPI</span>
              <span className="w-1 h-1 rounded-full bg-graphite-700" />
              <span>Next.js + TypeScript</span>
            </div>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}

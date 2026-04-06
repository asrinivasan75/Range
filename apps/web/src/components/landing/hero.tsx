"use client";

import { motion } from "framer-motion";
import { GridBackground } from "@/components/ui/motion";
import Link from "next/link";

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <GridBackground />

      {/* Radial glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-accent-blue/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] bg-accent-cyan/3 rounded-full blur-[100px] pointer-events-none" />

      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-graphite-700/50 bg-graphite-900/50 backdrop-blur-sm mb-8"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald animate-pulse" />
          <span className="text-xs font-medium text-graphite-300 tracking-wide uppercase">
            Game-Theoretic Poker Analysis
          </span>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: [0.25, 0.4, 0.25, 1] }}
          className="text-display-xl font-bold tracking-tight mb-6"
        >
          <span className="gradient-text">Range</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="text-xl md:text-2xl text-graphite-300 max-w-2xl mx-auto mb-4 font-light leading-relaxed"
        >
          Train near-optimal poker strategies through
          <br className="hidden sm:inline" />{" "}
          <span className="text-white font-medium">counterfactual regret minimization</span>
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.65 }}
          className="text-base text-graphite-400 max-w-xl mx-auto mb-12"
        >
          An open solver platform for No-Limit Hold&apos;em. Abstract. Train. Analyze. Iterate.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Link
            href="/dashboard"
            className="group relative px-8 py-3.5 bg-white text-graphite-950 rounded-lg font-semibold text-sm tracking-wide hover:bg-graphite-100 transition-all duration-300 overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-accent-blue to-accent-cyan opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className="relative z-10 group-hover:text-white transition-colors">
              Open Dashboard
            </span>
          </Link>
          <a
            href="#how-it-works"
            className="px-8 py-3.5 rounded-lg font-medium text-sm tracking-wide text-graphite-300 border border-graphite-700/50 hover:border-graphite-600 hover:text-white transition-all duration-300"
          >
            How It Works
          </a>
        </motion.div>

        {/* Stats bar */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="mt-20 grid grid-cols-3 gap-8 max-w-lg mx-auto"
        >
          {[
            { value: "CFR", label: "Algorithm" },
            { value: "2P", label: "Heads-Up" },
            { value: "NLHE", label: "Game Type" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="font-mono text-lg font-bold text-white">
                {stat.value}
              </div>
              <div className="text-xs text-graphite-500 uppercase tracking-wider mt-1">
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-graphite-950 to-transparent pointer-events-none" />

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="w-5 h-8 rounded-full border border-graphite-600 flex items-start justify-center pt-1.5"
        >
          <div className="w-1 h-2 rounded-full bg-graphite-400" />
        </motion.div>
      </motion.div>
    </section>
  );
}

"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const FadeIn = ({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay }} className={className}>
    {children}
  </motion.div>
);

function Arrow({ direction = "down", label = "" }: { direction?: "down" | "right"; label?: string }) {
  if (direction === "right") {
    return (
      <div className="flex items-center gap-1 px-2">
        <div className="w-8 h-0.5 bg-accent-blue/40" />
        <div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-transparent border-l-accent-blue/40" />
        {label && <span className="text-[9px] text-graphite-500 ml-1">{label}</span>}
      </div>
    );
  }
  return (
    <div className="flex flex-col items-center gap-0.5 py-1">
      <div className="w-0.5 h-6 bg-accent-blue/40" />
      <div className="w-0 h-0 border-l-4 border-r-4 border-t-6 border-transparent border-t-accent-blue/40" />
      {label && <span className="text-[9px] text-graphite-500">{label}</span>}
    </div>
  );
}

function Box({ title, items, color = "blue", wide = false }: { title: string; items: string[]; color?: string; wide?: boolean }) {
  const colors: Record<string, string> = {
    blue: "border-accent-blue/30 bg-accent-blue/5",
    emerald: "border-accent-emerald/30 bg-accent-emerald/5",
    amber: "border-accent-amber/30 bg-accent-amber/5",
    rose: "border-accent-rose/30 bg-accent-rose/5",
    cyan: "border-accent-cyan/30 bg-accent-cyan/5",
    purple: "border-purple-500/30 bg-purple-500/5",
  };
  const dotColors: Record<string, string> = {
    blue: "bg-accent-blue", emerald: "bg-accent-emerald", amber: "bg-accent-amber",
    rose: "bg-accent-rose", cyan: "bg-accent-cyan", purple: "bg-purple-500",
  };
  return (
    <div className={cn("rounded-xl border p-4", colors[color], wide ? "col-span-2" : "")}>
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("w-2 h-2 rounded-full", dotColors[color])} />
        <h3 className="text-sm font-bold text-white">{title}</h3>
      </div>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-graphite-400 flex items-start gap-1.5">
            <span className="text-graphite-600 mt-0.5">·</span>{item}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ArchitecturePage() {
  return (
    <div className="space-y-10 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">How It Works</h1>
        <p className="text-sm text-graphite-400 mt-1">Complete flow from game state to bot decision to learning</p>
      </div>

      {/* ── Q-LEARNING FLOW ── */}
      <FadeIn delay={0.1}>
        <div className="card space-y-4">
          <h2 className="text-lg font-bold text-white">Q-Learning Agent Flow</h2>
          <p className="text-xs text-graphite-500">72 parameters · 12 features · linear model · trained against Slumbot</p>

          <div className="flex flex-col items-center gap-0">
            <Box color="cyan" title="1. Game State" items={[
              "Your cards: Qd 6d",
              "Board: Kh 8c 5s",
              "Pot: 126, To call: 82",
              "Street: preflop, Position: BTN",
            ]} />
            <Arrow label="extract" />

            <Box color="blue" title="2. Feature Extraction (12 numbers)" items={[
              "equity=0.52, pot_odds=0.39, spr=0.04",
              "street=0.0, position=1.0, bet_ratio=0.65",
              "hand_cat=0.0, has_pair=0, commitment=0.18",
              "aggression=0.75, equity_edge=0.13, combined=0.52",
            ]} />
            <Arrow label="multiply by weights (12×6 matrix)" />

            <Box color="emerald" title="3. Q-Values (one per action)" items={[
              "fold: -0.34  ← features × W[:,0]",
              "check: -0.15  ← features × W[:,1]",
              "call: -0.59  ← features × W[:,2]",
              "bet_small: -0.12  ← features × W[:,3]",
              "bet_medium: -0.18  ← features × W[:,4]",
              "bet_large: +0.03  ← features × W[:,5]  ✓ HIGHEST",
            ]} />
            <Arrow label="pick highest legal Q-value" />

            <Box color="amber" title="4. Action → bet_large" items={[
              "Bot bets pot (100% of pot)",
              "Deterministic — always picks the highest Q-value",
              "5% chance of random action (epsilon exploration)",
            ]} />
            <Arrow label="hand ends, get result" />

            <Box color="rose" title="5. Learning (after hand ends)" items={[
              "Result: Lost 82bb (opponent had AA)",
              "Reward: -82bb → normalized to -0.82",
              "Update: W[:,bet_large] -= learning_rate × error × features",
              "Effect: next time similar features → bet_large Q-value is lower",
              "Over 1000s of hands: weights converge to profitable strategy",
            ]} />
          </div>
        </div>
      </FadeIn>

      {/* ── PPO NEURAL NET FLOW ── */}
      <FadeIn delay={0.2}>
        <div className="card space-y-4">
          <h2 className="text-lg font-bold text-white">PPO Neural Network Flow</h2>
          <p className="text-xs text-graphite-500">~100K parameters · 30 features · mixed strategies · trained via self-play</p>

          <div className="flex flex-col items-center gap-0">
            <Box color="cyan" title="1. Game State (same hand)" items={[
              "Your cards: Qd 6d",
              "Board: Kh 8c 5s",
              "Pot: 126, To call: 82",
              "+ board_wet=0.3, flush_possible=0, has_ace=0",
              "+ blocker info, overcards, connectedness",
            ]} />
            <Arrow label="extract 30 features" />

            <Box color="blue" title="2. Feature Vector (30 numbers)" items={[
              "Same 12 as Q-learning PLUS:",
              "board texture (wet/dry/paired), flush/straight possible",
              "top pair detection, pocket pair, suited, high card rank",
              "overcards, blocker effects, bluff signal, combined strength",
            ]} />
            <Arrow label="feed through neural network" />

            <div className="grid grid-cols-3 gap-3 w-full">
              <Box color="purple" title="3a. Encoder" items={[
                "30 → 256 (+ LayerNorm + GELU)",
                "256 → 256 (residual block)",
                "256 → 128 (residual block)",
                "Learns card interactions",
              ]} />
              <Box color="emerald" title="3b. Actor Head" items={[
                "128 → 64 → 7 actions",
                "Softmax → probabilities",
                "fold: 25%, call: 10%",
                "bet_sm: 15%, bet_lg: 30%",
                "all_in: 20%",
              ]} />
              <Box color="amber" title="3c. Critic Head" items={[
                "128 → 64 → 1 value",
                "Predicts expected winnings",
                "Value: -12.5 (expects to lose)",
                "Used for advantage calculation",
              ]} />
            </div>
            <Arrow label="SAMPLE from probability distribution (not greedy)" />

            <Box color="amber" title="4. Action → fold (sampled at 25%)" items={[
              "Key difference: SAMPLES from distribution, doesn't just pick max",
              "This means it naturally bluffs sometimes (bet 30%) and folds sometimes (25%)",
              "Mixed strategy = unexploitable by opponents",
              "Illegal actions masked to probability 0 before sampling",
            ]} />
            <Arrow label="buffer fills (1024 transitions), then PPO update" />

            <Box color="rose" title="5. PPO Learning (batch update)" items={[
              "Collect 1024 transitions (states, actions, rewards, probabilities)",
              "For each transition:",
              "  ratio = new_prob / old_prob (how much did policy change?)",
              "  advantage = actual_reward - critic_prediction (was it better than expected?)",
              "  loss = -min(ratio × advantage, clipped_ratio × advantage)",
              "  + value_loss (critic learns to predict better)",
              "  + entropy_bonus (keeps mixed strategies, prevents collapsing to one action)",
              "Clip ratio to [0.8, 1.2] — prevents catastrophic updates",
              "Run 4 epochs over the batch, then clear buffer",
            ]} />
          </div>
        </div>
      </FadeIn>

      {/* ── RANGE ESTIMATION ── */}
      <FadeIn delay={0.3}>
        <div className="card space-y-4">
          <h2 className="text-lg font-bold text-white">Range Estimation (New)</h2>
          <p className="text-xs text-graphite-500">Replaces "equity vs random" with "equity vs estimated opponent range"</p>

          <div className="flex flex-col items-center gap-0">
            <Box color="cyan" title="1. Observe Opponent Actions" items={[
              "Opponent opens → 3-bets → 5-bet shoves all-in preflop",
              "This is a very aggressive action sequence",
            ]} />
            <Arrow label="classify action sequence" />

            <Box color="purple" title="2. Map to Range Category" items={[
              "5-bet shove → GTO prior says: AA (16%), KK (16%), QQ (16%), JJ (16%)",
              "Plus some AKs (2%), AQs (2%), small % of bluffs",
              "Output: probability distribution over 169 starting hands",
            ]} />
            <Arrow label="compute equity AGAINST this range" />

            <div className="grid grid-cols-2 gap-3 w-full">
              <Box color="rose" title="OLD: Equity vs Random" items={[
                "Q6s vs random hand = 52%",
                "Pot odds = 39%",
                "52% > 39% → looks profitable!",
                "Bot calls → LOSES to AA/KK",
              ]} />
              <Box color="emerald" title="NEW: Equity vs Range" items={[
                "Q6s vs {AA,KK,QQ,JJ,AK} = 32%",
                "Pot odds = 39%",
                "32% < 39% → unprofitable!",
                "Bot FOLDS → correct decision",
              ]} />
            </div>
          </div>
        </div>
      </FadeIn>

      {/* ── REWARD CALCULATION ── */}
      <FadeIn delay={0.4}>
        <div className="card space-y-4">
          <h2 className="text-lg font-bold text-white">Reward Calculation</h2>
          <p className="text-xs text-graphite-500">How the agent knows whether a decision was good or bad</p>

          <div className="space-y-3">
            <Box color="amber" wide title="End of Hand → Reward Signal" items={[
              "Win 50bb → reward = +50 (normalized: +0.50)",
              "Lose 100bb → reward = -100 (normalized: -1.00)",
              "Fold preflop → reward = -1 (lost the blind)",
              "Opponent folds to our bet → reward = pot won",
            ]} />

            <Box color="blue" wide title="Discount Through the Hand" items={[
              "Each decision in the hand gets a discounted version of the final reward",
              "Last action (river call): discount = 0.99^0 = 1.00 × reward",
              "Second-to-last (turn bet): discount = 0.99^1 = 0.99 × reward",
              "First action (preflop raise): discount = 0.99^5 = 0.95 × reward",
              "Earlier decisions get slightly less credit/blame for the outcome",
            ]} />

            <Box color="emerald" wide title="Weight Update Direction" items={[
              "Positive reward → actions taken become MORE likely in similar states",
              "Negative reward → actions taken become LESS likely in similar states",
              "Example: called all-in with Q6s and lost → next time Q6s in similar spot → fold Q-value increases",
              "Example: bluffed river and won → next time weak hand on river → bet probability increases",
              "Over 1000s of hands: the weights/network converge to the strategy that maximizes long-term reward",
            ]} />

            <Box color="rose" wide title="The Fundamental Problem" items={[
              "Reward is NOISY — you can make the right call and still lose (bad luck)",
              "You can make the wrong call and still win (good luck)",
              "Only over MANY hands does the signal overcome the noise",
              "This is why 50K hands with 23 updates wasn't enough for PPO",
              "And why Q-learning's 72 parameters learn faster (less to learn, converges quicker)",
            ]} />
          </div>
        </div>
      </FadeIn>
    </div>
  );
}

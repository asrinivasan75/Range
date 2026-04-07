"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const API = "";

const SUIT_DISPLAY: Record<string, { sym: string; color: string }> = {
  h: { sym: "\u2665", color: "text-red-400" },
  d: { sym: "\u2666", color: "text-blue-400" },
  c: { sym: "\u2663", color: "text-emerald-400" },
  s: { sym: "\u2660", color: "text-graphite-200" },
};

interface HandLog {
  hand_number: number;
  our_cards: string[];
  our_position: string;
  client_pos: number;
  actions: { who: string; action: string; board: string[]; street: number; full_action?: string }[];
  board_flop: string[];
  board_turn: string | null;
  board_river: string | null;
  final_action_string: string;
  winnings: number;
  winnings_bb: number;
}

interface LogData {
  summary: {
    hands_played: number;
    total_bb: number;
    bb_per_100: number;
    win_count: number;
    loss_count: number;
    win_pct: number;
    running_total: number[];
    biggest_win: number;
    biggest_loss: number;
  };
  hands: HandLog[];
  config: { opponent: string; blinds: string; stack_bb: number; timestamp: string };
}

interface LogMeta { id: string; label: string; hands: number; total_bb: number; bb_per_100: number; timestamp: string }

// ── Card components ─────────────────────────────────────────

function CardView({ card, size = "md" }: { card: string; size?: "sm" | "md" }) {
  if (!card) return null;
  const sizes = { sm: "w-10 h-14 text-sm", md: "w-12 h-[68px] text-base" };
  const rank = card[0];
  const suit = card[1];
  const s = SUIT_DISPLAY[suit] || { sym: suit, color: "text-white" };
  return (
    <div className={cn(sizes[size], "rounded-lg bg-graphite-900 border border-graphite-700 flex flex-col items-center justify-center gap-0 shadow-md")}>
      <span className={cn("font-bold font-mono leading-tight", s.color)}>{rank}</span>
      <span className={cn("text-xs leading-tight", s.color)}>{s.sym}</span>
    </div>
  );
}

function HiddenCard({ size = "md" }: { size?: "sm" | "md" }) {
  const sizes = { sm: "w-10 h-14", md: "w-12 h-[68px]" };
  return (
    <div className={cn(sizes[size], "rounded-lg bg-gradient-to-br from-blue-900/30 to-cyan-900/20 border border-blue-500/15 flex items-center justify-center")}>
      <span className="text-blue-400/25 text-lg font-mono">?</span>
    </div>
  );
}

function EmptySlot() {
  return <div className="w-12 h-[68px] rounded-lg border border-dashed border-graphite-700/25" />;
}

// ── Parse Slumbot action string into per-street actions ─────

interface ParsedAction {
  who: "us" | "slumbot";
  type: "check" | "call" | "fold" | "bet";
  amount?: number;
  street: number;
}

function parseActionString(actionStr: string, clientPos: number): ParsedAction[] {
  if (!actionStr) return [];
  const result: ParsedAction[] = [];
  const streets = actionStr.split("/");

  for (let sIdx = 0; sIdx < streets.length; sIdx++) {
    const s = streets[sIdx];
    let actionIdx = 0;
    let i = 0;
    while (i < s.length) {
      const c = s[i];
      // Slumbot: action string pos 0 = BTN/SB (acts first preflop)
      // client_pos=0 -> we are BB -> we are odd indices (1, 3, 5...)
      // client_pos=1 -> we are BTN -> we are even indices (0, 2, 4...)
      const isUs = (actionIdx % 2) === (clientPos === 0 ? 1 : 0);

      if (c === 'k') {
        result.push({ who: isUs ? "us" : "slumbot", type: "check", street: sIdx });
        actionIdx++; i++;
      } else if (c === 'c') {
        result.push({ who: isUs ? "us" : "slumbot", type: "call", street: sIdx });
        actionIdx++; i++;
      } else if (c === 'f') {
        result.push({ who: isUs ? "us" : "slumbot", type: "fold", street: sIdx });
        actionIdx++; i++;
      } else if (c === 'b') {
        let j = i + 1;
        while (j < s.length && s[j] >= '0' && s[j] <= '9') j++;
        const amt = parseInt(s.substring(i + 1, j));
        result.push({ who: isUs ? "us" : "slumbot", type: "bet", amount: amt, street: sIdx });
        actionIdx++; i = j;
      } else {
        i++;
      }
    }
  }
  return result;
}

const STREET_NAMES = ["Preflop", "Flop", "Turn", "River"];

// ── Visual Hand Playback ────────────────────────────────────

function HandPlayback({ hand }: { hand: HandLog }) {
  const [step, setStep] = useState(0);

  const actions = useMemo(
    () => parseActionString(hand.final_action_string, hand.client_pos),
    [hand.final_action_string, hand.client_pos]
  );

  // Reset step when hand changes
  useEffect(() => setStep(actions.length), [hand.hand_number, actions.length]);

  // Board visible at each step
  const visibleBoard = useMemo(() => {
    if (step === 0) return [];
    const maxStreet = Math.max(...actions.slice(0, step).map(a => a.street), -1);
    const cards: string[] = [];
    if (maxStreet >= 1 && hand.board_flop) cards.push(...hand.board_flop);
    if (maxStreet >= 2 && hand.board_turn) cards.push(hand.board_turn);
    if (maxStreet >= 3 && hand.board_river) cards.push(hand.board_river);
    return cards;
  }, [step, actions, hand]);

  // Pot at current step (rough estimate from action amounts)
  const currentPot = useMemo(() => {
    let pot = 150; // blinds
    for (let i = 0; i < step && i < actions.length; i++) {
      const a = actions[i];
      if (a.type === "bet" && a.amount) pot = a.amount * 2; // rough: bet means total invested
      if (a.type === "call") pot += 0; // already counted
    }
    return Math.max(pot, 150);
  }, [step, actions]);

  const isShowdown = step >= actions.length && !actions.some(a => a.type === "fold");

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <span className="text-sm font-semibold text-white">Hand #{hand.hand_number}</span>
          <span className="text-xs text-graphite-500 ml-2">{hand.our_position}</span>
          <span className={cn("text-xs font-mono font-bold ml-3",
            hand.winnings_bb > 0 ? "text-emerald-400" : hand.winnings_bb < 0 ? "text-red-400" : "text-graphite-500")}>
            {hand.winnings_bb > 0 ? "+" : ""}{hand.winnings_bb}bb
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setStep(0)} className="px-2 py-1 rounded bg-graphite-800 text-graphite-400 text-xs hover:text-white">|&lt;</button>
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-2 py-1 rounded bg-graphite-800 text-graphite-400 text-xs hover:text-white disabled:opacity-30">&lt;</button>
          <span className="text-xs text-graphite-500 font-mono w-16 text-center">{step}/{actions.length}</span>
          <button onClick={() => setStep(Math.min(actions.length, step + 1))} disabled={step >= actions.length} className="px-2 py-1 rounded bg-graphite-800 text-graphite-400 text-xs hover:text-white disabled:opacity-30">&gt;</button>
          <button onClick={() => setStep(actions.length)} className="px-2 py-1 rounded bg-graphite-800 text-graphite-400 text-xs hover:text-white">&gt;|</button>
        </div>
      </div>

      {/* Visual table */}
      <div className="p-5 rounded-xl bg-gradient-to-b from-graphite-900/80 to-graphite-950/90 border border-graphite-800/30 space-y-3">
        {/* Slumbot (opponent) */}
        <div className="text-center">
          <div className="text-[10px] text-graphite-500 uppercase tracking-widest mb-1">
            Slumbot
            {actions[step - 1]?.who === "slumbot" && step > 0 && (
              <span className="ml-2 text-amber-400 normal-case text-xs">
                {formatAction(actions[step - 1])}
              </span>
            )}
          </div>
          <div className="flex justify-center gap-1.5">
            {isShowdown ? (
              <>
                <div className="w-12 h-[68px] rounded-lg bg-graphite-800 border border-amber-500/20 flex items-center justify-center">
                  <span className="text-amber-400/50 text-xs font-mono">?</span>
                </div>
                <div className="w-12 h-[68px] rounded-lg bg-graphite-800 border border-amber-500/20 flex items-center justify-center">
                  <span className="text-amber-400/50 text-xs font-mono">?</span>
                </div>
              </>
            ) : (
              <><HiddenCard /><HiddenCard /></>
            )}
          </div>
        </div>

        {/* Board */}
        <div className="text-center py-1">
          <div className="flex justify-center gap-1.5 min-h-[68px] items-center">
            <AnimatePresence mode="popLayout">
              {visibleBoard.map((c, i) => (
                <motion.div key={`${c}-${i}`} initial={{ rotateY: 90, opacity: 0 }} animate={{ rotateY: 0, opacity: 1 }} transition={{ duration: 0.3, delay: i * 0.1 }}>
                  <CardView card={c} />
                </motion.div>
              ))}
            </AnimatePresence>
            {Array.from({ length: 5 - visibleBoard.length }).map((_, i) => <EmptySlot key={`e-${i}`} />)}
          </div>
          {step >= actions.length && (
            <div className="mt-2 text-xs font-mono text-graphite-500">
              Pot: {hand.winnings > 0 ? hand.winnings * 2 : Math.abs(hand.winnings) * 2 || 150} chips
            </div>
          )}
        </div>

        {/* Our hand */}
        <div className="text-center">
          <div className="flex justify-center gap-1.5">
            {hand.our_cards.map((c, i) => <CardView key={i} card={c} />)}
          </div>
          <div className="text-[10px] text-graphite-500 uppercase tracking-widest mt-1">
            You ({hand.our_position})
            {actions[step - 1]?.who === "us" && step > 0 && (
              <span className="ml-2 text-blue-400 normal-case text-xs">
                {formatAction(actions[step - 1])}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Action timeline */}
      <div className="space-y-0.5">
        {actions.map((a, i) => {
          const isCurrentStep = i === step - 1;
          const isPast = i < step;
          const streetChanged = i > 0 && a.street !== actions[i - 1].street;
          return (
            <div key={i}>
              {streetChanged && (
                <div className="text-[10px] text-graphite-600 uppercase tracking-wider py-1 border-t border-graphite-800/40 mt-1">
                  {STREET_NAMES[a.street]}
                  {a.street === 1 && hand.board_flop && (
                    <span className="text-graphite-400 normal-case ml-2">{hand.board_flop.join(" ")}</span>
                  )}
                  {a.street === 2 && hand.board_turn && (
                    <span className="text-graphite-400 normal-case ml-2">{hand.board_turn}</span>
                  )}
                  {a.street === 3 && hand.board_river && (
                    <span className="text-graphite-400 normal-case ml-2">{hand.board_river}</span>
                  )}
                </div>
              )}
              <button onClick={() => setStep(i + 1)}
                className={cn("w-full flex items-center gap-2 px-2 py-1 rounded text-xs transition-colors text-left",
                  isCurrentStep ? "bg-accent-blue/10 border border-accent-blue/20" :
                  isPast ? "text-graphite-300" : "text-graphite-600")}>
                <span className="w-4 text-right font-mono text-graphite-600">{i + 1}</span>
                <span className={cn("font-medium", a.who === "us" ? "text-blue-400" : "text-amber-400")}>
                  {a.who === "us" ? "You" : "Bot"}
                </span>
                <span className="text-graphite-400">{formatAction(a)}</span>
              </button>
            </div>
          );
        })}
      </div>

      {/* Full action string */}
      <div className="p-2 rounded bg-graphite-900/60 border border-graphite-800/30">
        <code className="text-[11px] font-mono text-graphite-500 break-all">{hand.final_action_string}</code>
      </div>

      {/* Result */}
      <div className={cn("p-3 rounded-lg border text-center",
        hand.winnings_bb > 0 ? "bg-emerald-500/5 border-emerald-500/20" :
        hand.winnings_bb < 0 ? "bg-red-500/5 border-red-500/20" :
        "bg-graphite-800/30 border-graphite-700/30")}>
        <span className={cn("font-mono font-bold text-lg",
          hand.winnings_bb > 0 ? "text-emerald-400" : hand.winnings_bb < 0 ? "text-red-400" : "text-graphite-400")}>
          {hand.winnings_bb > 0 ? "+" : ""}{hand.winnings_bb}bb
        </span>
      </div>
    </div>
  );
}

function formatAction(a: ParsedAction): string {
  if (a.type === "check") return "checks";
  if (a.type === "call") return "calls";
  if (a.type === "fold") return "folds";
  if (a.type === "bet" && a.amount) return `bets ${a.amount}`;
  return a.type;
}

// ── Stat card ───────────────────────────────────────────────

function StatCard({ label, value, color = "text-white" }: { label: string; value: string; color?: string }) {
  return (
    <div className="card py-3">
      <div className="text-[10px] uppercase text-graphite-500 mb-1">{label}</div>
      <div className={cn("font-mono text-lg font-bold", color)}>{value}</div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────

export default function ReplayerPage() {
  const [logs, setLogs] = useState<LogMeta[]>([]);
  const [activeLog, setActiveLog] = useState("");
  const [data, setData] = useState<LogData | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [filter, setFilter] = useState<"all" | "wins" | "losses" | "big">("all");
  const [loading, setLoading] = useState(true);

  // Load available logs
  useEffect(() => {
    fetch(`${API}/api/slumbot-logs`).then(r => r.json()).then(d => {
      setLogs(d.logs || []);
      if (d.logs?.length > 0) {
        const best = d.logs.find((l: LogMeta) => l.id === "rl_v2")
          || d.logs.find((l: LogMeta) => l.id === "rl")
          || d.logs[0];
        setActiveLog(best.id);
      }
    }).catch(() => {});
  }, []);

  // Load selected log data
  useEffect(() => {
    if (!activeLog) return;
    setLoading(true);
    fetch(`${API}/api/slumbot-log?mode=${activeLog}`)
      .then(r => r.json())
      .then(d => { setData(d); setSelectedIdx(0); setLoading(false); })
      .catch(() => setLoading(false));
  }, [activeLog]);

  const filteredHands = useMemo(() => {
    if (!data?.hands) return [];
    switch (filter) {
      case "wins": return data.hands.filter(h => h.winnings_bb > 0);
      case "losses": return data.hands.filter(h => h.winnings_bb < 0);
      case "big": return data.hands.filter(h => Math.abs(h.winnings_bb) >= 10);
      default: return data.hands;
    }
  }, [data, filter]);

  const chartData = useMemo(() => {
    if (!data?.summary?.running_total) return [];
    return data.summary.running_total.map((val, i) => ({ hand: i + 1, bb: val }));
  }, [data]);

  const hand = filteredHands[selectedIdx];

  if (loading && !data) return <div className="text-graphite-500 py-12 text-center">Loading...</div>;
  if (!data || !data.hands?.length) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-white">Hand Replayer</h1>
        <div className="card text-center py-16">
          <p className="text-graphite-400 mb-2">No session data found.</p>
          <p className="text-xs text-graphite-500">Run <code className="text-accent-blue">./scripts/slumbot.sh username password 500</code></p>
        </div>
      </div>
    );
  }

  const s = data.summary;

  return (
    <div className="space-y-6">
      {/* Header + log switcher */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Hand Replayer</h1>
          <p className="text-sm text-graphite-400 mt-1">
            {s.hands_played || data.hands.length} hands vs {data.config?.opponent || "Slumbot"} — {data.config?.timestamp}
          </p>
        </div>
        {logs.length > 0 && (
          <select
            value={activeLog}
            onChange={e => setActiveLog(e.target.value)}
            className="px-3 py-2 rounded-lg bg-graphite-900 border border-graphite-700 text-sm text-white min-w-[250px]"
          >
            {logs.map(log => (
              <option key={log.id} value={log.id}>
                {log.label} — {log.hands} hands ({log.total_bb >= 0 ? "+" : ""}{log.total_bb}bb, {log.bb_per_100 >= 0 ? "+" : ""}{log.bb_per_100} bb/100)
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <StatCard label="Total" value={`${s.total_bb >= 0 ? "+" : ""}${s.total_bb}bb`}
          color={s.total_bb >= 0 ? "text-emerald-400" : "text-red-400"} />
        <StatCard label="bb/100" value={`${s.bb_per_100 >= 0 ? "+" : ""}${s.bb_per_100}`}
          color={s.bb_per_100 >= 0 ? "text-emerald-400" : "text-red-400"} />
        <StatCard label="Win Rate" value={`${s.win_pct}%`} />
        <StatCard label="Hands" value={`${s.hands_played || data.hands.length}`} />
        <StatCard label="Best" value={`+${s.biggest_win}bb`} color="text-emerald-400" />
        <StatCard label="Worst" value={`${s.biggest_loss}bb`} color="text-red-400" />
      </div>

      {/* Profit curve */}
      {chartData.length > 0 && (
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-3">Profit Curve</h2>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={s.total_bb >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={s.total_bb >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#2a2a30" strokeDasharray="3 3" />
                <XAxis dataKey="hand" stroke="#6b6b78" fontSize={10} />
                <YAxis stroke="#6b6b78" fontSize={10} />
                <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #333", borderRadius: "8px", fontSize: "12px" }} />
                <Area type="monotone" dataKey="bb" stroke={s.total_bb >= 0 ? "#10b981" : "#ef4444"}
                  fill="url(#pg)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Hand list + visual playback */}
      <div className="grid lg:grid-cols-5 gap-4">
        {/* Hand list */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-white">Hands</h2>
            <select value={filter} onChange={e => { setFilter(e.target.value as any); setSelectedIdx(0); }}
              className="text-xs bg-graphite-800 border border-graphite-700 rounded px-2 py-1 text-white">
              <option value="all">All ({data.hands.length})</option>
              <option value="wins">Wins</option>
              <option value="losses">Losses</option>
              <option value="big">Big (10bb+)</option>
            </select>
          </div>
          <div className="space-y-0.5 max-h-[500px] overflow-y-auto">
            {filteredHands.map((h, i) => (
              <button key={h.hand_number} onClick={() => setSelectedIdx(i)}
                className={cn("w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs transition-colors",
                  selectedIdx === i ? "bg-graphite-800 text-white" : "text-graphite-400 hover:bg-graphite-800/50")}>
                <span className="font-mono flex items-center gap-1.5">
                  <span className="text-graphite-600 w-6 text-right">#{h.hand_number}</span>
                  {h.our_cards.map((c, ci) => {
                    const suit = c[1];
                    const sd = SUIT_DISPLAY[suit];
                    return <span key={ci} className={cn("font-bold", sd?.color || "text-white")}>{c[0]}{sd?.sym}</span>;
                  })}
                </span>
                <span className="flex items-center gap-2">
                  <span className="text-graphite-600">{h.our_position}</span>
                  <span className={cn("font-mono font-bold w-14 text-right",
                    h.winnings_bb > 0 ? "text-emerald-400" : h.winnings_bb < 0 ? "text-red-400" : "text-graphite-600")}>
                    {h.winnings_bb > 0 ? "+" : ""}{h.winnings_bb}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Hand playback */}
        <div className="card lg:col-span-3">
          {hand ? <HandPlayback hand={hand} /> : (
            <p className="text-graphite-500 text-sm py-12 text-center">Select a hand</p>
          )}
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

const API = "";

const SUIT_DISPLAY: Record<string, { sym: string; color: string }> = {
  h: { sym: "\u2665", color: "text-red-400" },
  d: { sym: "\u2666", color: "text-blue-400" },
  c: { sym: "\u2663", color: "text-emerald-400" },
  s: { sym: "\u2660", color: "text-graphite-200" },
};

const ACTION_COLORS: Record<string, string> = {
  fold: "bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/20",
  check: "bg-graphite-800 text-white border-graphite-700 hover:bg-graphite-700",
  call: "bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20",
  bet: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20",
  raise: "bg-amber-500/10 text-amber-400 border-amber-500/20 hover:bg-amber-500/20",
  all_in: "bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20",
};

interface Run { id: string; name: string; config: { game_type: string; n_iterations?: number } }

interface GameAction {
  type: string;
  label: string;
  amount?: number;
  min_amount?: number;
  max_amount?: number;
}

interface SizingSuggestion {
  name: string;
  amount: number;
  fraction: number;
  rationale: string;
}

interface HandResponse {
  hand_id: string;
  street: string;
  pot: number;
  player_stack: number;
  bot_stack: number;
  player_bet: number;
  bot_bet: number;
  player_cards: string[];
  bot_cards: string[];
  board: string[];
  legal_actions: GameAction[];
  sizing_suggestions: SizingSuggestion[];
  is_player_turn: boolean;
  is_terminal: boolean;
  winner: string | null;
  message: string;
  bot_action: { label: string; sizing?: string; rationale?: string; source?: string; q_values?: Record<string, number> } | null;
  action_log: string[];
  session_id: string;
}

interface Advice {
  equity_pct: number;
  chen_score: number;
  preflop_bucket: number;
  spr: number;
  effective_stack: number;
  recommended: string;
  sizing_suggestions: SizingSuggestion[];
}

function CardView({ card, hidden = false, size = "md" }: { card: string; hidden?: boolean; size?: "sm" | "md" | "lg" }) {
  const sizes = { sm: "w-10 h-14 text-sm", md: "w-14 h-20 text-lg", lg: "w-16 h-24 text-xl" };
  if (hidden || card === "??") {
    return (
      <div className={cn(sizes[size], "rounded-lg bg-gradient-to-br from-blue-900/40 to-cyan-900/30 border border-blue-500/20 flex items-center justify-center")}>
        <span className="text-blue-400/30 text-2xl font-mono">?</span>
      </div>
    );
  }
  const rank = card[0];
  const suit = card[1];
  const s = SUIT_DISPLAY[suit] || { sym: suit, color: "text-white" };
  return (
    <motion.div initial={{ rotateY: 90, opacity: 0 }} animate={{ rotateY: 0, opacity: 1 }} transition={{ duration: 0.3 }}
      className={cn(sizes[size], "rounded-lg bg-graphite-900 border border-graphite-700 flex flex-col items-center justify-center gap-0.5 shadow-lg")}>
      <span className={cn("font-bold font-mono", s.color)}>{rank}</span>
      <span className={cn("text-sm", s.color)}>{s.sym}</span>
    </motion.div>
  );
}

function ChipStack({ amount, side }: { amount: number; side: "player" | "bot" }) {
  if (amount <= 0) return null;
  const chipCount = Math.min(Math.ceil(amount / 3), 8);
  return (
    <motion.div initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
      className={cn("flex flex-col items-center gap-0.5", side === "bot" ? "mb-1" : "mt-1")}>
      <div className="flex items-end gap-0.5">
        {Array.from({ length: chipCount }).map((_, i) => (
          <div key={i} className={cn("w-4 h-1.5 rounded-full", side === "player" ? "bg-blue-400" : "bg-amber-400")}
            style={{ opacity: 0.4 + (i / chipCount) * 0.6 }} />
        ))}
      </div>
      <span className={cn("font-mono text-xs font-bold", side === "player" ? "text-blue-400" : "text-amber-400")}>
        {amount.toFixed(1)}
      </span>
    </motion.div>
  );
}

export default function PlayPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState("");
  const [game, setGame] = useState<HandResponse | null>(null);
  const [advice, setAdvice] = useState<Advice | null>(null);
  const [showAdvisor, setShowAdvisor] = useState(true);
  const [loading, setLoading] = useState(false);
  const [raiseAmount, setRaiseAmount] = useState(0);
  const [showRaiseSlider, setShowRaiseSlider] = useState(false);
  const [bots, setBots] = useState<{id: string; label: string; type: string; weights: string}[]>([]);
  const [selectedBot, setSelectedBot] = useState("heuristic");
  const [sessionId, setSessionId] = useState("");
  const [handsPlayed, setHandsPlayed] = useState(0);
  const [sessionBB, setSessionBB] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load available bots
    fetch(`${API}/api/play/bots`).then(r => r.json()).then(d => {
      setBots(d.bots || []);
      // Default to first QL agent if available, else heuristic
      const ql = (d.bots || []).find((b: any) => b.type === "ql" && b.label.includes("v5"));
      if (ql) setSelectedBot(ql.id);
    }).catch(() => {});

    // Load CFR runs
    fetch(`${API}/api/runs`).then(r => r.json()).then(d => {
      const valid = (d.runs || []).filter((r: any) => r.status === "completed" && r.config?.game_type !== "kuhn");
      valid.sort((a: any, b: any) => (b.config?.n_iterations || 0) - (a.config?.n_iterations || 0));
      setRuns(valid);
      if (valid.length > 0) setRunId(valid[0].id);
    }).catch(() => {});
  }, []);

  useEffect(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight); }, [game?.action_log]);

  const fetchAdvice = useCallback(async (handId: string) => {
    if (!showAdvisor) return;
    try {
      const r = await fetch(`${API}/api/play/advisor`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hand_id: handId }),
      });
      if (r.ok) setAdvice(await r.json());
    } catch { }
  }, [showAdvisor]);

  const dealHand = useCallback(async () => {
    if (!runId) return;
    setLoading(true);
    setShowRaiseSlider(false);
    try {
      const bot = bots.find(b => b.id === selectedBot);
      const r = await fetch(`${API}/api/play/new-hand`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          bot_type: bot?.type || "heuristic",
          weights_path: bot?.weights || "",
          session_id: sessionId,
        }),
      });
      const d: HandResponse = await r.json();
      setGame(d);
      setAdvice(null);
      if (d.session_id && !sessionId) setSessionId(d.session_id);
      if (d.is_player_turn) fetchAdvice(d.hand_id);
    } finally { setLoading(false); }
  }, [runId, fetchAdvice, selectedBot, bots, sessionId]);

  const act = useCallback(async (actionType: string, amount: number = 0) => {
    if (!game || game.is_terminal || !game.is_player_turn) return;
    setLoading(true);
    setShowRaiseSlider(false);
    try {
      const r = await fetch(`${API}/api/play/act`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hand_id: game.hand_id, action_type: actionType, amount }),
      });
      const d: HandResponse = await r.json();
      setGame(d);
      if (d.session_id && !sessionId) setSessionId(d.session_id);
      if (d.is_player_turn && !d.is_terminal) fetchAdvice(d.hand_id);
      else setAdvice(null);

      // Track session stats when hand completes
      if (d.is_terminal && d.winner) {
        setHandsPlayed(p => p + 1);
        // Fetch session stats from server
        if (d.session_id) {
          fetch(`${API}/api/play/sessions`).then(r => r.json()).then(s => {
            const sess = (s.sessions || []).find((x: any) => x.id === d.session_id);
            if (sess) {
              setHandsPlayed(sess.hands);
              setSessionBB(sess.total_bb);
            }
          }).catch(() => {});
        }
      }
    } finally { setLoading(false); }
  }, [game, fetchAdvice, sessionId]);

  // Find bet/raise action for slider
  const betRaiseAction = (game?.legal_actions || []).find(a => a.type === "bet" || a.type === "raise");
  const hasBetRaise = !!betRaiseAction;

  const openRaiseSlider = useCallback(() => {
    if (betRaiseAction) {
      setRaiseAmount(game?.sizing_suggestions?.[0]?.amount || betRaiseAction.min_amount || 0);
      setShowRaiseSlider(true);
    }
  }, [betRaiseAction, game]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Play vs Bot</h1>
          <p className="text-sm text-graphite-400 mt-1">Heads-up against the solver — proper NLHE rules</p>
        </div>
        <label className="flex items-center gap-2 text-xs text-graphite-400 cursor-pointer select-none">
          <input type="checkbox" checked={showAdvisor} onChange={e => setShowAdvisor(e.target.checked)} className="rounded" />
          Show Advisor
        </label>
      </div>

      {/* Controls */}
      {/* Session + Bot controls */}
      <div className="card p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs text-graphite-500">Bot:</span>
          <select value={selectedBot} onChange={e => { setSelectedBot(e.target.value); }}
            className="px-3 py-1.5 rounded-lg bg-graphite-900 border border-graphite-800 text-sm text-white max-w-[280px]">
            {bots.map(b => <option key={b.id} value={b.id}>{b.label}</option>)}
          </select>
          <button onClick={() => { setSessionId(""); setHandsPlayed(0); setSessionBB(0); setGame(null); }}
            className="px-4 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium hover:bg-emerald-500/20 transition-colors">
            New Session
          </button>
          <button onClick={dealHand} disabled={loading}
            className="px-5 py-1.5 rounded-lg bg-accent-blue text-white text-sm font-medium hover:bg-accent-blue/90 transition-colors disabled:opacity-50">
            {!game ? "Deal Hand" : "New Hand"}
          </button>
          {game && (
            <span className="text-xs text-graphite-500 ml-auto font-mono">
              {game.is_terminal ? "COMPLETE" : (game.street || "").toUpperCase()}
              {!game.is_terminal && ` | Stacks: ${(game.player_stack ?? 0).toFixed(0)} / ${(game.bot_stack ?? 0).toFixed(0)}`}
            </span>
          )}
        </div>
        {sessionId && (
          <div className="flex items-center justify-between p-2 rounded-lg bg-graphite-800/40 border border-graphite-700/30">
            <div className="flex items-center gap-4 text-xs">
              <span className="text-graphite-400">Session <span className="font-mono text-graphite-300">{sessionId}</span></span>
              <span className="text-graphite-500">Hands: <span className="font-mono text-white">{handsPlayed}</span></span>
              <span className="text-graphite-500">Result: <span className={cn("font-mono font-bold", sessionBB >= 0 ? "text-emerald-400" : "text-red-400")}>{sessionBB >= 0 ? "+" : ""}{sessionBB.toFixed(1)}bb</span></span>
            </div>
            <a href="/dashboard/replayer" target="_blank"
              className="text-xs text-accent-blue hover:text-accent-cyan transition-colors">
              View in Replayer →
            </a>
          </div>
        )}
      </div>

      {/* ── TABLE ─────────────────────────────────────────── */}
      <div className="card relative overflow-hidden min-h-[520px]">
        <div className="absolute inset-0 bg-gradient-to-b from-graphite-900/80 to-graphite-950/90 rounded-xl" />
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-20 rounded-xl" />

        {!game ? (
          <div className="relative z-10 flex items-center justify-center h-[480px]">
            <p className="text-graphite-500 text-sm">Select a strategy and deal a hand.</p>
          </div>
        ) : (
          <div className="relative z-10 flex flex-col items-center py-6 gap-3">
            {/* Bot */}
            <div className="text-center">
              <div className="text-[10px] text-graphite-500 uppercase tracking-widest mb-1">
                Bot
                <span className="ml-2 text-graphite-600 normal-case">{(game.bot_stack ?? 0).toFixed(0)} chips</span>
                {game.bot_action && !game.is_terminal && (
                  <span className="ml-2 text-amber-400 normal-case text-xs font-medium">{game.bot_action.label}</span>
                )}
              </div>
              <div className="flex justify-center gap-2">
                {game.bot_cards.map((c, i) => <CardView key={`bot-${i}-${c}`} card={c} size="md" />)}
              </div>
            </div>

            <AnimatePresence>{game.bot_bet > 0 && <ChipStack amount={game.bot_bet} side="bot" />}</AnimatePresence>

            {/* Board + Pot */}
            <div className="text-center py-2">
              <div className="flex justify-center gap-2 min-h-[80px] items-center">
                <AnimatePresence mode="popLayout">
                  {game.board.map((c, i) => <CardView key={`board-${i}-${c}`} card={c} size="md" />)}
                </AnimatePresence>
                {game.board.length < 5 && Array.from({ length: 5 - game.board.length }).map((_, i) => (
                  <div key={`e-${i}`} className="w-14 h-20 rounded-lg border border-dashed border-graphite-700/30" />
                ))}
              </div>
              <div className="mt-3 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-graphite-800/80 border border-graphite-700/50">
                <span className="text-[10px] uppercase tracking-wider text-graphite-400">Pot</span>
                <span className="font-mono text-lg text-white font-bold">{game.pot}</span>
              </div>
            </div>

            <AnimatePresence>{game.player_bet > 0 && <ChipStack amount={game.player_bet} side="player" />}</AnimatePresence>

            {/* Player */}
            <div className="text-center">
              <div className="flex justify-center gap-2">
                {game.player_cards.map((c, i) => <CardView key={`player-${i}-${c}`} card={c} size="lg" />)}
              </div>
              <div className="text-[10px] text-graphite-500 uppercase tracking-widest mt-1">
                You <span className="text-graphite-600 normal-case">{(game.player_stack ?? 0).toFixed(0)} chips</span>
              </div>
            </div>

            {/* Actions */}
            {game.is_player_turn && !game.is_terminal && (
              <div className="space-y-3 mt-2 w-full max-w-lg mx-auto">
                {/* Main action buttons */}
                <div className="flex justify-center gap-2 flex-wrap">
                  {game.legal_actions.map(a => {
                    if (a.type === "bet" || a.type === "raise") {
                      return (
                        <button key={a.type} onClick={openRaiseSlider} disabled={loading}
                          className={cn("px-5 py-2.5 rounded-lg text-sm font-medium border transition-all disabled:opacity-40",
                            ACTION_COLORS[a.type])}>
                          {a.type === "bet" ? "Bet" : "Raise"}
                        </button>
                      );
                    }
                    return (
                      <button key={a.type} onClick={() => act(a.type, a.amount || 0)} disabled={loading}
                        className={cn("px-5 py-2.5 rounded-lg text-sm font-medium border transition-all disabled:opacity-40",
                          ACTION_COLORS[a.type] || ACTION_COLORS.check)}>
                        {a.label}
                      </button>
                    );
                  })}
                </div>

                {/* Sizing panel */}
                {showRaiseSlider && betRaiseAction && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-lg bg-graphite-800/60 border border-graphite-700/50 space-y-3">
                    {/* Sizing presets */}
                    <div className="flex flex-wrap gap-1.5 justify-center">
                      {game.sizing_suggestions.map(s => (
                        <button key={s.name} onClick={() => setRaiseAmount(s.amount)}
                          className={cn("px-3 py-1.5 rounded text-xs font-medium border transition-colors",
                            Math.abs(raiseAmount - s.amount) < 0.5
                              ? "bg-accent-blue/20 text-accent-blue border-accent-blue/30"
                              : "bg-graphite-800 text-graphite-300 border-graphite-700 hover:text-white")}>
                          {s.name} ({s.amount})
                        </button>
                      ))}
                    </div>

                    {/* Slider */}
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-graphite-500 font-mono w-10">{betRaiseAction.min_amount}</span>
                      <input type="range"
                        min={betRaiseAction.min_amount} max={betRaiseAction.max_amount}
                        step={0.5} value={raiseAmount}
                        onChange={e => setRaiseAmount(parseFloat(e.target.value))}
                        className="flex-1 accent-accent-blue" />
                      <span className="text-xs text-graphite-500 font-mono w-10 text-right">{betRaiseAction.max_amount}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="font-mono text-lg text-white font-bold">{raiseAmount.toFixed(1)}</span>
                      <div className="flex gap-2">
                        <button onClick={() => setShowRaiseSlider(false)}
                          className="px-3 py-1.5 text-xs text-graphite-400 hover:text-white">Cancel</button>
                        <button onClick={() => act(betRaiseAction.type, raiseAmount)} disabled={loading}
                          className="px-4 py-1.5 rounded-lg bg-accent-blue text-white text-sm font-medium hover:bg-accent-blue/90 disabled:opacity-50">
                          {betRaiseAction.type === "bet" ? `Bet ${raiseAmount.toFixed(1)}` : `Raise to ${raiseAmount.toFixed(1)}`}
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            )}

            {!game.is_player_turn && !game.is_terminal && (
              <motion.span animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity }}
                className="text-sm text-graphite-400 mt-2">Bot thinking...</motion.span>
            )}

            {game.is_terminal && game.message && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-2">
                <div className={cn("inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border text-sm font-medium",
                  game.winner === "player" ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" :
                  game.winner === "bot" ? "bg-red-500/10 border-red-500/20 text-red-400" :
                  "bg-graphite-800 border-graphite-700 text-graphite-300")}>
                  {game.message}
                </div>
              </motion.div>
            )}
          </div>
        )}
      </div>

      {/* Advisor + Log */}
      <div className="grid md:grid-cols-2 gap-4">
        {showAdvisor && advice && (
          <div className="card border-blue-500/20">
            <h3 className="text-sm font-semibold text-white mb-3">Solver Advisor</h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
              <div><div className="text-[10px] uppercase text-graphite-500">Equity</div><div className="font-mono text-lg text-white font-bold">{advice.equity_pct}%</div></div>
              <div><div className="text-[10px] uppercase text-graphite-500">SPR</div><div className="font-mono text-lg text-white font-bold">{advice.spr}</div></div>
              <div><div className="text-[10px] uppercase text-graphite-500">Eff Stack</div><div className="font-mono text-lg text-white font-bold">{advice.effective_stack}</div></div>
              <div><div className="text-[10px] uppercase text-graphite-500">Bucket</div><div className="font-mono text-lg text-white font-bold">{advice.preflop_bucket}/7</div></div>
            </div>
            <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 mb-3">
              <div className="text-[10px] uppercase text-blue-400/70 mb-1">Recommended</div>
              <div className="text-lg font-bold text-blue-400">{advice.recommended}</div>
            </div>
            {advice.sizing_suggestions.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] uppercase text-graphite-500">Sizing Options</div>
                {advice.sizing_suggestions.map(s => (
                  <div key={s.name} className="flex items-center justify-between text-xs py-0.5">
                    <span className="text-graphite-300">{s.name}</span>
                    <span className="font-mono text-graphite-400">{s.amount}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className={cn("card", !showAdvisor || !advice ? "md:col-span-2" : "")}>
          <h3 className="text-sm font-semibold text-white mb-2">Action Log</h3>
          <div ref={logRef} className="space-y-0.5 max-h-64 overflow-y-auto font-mono text-xs">
            {!game?.action_log?.length ? <p className="text-graphite-600">No actions yet</p> :
              game.action_log.map((e, i) => (
                <div key={i} className={cn("py-0.5",
                  e.startsWith("You") ? "text-blue-400" :
                  e.startsWith("Bot") ? "text-amber-400" :
                  "text-graphite-400"
                )}>{e}</div>
              ))
            }
          </div>
        </div>
      </div>
    </div>
  );
}

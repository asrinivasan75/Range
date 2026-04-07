"""Slumbot integration — play our bot against Slumbot's API.

Slumbot: strong heads-up NLHE bot (200bb deep, 50/100 blinds).
API: https://slumbot.com/slumbot/api/
Free, register at slumbot.com.

Protocol:
  POST /api/new_hand -> returns action string (may include bot's first action)
  POST /api/act      -> send our action, get response
  When "winnings" is present and action string ends with terminal action -> hand over
"""

from __future__ import annotations
import time
import json
import httpx
import random
from dataclasses import dataclass, field

from packages.poker.card import Card
from packages.poker.evaluator import hand_strength_monte_carlo, evaluate_hand
from packages.poker.hand import HandCategory
from packages.solver.rl_agent import (
    QLearningAgent, extract_features as rl_extract_features, rl_action_to_slumbot,
    get_legal_mask as rl_get_legal_mask, Experience,
)
from packages.solver.neural_agent import (
    PPOAgent, extract_features as ppo_extract_features, ppo_action_to_slumbot,
    get_legal_mask as ppo_get_legal_mask, Transition,
)

SLUMBOT_URL = "https://slumbot.com/slumbot"
SB = 50
BB = 100
STACK = 20000


@dataclass
class SlumbotSession:
    username: str
    password: str
    token: str = ""
    hands_played: int = 0
    total_winnings: int = 0
    results: list[dict] = field(default_factory=list)

    def login(self) -> bool:
        r = httpx.post(f"{SLUMBOT_URL}/api/login",
                       json={"username": self.username, "password": self.password},
                       headers={"Content-Type": "application/json"})
        if r.status_code == 200:
            data = r.json()
            self.token = data.get("token", "")
            return bool(self.token)
        return False

    def new_hand(self) -> dict:
        body = {"token": self.token} if self.token else {}
        r = httpx.post(f"{SLUMBOT_URL}/api/new_hand", json=body,
                       headers={"Content-Type": "application/json"})
        data = r.json()
        if "token" in data:
            self.token = data["token"]
        return data

    def act(self, incr: str) -> dict:
        r = httpx.post(f"{SLUMBOT_URL}/api/act",
                       json={"token": self.token, "incr": incr},
                       headers={"Content-Type": "application/json"})
        data = r.json()
        if "token" in data:
            self.token = data["token"]
        return data


def _parse_cards(strs: list[str]) -> list[Card]:
    return [Card.from_str(s) for s in strs]


def _parse_action(action_str: str) -> list[dict]:
    """Parse Slumbot action string into a list of individual actions."""
    result = []
    i = 0
    while i < len(action_str):
        c = action_str[i]
        if c == '/':
            result.append({"type": "sep"})
            i += 1
        elif c in ('k', 'c', 'f'):
            result.append({"type": c})
            i += 1
        elif c == 'b':
            j = i + 1
            while j < len(action_str) and action_str[j].isdigit():
                j += 1
            result.append({"type": "b", "amount": int(action_str[i+1:j])})
            i = j
        else:
            i += 1
    return result


def _count_actions_on_current_street(action_str: str) -> int:
    """Count how many player actions have been taken on the current street."""
    # Find the last street separator
    last_sep = action_str.rfind('/')
    current = action_str[last_sep+1:] if last_sep >= 0 else action_str
    # Count non-separator actions
    count = 0
    i = 0
    while i < len(current):
        c = current[i]
        if c == 'b':
            j = i + 1
            while j < len(current) and current[j].isdigit():
                j += 1
            count += 1
            i = j
        elif c in 'kcf':
            count += 1
            i += 1
        else:
            i += 1
    return count


def _compute_state(action_str: str, client_pos: int) -> dict:
    """Compute game state from the full action string.

    Returns dict with: street, pot, our_invested, opp_invested, to_call,
    our_total_bet, opp_total_bet, whose_turn (0 or 1 index on current street).
    """
    streets_raw = action_str.split('/') if action_str else ['']
    street = len(streets_raw) - 1

    # Track total invested across all streets
    our_total = SB if client_pos == 0 else BB
    opp_total = BB if client_pos == 0 else SB

    for s_idx, s_actions in enumerate(streets_raw):
        # Preflop: button/SB acts first (even indices)
        # Postflop: BB acts first (even indices)
        if s_idx == 0:
            # Preflop: position 0 (SB/button) acts at even indices
            our_parity = 0 if client_pos == 0 else 1
        else:
            # Postflop: position 1 (BB) acts at even indices
            our_parity = 1 if client_pos == 0 else 0

        # Parse actions on this street
        bets = [0, 0]  # [even_idx_player, odd_idx_player] bets on this street
        if s_idx == 0:
            bets = [SB, BB] if client_pos == 0 else [BB, SB]
            # Actually: bets[0] = SB player's bet, bets[1] = BB player's bet
            # For preflop, even index = SB (client_pos=0) or BB (client_pos=1)
            bets_by_pos = [SB, BB]  # [SB_bet, BB_bet]
        else:
            bets_by_pos = [0, 0]

        action_idx = 0
        i = 0
        while i < len(s_actions):
            c = s_actions[i]
            if c == 'b':
                j = i + 1
                while j < len(s_actions) and s_actions[j].isdigit():
                    j += 1
                amount = int(s_actions[i+1:j])
                # amount is total bet/raise TO, not increment
                # Determine which position this action belongs to
                if s_idx == 0:
                    pos = action_idx % 2  # 0=SB, 1=BB in preflop
                else:
                    pos = (action_idx % 2 + 1) % 2  # 0=BB, 1=SB in postflop
                    # Actually postflop: even action_idx = first actor = BB position
                    pos = 1 - (action_idx % 2)  # even=BB(pos1 if client=0), odd=SB
                    # Let me just track by action index parity
                    pass

                # Simpler: just track max bet
                bets_by_pos[action_idx % 2] = amount
                action_idx += 1
                i = j
            elif c == 'c':
                bets_by_pos[action_idx % 2] = max(bets_by_pos)
                action_idx += 1
                i += 1
            elif c in 'kf':
                action_idx += 1
                i += 1
            else:
                i += 1

    # For simplicity, just compute from the full action string
    # Our total and opp total invested
    pot = our_total + opp_total

    # Actually, let me use a much simpler approach:
    # Parse ALL bet amounts and calls to compute the pot
    # Slumbot convention: client_pos=0 -> we are BB, client_pos=1 -> we are BTN/SB
    # In action string: position that acts first preflop is BTN/SB (odd index = us when client_pos=0)
    # Action string position 0 = BTN/SB = the one who acts first preflop
    total_invested = [0, 0]  # [action_string_pos_0 = BTN/SB, action_string_pos_1 = BB]

    # Blinds: pos 0 in action = BTN/SB = 50, pos 1 in action = BB = 100
    total_invested[0] = SB  # BTN/SB
    total_invested[1] = BB  # BB

    for s_idx, s_actions in enumerate(streets_raw):
        street_bets = [0, 0]
        if s_idx == 0:
            street_bets = [SB, BB]

        action_idx = 0
        i = 0
        while i < len(s_actions):
            c = s_actions[i]
            if c == 'b':
                j = i + 1
                while j < len(s_actions) and s_actions[j].isdigit():
                    j += 1
                amt = int(s_actions[i+1:j])
                # Slumbot: position 0 ALWAYS acts first, no postflop flip
                who = action_idx % 2
                diff = amt - street_bets[who]
                total_invested[who] += max(diff, 0)
                street_bets[who] = amt
                action_idx += 1
                i = j
            elif c == 'c':
                who = action_idx % 2
                diff = max(street_bets) - street_bets[who]
                total_invested[who] += max(diff, 0)
                street_bets[who] = max(street_bets)
                action_idx += 1
                i += 1
            elif c in 'kf':
                action_idx += 1
                i += 1
            else:
                i += 1

    # Map back to our/opp based on client_pos
    # client_pos=0 -> we are BB -> we are action string position 1
    # client_pos=1 -> we are BTN/SB -> we are action string position 0
    if client_pos == 0:
        our_inv = total_invested[1]  # we are BB = action pos 1
        opp_inv = total_invested[0]  # Slumbot is BTN = action pos 0
    else:
        our_inv = total_invested[0]  # we are BTN = action pos 0
        opp_inv = total_invested[1]
        opp_inv = total_invested[0]

    pot = our_inv + opp_inv
    to_call = max(0, opp_inv - our_inv)

    return {
        "street": street,
        "pot": pot,
        "our_invested": our_inv,
        "opp_invested": opp_inv,
        "to_call": to_call,
        "our_remaining": STACK - our_inv,
    }


def _choose_action(
    hole_cards: list[Card],
    board: list[Card],
    state: dict,
    action_str: str = "",
    client_pos: int = 0,
) -> str:
    """Choose our action. Returns Slumbot action string."""
    equity = hand_strength_monte_carlo(hole_cards, board, n_simulations=150)
    pot = state["pot"]
    to_call = state["to_call"]
    our_remaining = state["our_remaining"]
    our_invested = state["our_invested"]
    street = state["street"]

    # Slumbot uses "c" not "k" for BB checking preflop when facing a limp/open
    # client_pos=0 -> we are BB. When action_str has Slumbot's open (e.g. 'b200'),
    # BB option is fold/call/raise. "c" = call the open.
    # When action_str is just SB completing (empty after Slumbot limps), "c" = check BB option.
    # client_pos=1 -> we are BTN/SB, act first, "k" is never valid preflop.
    # Preflop: always use 'c' for checking/calling, never 'k'. 'k' is only valid postflop.
    use_check_as_call = (street == 0)

    # Hand strength check for later streets
    has_pair_plus = False
    if len(board) >= 3:
        result = evaluate_hand(hole_cards, board)
        has_pair_plus = result[0].value >= 1

    # Discount equity based on street (opponent who bets has a stronger range)
    discount = [0.9, 0.8, 0.7, 0.6][min(street, 3)]
    adj_equity = equity * discount

    if to_call > 0:
        # Facing a bet
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        bet_ratio = to_call / max(pot - to_call, 1)

        # Fold conditions
        should_fold = False
        if street == 3 and not has_pair_plus and bet_ratio > 0.2:
            should_fold = True
        elif adj_equity < pot_odds * 0.9:
            should_fold = True
        elif adj_equity < 0.2 and bet_ratio > 0.4:
            should_fold = True

        if should_fold:
            return "f"

        # Raise with strong hands
        if adj_equity > 0.7 and our_remaining > to_call * 2:
            # Raise to ~2.5-3x the current bet
            raise_size = int(state["opp_invested"] * 2.5)
            raise_size = max(raise_size, state["opp_invested"] + BB)
            raise_size = min(raise_size, our_invested + our_remaining)
            return f"b{raise_size}"

        return "c"
    else:
        # Can check or bet
        check_char = "c" if use_check_as_call else "k"

        if equity > 0.6 or (equity > 0.35 and random.random() < 0.3):
            if equity > 0.75:
                bet = int(pot * 0.75)
            elif equity > 0.5:
                bet = int(pot * 0.5)
            else:
                bet = int(pot * 0.33)

            bet = max(bet, BB)
            bet = min(bet, our_remaining)
            total = our_invested + bet
            return f"b{total}"

        return check_char


def play_hand(
    session: SlumbotSession,
    verbose: bool = True,
    rl_agent: QLearningAgent | None = None,
    ppo_agent: PPOAgent | None = None,
) -> dict:
    """Play one hand against Slumbot. Returns full hand history."""
    data = session.new_hand()

    if "error" in data:
        if verbose:
            print(f"  Error: {data['error']}")
        return {"hand": session.hands_played + 1, "winnings": 0, "error": data["error"]}

    client_pos = data.get("client_pos", 0)
    hole_cards = _parse_cards(data.get("hole_cards", []))
    hole_str = data.get("hole_cards", [])

    hand_log = {
        "hand_number": session.hands_played + 1,
        "our_cards": hole_str,
        "our_position": "BB" if client_pos == 0 else "BTN",
        "client_pos": client_pos,
        "actions": [],
        "board_flop": [],
        "board_turn": None,
        "board_river": None,
    }

    if verbose:
        pos = "BTN" if client_pos == 0 else "BB"
        print(f"  #{session.hands_played + 1}: {hole_str} ({pos})", end="")

    last_action_str = ""
    hand_experiences: list[Experience] = []
    is_first_response = True

    # Handle immediate fold from Slumbot (e.g., Slumbot folds preflop from BTN)
    initial_action = data.get("action", "")
    if initial_action.endswith("f"):
        # Slumbot folded immediately. We win the SB (50 chips = 0.5bb)
        hand_log["final_action_string"] = initial_action
        hand_log["winnings"] = SB
        hand_log["winnings_bb"] = SB / BB
        hand_log["actions"].append({"who": "slumbot", "action": "f", "full_action": initial_action, "board": [], "street": 0})
        session.hands_played += 1
        session.total_winnings += SB
        session.results.append(hand_log)
        if verbose:
            print(f" -> +{SB/BB:.1f}bb | {initial_action} (Slumbot folded)")
        if rl_agent:
            rl_agent.end_hand(SB / BB, [])
        return hand_log

    for _ in range(20):
        action_str = data.get("action", "")
        board_strs = data.get("board", [])

        if len(board_strs) >= 3 and not hand_log["board_flop"]:
            hand_log["board_flop"] = board_strs[:3]
        if len(board_strs) >= 4 and not hand_log["board_turn"]:
            hand_log["board_turn"] = board_strs[3]
        if len(board_strs) >= 5 and not hand_log["board_river"]:
            hand_log["board_river"] = board_strs[4]

        if action_str != last_action_str and last_action_str != "":
            new_part = action_str[len(last_action_str):]
            if new_part and new_part != "/":
                hand_log["actions"].append({
                    "who": "slumbot", "action": new_part,
                    "full_action": action_str, "board": board_strs,
                    "street": action_str.count("/"),
                })

        # Hand is over ONLY if this is NOT the first response (new_hand includes
        # the previous hand's winnings/action, not the current hand's)
        if not is_first_response and "winnings" in data and data["winnings"] is not None:
            break
        if "error" in data and not is_first_response:
            break
        is_first_response = False

        board = _parse_cards(board_strs)
        state = _compute_state(action_str, client_pos)

        # Choose action — PPO agent > RL agent > heuristic
        n_bets = _count_actions_on_current_street(action_str)
        use_c = (state["street"] == 0)

        if ppo_agent:
            features = ppo_extract_features(
                hole_cards, board, state["pot"], state["to_call"],
                state["our_invested"], state["our_remaining"],
                state["street"], client_pos == 1, n_bets,
            )
            legal_mask = ppo_get_legal_mask(state["to_call"], state["our_remaining"])
            action_idx, log_prob, value = ppo_agent.choose_action(features, legal_mask)
            our_action = ppo_action_to_slumbot(
                action_idx, state["pot"], state["to_call"],
                state["our_invested"], state["our_remaining"],
                use_call_for_check=use_c,
            )
            hand_experiences.append(Transition(features, action_idx, log_prob, value, 0.0, legal_mask))
        elif rl_agent:
            features = rl_extract_features(
                hole_cards, board, state["pot"], state["to_call"],
                state["our_invested"], state["our_remaining"],
                state["street"], client_pos == 1, n_bets,
            )
            legal_mask = rl_get_legal_mask(state["to_call"], state["our_remaining"])
            action_idx = rl_agent.choose_action(features, legal_mask)
            our_action = rl_action_to_slumbot(
                action_idx, state["pot"], state["to_call"],
                state["our_invested"], state["our_remaining"],
                use_call_for_check=use_c,
            )
            hand_experiences.append(Experience(features, action_idx, 0.0, legal_mask))
        else:
            our_action = _choose_action(hole_cards, board, state, action_str, client_pos)

        hand_log["actions"].append({
            "who": "us", "action": our_action,
            "full_action": action_str + our_action, "board": board_strs,
            "street": action_str.count("/"),
        })

        last_action_str = action_str + our_action
        data = session.act(our_action)

    # Final result
    winnings = data.get("winnings", 0) or 0
    final_action = data.get("action", "")
    final_board = data.get("board", [])

    # Update board from final state
    if len(final_board) >= 3:
        hand_log["board_flop"] = final_board[:3]
    if len(final_board) >= 4:
        hand_log["board_turn"] = final_board[3]
    if len(final_board) >= 5:
        hand_log["board_river"] = final_board[4]

    hand_log["final_action_string"] = final_action
    hand_log["winnings"] = winnings
    hand_log["winnings_bb"] = round(winnings / BB, 1)

    # Log any final slumbot action
    if final_action != last_action_str:
        new_part = final_action[len(last_action_str):] if len(final_action) > len(last_action_str) else ""
        if new_part and new_part not in ("/", "///"):
            hand_log["actions"].append({
                "who": "slumbot",
                "action": new_part,
                "full_action": final_action,
                "board": final_board,
                "street": final_action.count("/"),
            })

    # Learning: update agent with hand outcome
    if ppo_agent and hand_experiences:
        ppo_agent.end_hand(winnings / BB, hand_experiences)
    elif rl_agent and hand_experiences:
        rl_agent.end_hand(winnings / BB, hand_experiences)

    session.hands_played += 1
    session.total_winnings += winnings
    session.results.append(hand_log)

    if verbose:
        w = winnings / BB
        print(f" -> {'+' if w >= 0 else ''}{w:.1f}bb | {final_action}")

    return hand_log


def run_benchmark(
    username: str,
    password: str,
    n_hands: int = 100,
    verbose: bool = True,
    save_path: str = "data/slumbot_log.json",
    use_rl: bool = False,
    use_ppo: bool = False,
    weights_path: str = "data/rl_agent.pkl",
) -> dict:
    """Run benchmark against Slumbot. Saves full hand logs.

    Args:
        use_rl: Use linear Q-learning agent.
        use_ppo: Use PPO neural network agent (overrides use_rl).
        weights_path: Path to agent weights file.
    """
    session = SlumbotSession(username=username, password=password)

    print(f"Logging in to Slumbot as '{username}'...")
    if not session.login():
        print("Login failed. Register at https://slumbot.com first.")
        return {"error": "login_failed"}

    agent = None
    ppo = None
    if use_ppo:
        ppo = PPOAgent()
        ppo_path = weights_path.replace("rl_agent", "ppo_agent")
        if ppo.load(ppo_path):
            print(f"Loaded PPO agent ({ppo.hands_trained} hands, {ppo.updates_done} updates)")
        else:
            print("Starting fresh PPO neural network agent")
        print(f"Mode: PPO NEURAL NETWORK (learns mixed strategies)\n")
    elif use_rl:
        agent = QLearningAgent()
        if agent.load(weights_path):
            print(f"Loaded RL agent from {weights_path} ({agent.hands_trained} hands trained, eps={agent.epsilon:.3f})")
        else:
            print("Starting fresh RL agent")
        print(f"Mode: Q-LEARNING (learns as it plays)\n")
    else:
        print(f"Mode: Static heuristic\n")

    print(f"Playing {n_hands} hands against Slumbot (200bb, 50/100)...\n")

    for i in range(n_hands):
        try:
            play_hand(session, verbose=verbose, rl_agent=agent, ppo_agent=ppo)
            time.sleep(0.1)
        except Exception as e:
            print(f"  Error on hand {i+1}: {e}")
            time.sleep(1)

        # Save progress every 50 hands
        if (i + 1) % 50 == 0:
            _save_log(session, save_path)
            if ppo:
                ppo_path = weights_path.replace("rl_agent", "ppo_agent")
                ppo.save(ppo_path)
                if verbose:
                    stats = ppo.get_stats()
                    total_bb = session.total_winnings / BB
                    print(f"  --- {i+1} hands | Total: {total_bb:+.0f}bb | PPO updates: {stats['updates_done']} ---")
            elif agent:
                agent.save(weights_path)
                if verbose:
                    stats = agent.get_stats()
                    total_bb = session.total_winnings / BB
                    print(f"  --- {i+1} hands | Total: {total_bb:+.0f}bb | RL eps: {stats['epsilon']:.3f} ---")

    total_bb = session.total_winnings / BB
    avg_bb = total_bb / max(session.hands_played, 1)
    wins = sum(1 for r in session.results if r.get("winnings_bb", 0) > 0)
    losses = sum(1 for r in session.results if r.get("winnings_bb", 0) < 0)

    # Compute running total for chart
    running = []
    cumulative = 0
    for r in session.results:
        cumulative += r.get("winnings_bb", 0)
        running.append(round(cumulative, 1))

    summary = {
        "hands_played": session.hands_played,
        "total_bb": round(total_bb, 1),
        "bb_per_100": round(avg_bb * 100, 1),
        "win_count": wins,
        "loss_count": losses,
        "win_pct": round(wins / max(session.hands_played, 1) * 100, 1),
        "running_total": running,
        "biggest_win": max((r.get("winnings_bb", 0) for r in session.results), default=0),
        "biggest_loss": min((r.get("winnings_bb", 0) for r in session.results), default=0),
    }

    # Save full log
    log_data = {
        "summary": summary,
        "hands": session.results,
        "config": {
            "opponent": "Slumbot",
            "blinds": f"{SB}/{BB}",
            "stack": f"{STACK} ({STACK // BB}bb)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    _save_log(session, save_path, summary)

    print(f"\n{'='*50}")
    print(f"RESULTS ({session.hands_played} hands vs Slumbot)")
    print(f"{'='*50}")
    print(f"Total:     {'+' if total_bb >= 0 else ''}{total_bb:.1f} bb")
    print(f"Rate:      {'+' if avg_bb >= 0 else ''}{avg_bb:.2f} bb/hand ({summary['bb_per_100']} bb/100)")
    print(f"Wins:      {wins} ({summary['win_pct']}%)")
    print(f"Losses:    {losses}")
    print(f"Biggest W: +{summary['biggest_win']}bb")
    print(f"Biggest L: {summary['biggest_loss']}bb")

    if ppo:
        ppo_path = weights_path.replace("rl_agent", "ppo_agent")
        ppo.save(ppo_path)
        stats = ppo.get_stats()
        print(f"\nPPO Agent: {stats['hands_trained']} hands, {stats['updates_done']} updates, lr={stats['lr']:.6f}")
        print(f"Saved to {ppo_path}")
    elif agent:
        agent.save(weights_path)
        stats = agent.get_stats()
        print(f"\nRL Agent: {stats['hands_trained']} hands trained, eps={stats['epsilon']:.4f}")
        print(f"Agent saved to {weights_path}")

    print(f"\nFull log saved to {save_path}")
    print(f"View at: http://localhost:3333/dashboard/replayer")

    return summary


def _save_log(session: SlumbotSession, path: str, summary: dict | None = None):
    """Save hand log to JSON file."""
    import json
    from pathlib import Path

    running = []
    cumulative = 0
    for r in session.results:
        cumulative += r.get("winnings_bb", 0)
        running.append(round(cumulative, 1))

    n = max(session.hands_played, 1)
    total_bb = session.total_winnings / BB
    wins = sum(1 for r in session.results if r.get("winnings_bb", 0) > 0)
    all_bb = [r.get("winnings_bb", 0) for r in session.results]

    computed_summary = {
        "hands_played": session.hands_played,
        "total_bb": round(total_bb, 1),
        "bb_per_100": round(total_bb / n * 100, 1),
        "win_count": wins,
        "loss_count": sum(1 for x in all_bb if x < 0),
        "win_pct": round(wins / n * 100, 1),
        "biggest_win": round(max(all_bb, default=0), 1),
        "biggest_loss": round(min(all_bb, default=0), 1),
        "running_total": running,
    }

    data = {
        "summary": summary or computed_summary,
        "hands": session.results,
        "config": {
            "opponent": "Slumbot",
            "blinds": f"{SB}/{BB}",
            "stack_bb": STACK // BB,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m packages.solver.slumbot <username> <password> [n_hands] [--rl]")
        print("  --rl  Use reinforcement learning agent (learns as it plays)")
        sys.exit(1)

    use_rl = "--rl" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--rl"]
    n = int(args[2]) if len(args) > 2 else 50
    run_benchmark(args[0], args[1], n_hands=n, use_rl=use_rl)

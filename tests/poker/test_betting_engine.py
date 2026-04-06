"""Comprehensive tests for the NLHE betting engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from packages.poker.betting_engine import (
    BettingEngine, BettingState, PlayerChips, LegalAction, ActionKind, ActionRecord,
)
from packages.poker.game_state import Street

E = BettingEngine


def hand(stacks=(100.0, 100.0), button=0, sb=1.0, bb=2.0) -> BettingState:
    """Shortcut to create a standard hand."""
    return E.create_hand(button_seat=button, stacks=stacks, sb=sb, bb=bb)


def legal_kinds(state: BettingState) -> set[ActionKind]:
    return {a.kind for a in E.get_legal_actions(state)}


def find_action(state: BettingState, kind: ActionKind) -> LegalAction | None:
    for a in E.get_legal_actions(state):
        if a.kind == kind:
            return a
    return None


# ── Group 1: Hand Creation & Blinds ─────────────────────────

class TestHandCreation:
    def test_posts_blinds(self):
        s = hand()
        # Button (seat 0) posts SB=1, non-button (seat 1) posts BB=2
        assert s.players[0].bet_this_street == 1.0  # SB
        assert s.players[1].bet_this_street == 2.0  # BB
        assert s.players[0].stack == 99.0
        assert s.players[1].stack == 98.0
        assert s.pot == 3.0

    def test_preflop_button_acts_first(self):
        s = hand(button=0)
        assert s.current_player_idx == 0  # button/SB acts first preflop

    def test_preflop_button_seat_1(self):
        s = hand(button=1)
        assert s.current_player_idx == 1  # button is seat 1, acts first
        assert s.players[1].bet_this_street == 1.0  # SB
        assert s.players[0].bet_this_street == 2.0  # BB

    def test_street_is_preflop(self):
        s = hand()
        assert s.street == Street.PREFLOP

    def test_hand_not_complete(self):
        s = hand()
        assert not s.is_hand_complete


# ── Group 2: Legal Actions — No Raise Cap ────────────────────

class TestLegalActions:
    def test_preflop_sb_can_fold_call_raise(self):
        s = hand()
        kinds = legal_kinds(s)
        assert ActionKind.FOLD in kinds
        assert ActionKind.CALL in kinds
        assert ActionKind.RAISE in kinds

    def test_preflop_sb_call_amount(self):
        s = hand()
        call = find_action(s, ActionKind.CALL)
        assert call is not None
        assert call.amount == 1.0  # call 1 more to match BB of 2

    def test_unopened_pot_check_or_bet(self):
        s = hand()
        # SB calls
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        # BB now has check or bet
        kinds = legal_kinds(s)
        assert ActionKind.CHECK in kinds
        assert ActionKind.BET in kinds
        assert ActionKind.FOLD not in kinds  # nothing to fold to

    def test_facing_bet_fold_call_raise(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)  # SB limps
        s = E.apply_action(s, ActionKind.BET, 4.0)    # BB bets 4
        kinds = legal_kinds(s)
        assert ActionKind.FOLD in kinds
        assert ActionKind.CALL in kinds
        assert ActionKind.RAISE in kinds

    def test_facing_raise_can_reraise(self):
        """THE CRITICAL BUG 1 REGRESSION TEST"""
        s = hand()
        # SB raises to 6
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        # BB re-raises to 14
        s = E.apply_action(s, ActionKind.RAISE, 14.0)
        # SB should be able to RE-RE-RAISE (no cap!)
        kinds = legal_kinds(s)
        assert ActionKind.RAISE in kinds, "Must be able to re-raise in no-limit!"

    def test_five_reraises_still_legal(self):
        s = hand(stacks=(1000.0, 1000.0))
        # SB raises to 6
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        # Series of re-raises
        amounts = [14, 26, 50, 98]
        for i, amt in enumerate(amounts):
            s = E.apply_action(s, ActionKind.RAISE, float(amt))
        # After 5 total aggressive actions, should STILL be able to raise
        kinds = legal_kinds(s)
        assert ActionKind.RAISE in kinds

    def test_no_bet_when_nothing_to_bet_with(self):
        # Player with 0 stack after posting blind
        s = hand(stacks=(2.0, 2.0))  # SB posts 1, BB posts 2 (all-in)
        # BB is all-in with 0 stack, so SB faces BB's 2
        # SB has 1 chip left, to_call is 1
        kinds = legal_kinds(s)
        assert ActionKind.CALL in kinds or ActionKind.ALL_IN in kinds


# ── Group 3: Min Raise Calculations ──────────────────────────

class TestMinRaise:
    def test_min_raise_preflop(self):
        s = hand()
        # BB=2, so min raise = 2 + 2 = 4 (raise TO 4)
        raise_action = find_action(s, ActionKind.RAISE)
        assert raise_action is not None
        assert raise_action.min_amount == 4.0  # 2 (BB) + 2 (last raise = BB)

    def test_min_3bet(self):
        s = hand()
        # SB raises to 6 (increment of 4 over BB's 2)
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        # Min 3bet = 6 + 4 = 10
        raise_action = find_action(s, ActionKind.RAISE)
        assert raise_action is not None
        assert raise_action.min_amount == 10.0

    def test_min_4bet(self):
        s = hand(stacks=(500.0, 500.0))
        s = E.apply_action(s, ActionKind.RAISE, 6.0)   # raise to 6 (incr 4)
        s = E.apply_action(s, ActionKind.RAISE, 16.0)  # 3bet to 16 (incr 10)
        # Min 4bet = 16 + 10 = 26
        raise_action = find_action(s, ActionKind.RAISE)
        assert raise_action is not None
        assert raise_action.min_amount == 26.0

    def test_max_raise_is_stack(self):
        s = hand()
        raise_action = find_action(s, ActionKind.RAISE)
        assert raise_action is not None
        # Max = current bet (1) + stack (99) = 100
        assert raise_action.max_amount == 100.0

    def test_all_in_for_less_than_min_raise(self):
        """Short stack goes all-in for less than a full raise."""
        s = hand(stacks=(100.0, 8.0))  # BB has only 8 total (6 after posting BB)
        # SB raises to 6
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        # BB has 6 stack, facing raise to 6, to_call = 4
        # BB can call 4, or all-in for 6 total (bet_this_street becomes 8)
        # all-in would be 8 total, which is only +2 over the 6 raise (not a full raise of 4)
        kinds = legal_kinds(s)
        assert ActionKind.CALL in kinds or ActionKind.ALL_IN in kinds

    def test_short_all_in_does_not_reopen(self):
        """All-in for less than a full raise should NOT reopen raising."""
        s = hand(stacks=(100.0, 8.0), button=0)
        # SB raises to 6 (increment 4)
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        # BB all-in for 8 total (only +2 over 6, less than full raise of 4)
        s = E.apply_action(s, ActionKind.ALL_IN)
        # SB must still act (fold or call), but should NOT be able to raise
        kinds = legal_kinds(s)
        assert ActionKind.RAISE not in kinds, "Short all-in must NOT reopen raising"
        assert ActionKind.FOLD in kinds
        assert ActionKind.CALL in kinds


# ── Group 4: Street Completion ───────────────────────────────

class TestStreetCompletion:
    def test_check_check_ends_street(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)   # SB limps
        s = E.apply_action(s, ActionKind.CHECK)        # BB checks
        assert E.is_street_complete(s)

    def test_bet_call_ends_street(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)   # SB limps
        s = E.apply_action(s, ActionKind.BET, 4.0)    # BB bets 4
        s = E.apply_action(s, ActionKind.CALL, 4.0)   # SB calls
        assert E.is_street_complete(s)

    def test_bet_raise_call_ends_street(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)   # SB raises to 6
        s = E.apply_action(s, ActionKind.RAISE, 14.0)  # BB re-raises to 14
        s = E.apply_action(s, ActionKind.CALL, 14.0)   # SB calls
        assert E.is_street_complete(s)

    def test_bet_raise_does_not_end(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)   # SB raises to 6
        s = E.apply_action(s, ActionKind.RAISE, 14.0)  # BB re-raises to 14
        # SB has NOT acted since BB's raise
        assert not E.is_street_complete(s)

    def test_bet_fold_ends_hand(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)  # SB raises
        s = E.apply_action(s, ActionKind.FOLD)         # BB folds
        assert s.is_hand_complete

    def test_preflop_limp_bb_check(self):
        """SB limps, BB checks -> street over."""
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        assert not E.is_street_complete(s)  # BB hasn't acted
        s = E.apply_action(s, ActionKind.CHECK)
        assert E.is_street_complete(s)


# ── Group 5: Position ────────────────────────────────────────

class TestPosition:
    def test_postflop_nonbutton_acts_first(self):
        s = hand(button=0)
        s = E.apply_action(s, ActionKind.CALL, 1.0)  # SB limps
        s = E.apply_action(s, ActionKind.CHECK)       # BB checks
        s = E.advance_street(s)
        # Postflop: non-button (seat 1, BB) acts first
        assert s.current_player_idx == 1

    def test_postflop_button1_nonbutton_acts_first(self):
        s = hand(button=1)
        s = E.apply_action(s, ActionKind.CALL, 1.0)  # SB(seat 1) limps
        s = E.apply_action(s, ActionKind.CHECK)       # BB(seat 0) checks
        s = E.advance_street(s)
        # Non-button is seat 0
        assert s.current_player_idx == 0

    def test_advance_to_flop(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        s = E.apply_action(s, ActionKind.CHECK)
        s = E.advance_street(s)
        assert s.street == Street.FLOP

    def test_advance_to_turn(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        s = E.apply_action(s, ActionKind.CHECK)
        s = E.advance_street(s)  # flop
        s = E.apply_action(s, ActionKind.CHECK)
        s = E.apply_action(s, ActionKind.CHECK)
        s = E.advance_street(s)  # turn
        assert s.street == Street.TURN

    def test_advance_past_river_completes_hand(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        s = E.apply_action(s, ActionKind.CHECK)
        for _ in range(3):  # flop, turn, river
            s = E.advance_street(s)
            s = E.apply_action(s, ActionKind.CHECK)
            s = E.apply_action(s, ActionKind.CHECK)
        # After river, advance should complete hand
        s = E.advance_street(s)
        assert s.is_hand_complete


# ── Group 6: Stack Tracking ──────────────────────────────────

class TestStacks:
    def test_bet_reduces_stack(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)  # cost 5 more (already in for 1)
        assert s.players[0].stack == 94.0  # 100 - 1 (SB) - 5 (raise)

    def test_call_reduces_stack(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        s = E.apply_action(s, ActionKind.CALL, 6.0)   # BB calls, cost = 6-2 = 4
        assert s.players[1].stack == 94.0  # 100 - 2 (BB) - 4 (call)

    def test_all_in_zeroes_stack(self):
        s = hand()
        s = E.apply_action(s, ActionKind.ALL_IN)
        assert s.players[0].stack == 0.0
        assert s.players[0].is_all_in

    def test_pot_accumulates(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)   # SB puts in 6 total (5 more)
        s = E.apply_action(s, ActionKind.CALL, 6.0)    # BB puts in 6 total (4 more)
        assert s.pot == 12.0  # 6 + 6

    def test_street_reset_bets(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        s = E.apply_action(s, ActionKind.CHECK)
        s = E.advance_street(s)
        assert s.players[0].bet_this_street == 0.0
        assert s.players[1].bet_this_street == 0.0
        assert s.pot == 4.0  # pot preserved


# ── Group 7: Immutability ────────────────────────────────────

class TestImmutability:
    def test_apply_returns_new_state(self):
        s1 = hand()
        s2 = E.apply_action(s1, ActionKind.RAISE, 6.0)
        # s1 should be unchanged
        assert s1.players[0].stack == 99.0
        assert s2.players[0].stack == 94.0
        assert s1 is not s2

    def test_advance_returns_new_state(self):
        s = hand()
        s = E.apply_action(s, ActionKind.CALL, 1.0)
        s = E.apply_action(s, ActionKind.CHECK)
        s2 = E.advance_street(s)
        assert s.street == Street.PREFLOP
        assert s2.street == Street.FLOP


# ── Group 8: Action Log ─────────────────────────────────────

class TestActionLog:
    def test_log_records_actions(self):
        s = hand()
        s = E.apply_action(s, ActionKind.RAISE, 6.0)
        s = E.apply_action(s, ActionKind.CALL, 6.0)
        # 2 blind posts + 2 actions = 4
        assert len(s.action_log) == 4

    def test_log_preserves_order(self):
        s = hand()
        s = E.apply_action(s, ActionKind.FOLD)
        last = s.action_log[-1]
        assert last.kind == ActionKind.FOLD
        assert last.player_idx == 0


# ── Group 9: Edge Cases ─────────────────────────────────────

class TestEdgeCases:
    def test_very_short_stack_sb(self):
        """SB has less than 1 blind."""
        s = E.create_hand(button_seat=0, stacks=(0.5, 100.0), sb=1.0, bb=2.0)
        # SB posts 0.5 (all they have), BB posts 2
        assert s.players[0].stack == 0.0
        assert s.players[0].is_all_in
        assert s.players[0].bet_this_street == 0.5

    def test_equal_stacks_all_in_both(self):
        s = hand(stacks=(50.0, 50.0))
        s = E.apply_action(s, ActionKind.ALL_IN)   # SB all-in
        s = E.apply_action(s, ActionKind.CALL, 49.0)  # BB calls
        assert s.pot == 100.0
        assert E.is_street_complete(s)

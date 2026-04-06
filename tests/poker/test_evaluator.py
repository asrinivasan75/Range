"""Tests for the poker hand evaluator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.poker.card import Card, Rank, Suit
from packages.poker.evaluator import evaluate_five, evaluate_hand, chen_formula, hand_strength_monte_carlo
from packages.poker.hand import HandCategory


def c(s: str) -> Card:
    return Card.from_str(s)


def test_royal_flush():
    cards = [c("As"), c("Ks"), c("Qs"), c("Js"), c("Ts")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.ROYAL_FLUSH


def test_straight_flush():
    cards = [c("9h"), c("8h"), c("7h"), c("6h"), c("5h")]
    cat, kickers = evaluate_five(cards)
    assert cat == HandCategory.STRAIGHT_FLUSH
    assert kickers == (9,)


def test_four_of_a_kind():
    cards = [c("Ah"), c("As"), c("Ad"), c("Ac"), c("Ks")]
    cat, kickers = evaluate_five(cards)
    assert cat == HandCategory.FOUR_OF_A_KIND
    assert kickers[0] == 14  # Aces


def test_full_house():
    cards = [c("Ah"), c("As"), c("Ad"), c("Kc"), c("Ks")]
    cat, kickers = evaluate_five(cards)
    assert cat == HandCategory.FULL_HOUSE


def test_flush():
    cards = [c("Ah"), c("Kh"), c("Th"), c("7h"), c("2h")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.FLUSH


def test_straight():
    cards = [c("9h"), c("8d"), c("7c"), c("6s"), c("5h")]
    cat, kickers = evaluate_five(cards)
    assert cat == HandCategory.STRAIGHT
    assert kickers == (9,)


def test_wheel_straight():
    cards = [c("Ah"), c("2d"), c("3c"), c("4s"), c("5h")]
    cat, kickers = evaluate_five(cards)
    assert cat == HandCategory.STRAIGHT
    assert kickers == (5,)  # Ace plays low


def test_three_of_a_kind():
    cards = [c("Ah"), c("As"), c("Ad"), c("Kc"), c("Qs")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.THREE_OF_A_KIND


def test_two_pair():
    cards = [c("Ah"), c("As"), c("Kd"), c("Kc"), c("Qs")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.TWO_PAIR


def test_one_pair():
    cards = [c("Ah"), c("As"), c("Kd"), c("Qc"), c("Js")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.ONE_PAIR


def test_high_card():
    cards = [c("Ah"), c("Ks"), c("Qd"), c("Jc"), c("9h")]
    cat, _ = evaluate_five(cards)
    assert cat == HandCategory.HIGH_CARD


def test_hand_comparison():
    flush = evaluate_five([c("Ah"), c("Kh"), c("Th"), c("7h"), c("2h")])
    straight = evaluate_five([c("9h"), c("8d"), c("7c"), c("6s"), c("5h")])
    assert flush > straight


def test_7_card_evaluation():
    hole = [c("Ah"), c("Kh")]
    board = [c("Qh"), c("Jh"), c("Th"), c("2d"), c("3c")]
    cat, _ = evaluate_hand(hole, board)
    assert cat == HandCategory.ROYAL_FLUSH


def test_chen_formula():
    # AA should be highest
    aa = chen_formula(c("Ah"), c("As"))
    k2o = chen_formula(c("Kh"), c("2s"))
    assert aa > k2o

    # AKs should be high
    aks = chen_formula(c("Ah"), c("Kh"))
    assert aks > 10


def test_monte_carlo_equity():
    # AA should have high equity vs random
    hole = [c("Ah"), c("As")]
    equity = hand_strength_monte_carlo(hole, [], n_simulations=500, seed=42)
    assert equity > 0.75  # AA wins ~85% heads up

    # 72o should have low equity
    hole2 = [c("7h"), c("2s")]
    equity2 = hand_strength_monte_carlo(hole2, [], n_simulations=500, seed=42)
    assert equity2 < 0.45

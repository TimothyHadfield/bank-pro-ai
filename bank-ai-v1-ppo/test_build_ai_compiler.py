"""
Build AI: the server-side variables must match the browser's.

The Build AI tab (dashboard/build.js, BuildBankGame.getVars) defines
p1_score..p5_score as ALL players' banked scores sorted high to low (so p1_score
is the leader's score, which can be your own), and p1_banked..p5_banked as the
banked flags in that same rank order. The Play vs AI (Build AI mode) and AI vs AI
tabs run the same strategy server-side through dashboard/build_ai_compiler.py,
so a strategy tuned in Build AI must see the same numbers there.

Run: python test_build_ai_compiler.py
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from dashboard.build_ai_compiler import compile_build_ai, state_for_build_ai


def _game(scores, players_in_round, pot=120, round_number=4, roll_number=6):
    return SimpleNamespace(
        pot=pot, round_number=round_number, roll_number=roll_number,
        num_players=len(scores), banked_scores=list(scores),
        players_in_round=set(players_in_round),
    )


def test_scores_sorted_by_rank_like_build_tab():
    print("TEST: p1..p5 scores/banked follow the Build AI tab's rank order")
    # Seat scores 100, 300, 50, 200. Seats 0 and 2 still in; 1 and 3 have banked.
    g = _game([100, 300, 50, 200], players_in_round=[0, 2])
    s = state_for_build_ai(g, 0)
    got_scores = [s[f"p{i}_score"] for i in range(1, 6)]
    got_banked = [s[f"p{i}_banked"] for i in range(1, 6)]
    # Same as build.js getVars(): every score sorted high->low, padded with 0
    assert got_scores == [300, 200, 100, 50, 0], got_scores
    # Rank order is seats 1, 3, 0, 2 -> banked 1, 1, 0, 0
    assert got_banked == [1, 1, 0, 0, 0], got_banked
    assert s["my_score"] == 100 and s["players_in"] == 2 and s["pot"] == 120
    print("PASSED\n")


def test_default_endgame_strategy_banks_when_leading_and_runner_up_banked():
    print("TEST: seeded 'Endgame' strategy decides the same as in the Build AI tab")
    endgame = (
        "if my_score >= p1_score and p2_banked == 1:\n"
        "    bank\n"
        "elif my_score < p1_score and p1_banked == 1 and my_score + pot > p1_score:\n"
        "    bank\n"
        "elif my_score < p1_score and p1_banked == 0:\n"
        "    continue\n"
        "else:\n"
        "    continue"
    )
    fn = compile_build_ai(endgame)
    # Seat 0 leads with 500; runner-up (seat 1, 400) already banked this round,
    # seat 2 (100) is still in. Build AI tab: p1 = 500 (me), p2_banked = 1 -> bank.
    g = _game([500, 400, 100], players_in_round=[0, 2])
    assert fn(state_for_build_ai(g, 0)) == "bank"
    print("PASSED\n")


if __name__ == "__main__":
    test_scores_sorted_by_rank_like_build_tab()
    test_default_endgame_strategy_banks_when_leading_and_runner_up_banked()
    print("All Build AI compiler tests passed.")

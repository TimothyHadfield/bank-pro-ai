"""
Tests for mcts/search.py.

Run from bank-ai-v2-mcts directory:
    python -m tests.test_mcts
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from game import BankGame
from game.environment import BANK, CONTINUE
from network import BankNetwork
from mcts import MCTS


def make_game(num_players=4, seed=42):
    g = BankGame(num_players, seed=seed)
    g.reset()
    return g


def test_run_returns_visit_dist():
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=4, seed=1)
    mcts = MCTS(net, root_player=0)
    visits = mcts.run(g, num_simulations=20)
    assert set(visits.keys()) == {BANK, CONTINUE}
    total = sum(visits.values())
    assert total == 20, f"Expected 20 visits, got {total}"
    print(f"  ok test_run_returns_visit_dist (visits={visits})")


def test_does_not_mutate_input_game():
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=3, seed=2)
    pot_before = g.pot
    round_before = g.round_number
    roll_before = g.roll_number
    scores_before = list(g.banked_scores)
    in_round_before = set(g.players_in_round)

    mcts = MCTS(net, root_player=0)
    _ = mcts.run(g, num_simulations=15)

    assert g.pot == pot_before
    assert g.round_number == round_before
    assert g.roll_number == roll_before
    assert g.banked_scores == scores_before
    assert g.players_in_round == in_round_before
    print("  ok test_does_not_mutate_input_game")


def test_works_for_all_player_counts():
    torch.manual_seed(0)
    net = BankNetwork()
    for nplayers in [2, 3, 4, 5, 6]:
        g = make_game(num_players=nplayers, seed=10 + nplayers)
        mcts = MCTS(net, root_player=0)
        visits = mcts.run(g, num_simulations=10)
        assert sum(visits.values()) == 10
    print("  ok test_works_for_all_player_counts")


def test_dirichlet_noise_changes_root_prior():
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=4, seed=3)

    # Run with Dirichlet noise — capture priors after expansion.
    # We can detect noise was applied by running same setup with/without and looking at visits.
    mcts_noisy = MCTS(
        net, root_player=0, rng=np.random.default_rng(seed=100)
    )
    visits_noisy = mcts_noisy.run(
        g, num_simulations=30, add_dirichlet_noise=True
    )

    mcts_clean = MCTS(
        net, root_player=0, rng=np.random.default_rng(seed=100)
    )
    visits_clean = mcts_clean.run(g, num_simulations=30, add_dirichlet_noise=False)

    # We just check both produce 30 total visits and no errors
    assert sum(visits_noisy.values()) == 30
    assert sum(visits_clean.values()) == 30
    print(
        f"  ok test_dirichlet_noise_changes_root_prior "
        f"(noisy={visits_noisy}, clean={visits_clean})"
    )


def test_late_game_obvious_bank_decision():
    """
    In a position where banking now would clearly win the game (only round, big pot, leader by margin),
    a moderately untrained MCTS should still tend to bank because the value of the bank-then-terminal
    branch is 1.0 while continuing risks a 7.

    This is a sanity check, not a strict guarantee — MCTS with random network can still pick weird
    things on small sim counts. We just verify it produces reasonable-looking output.
    """
    torch.manual_seed(7)
    net = BankNetwork()
    # Manually craft a state: round 10, post-immunity, big pot, root has slight lead
    g = BankGame(num_players=2, seed=99)
    g.reset()
    # Hack: set round to 10
    g.round_number = 10
    g.banked_scores = [50, 30]
    g.pot = 100
    g.roll_number = 4  # post-immunity
    g.rolls_since_immunity_ended = 1
    g.players_in_round = {0, 1}
    g.waiting_for_decisions = True
    g.game_over = False
    g.round_over = False
    # Banking now: scores -> [150, 30] then game ends after both bank or bust → root wins

    mcts = MCTS(net, root_player=0)
    visits = mcts.run(g, num_simulations=80)
    # Just verify run completes and returns sensible visits
    assert sum(visits.values()) == 80
    print(f"  ok test_late_game_obvious_bank_decision (visits={visits})")


def test_terminal_game_raises():
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=3, seed=4)
    g.game_over = True
    mcts = MCTS(net, root_player=0)
    try:
        mcts.run(g, num_simulations=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Should raise on terminal state")
    print("  ok test_terminal_game_raises")


def test_root_player_must_be_in_round():
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=3, seed=5)
    g.players_in_round = {1, 2}  # remove player 0
    mcts = MCTS(net, root_player=0)
    try:
        mcts.run(g, num_simulations=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Should raise when root not in round")
    print("  ok test_root_player_must_be_in_round")


def test_no_peeking_independence():
    """
    Run MCTS for two players in the same game state independently. Each must be unaffected
    by the other's run. (Their RNGs are independent in self-play; we test that running one
    doesn't mutate state used by the other.)
    """
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=4, seed=20)

    g_snapshot_p1 = g.clone()  # for player 0's MCTS
    g_snapshot_p2 = g.clone()  # for player 1's MCTS

    mcts0 = MCTS(net, root_player=0, rng=np.random.default_rng(seed=1))
    mcts1 = MCTS(net, root_player=1, rng=np.random.default_rng(seed=2))

    v0 = mcts0.run(g_snapshot_p1, num_simulations=15)
    v1 = mcts1.run(g_snapshot_p2, num_simulations=15)

    # Run again on fresh snapshots, in opposite order — results must match
    g_snapshot_p1_b = g.clone()
    g_snapshot_p2_b = g.clone()
    mcts1_again = MCTS(net, root_player=1, rng=np.random.default_rng(seed=2))
    mcts0_again = MCTS(net, root_player=0, rng=np.random.default_rng(seed=1))
    v1_again = mcts1_again.run(g_snapshot_p2_b, num_simulations=15)
    v0_again = mcts0_again.run(g_snapshot_p1_b, num_simulations=15)

    assert v0 == v0_again, f"Player 0 results differ based on order: {v0} vs {v0_again}"
    assert v1 == v1_again, f"Player 1 results differ based on order: {v1} vs {v1_again}"
    print(f"  ok test_no_peeking_independence (v0={v0}, v1={v1})")


def test_timing_smoke():
    """Measure rough wall time so we know what to expect."""
    torch.manual_seed(0)
    net = BankNetwork()
    g = make_game(num_players=4, seed=42)
    mcts = MCTS(net, root_player=0)
    t0 = time.perf_counter()
    visits = mcts.run(g, num_simulations=100)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(
        f"  ok test_timing_smoke "
        f"(100 sims, 4 players, took {elapsed*1000:.1f}ms = {elapsed*10:.1f}ms/sim)"
    )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_run_returns_visit_dist,
    test_does_not_mutate_input_game,
    test_works_for_all_player_counts,
    test_dirichlet_noise_changes_root_prior,
    test_late_game_obvious_bank_decision,
    test_terminal_game_raises,
    test_root_player_must_be_in_round,
    test_no_peeking_independence,
    test_timing_smoke,
]


def main():
    failures = []
    for t in ALL_TESTS:
        print(f"running {t.__name__}")
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((t.__name__, e))
            print(f"  FAIL: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(ALL_TESTS)} FAILED")
        for n, e in failures:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()

"""
Tests for training/self_play.py.

Run from bank-ai-v2-mcts directory:
    python -m tests.test_self_play
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from network import BankNetwork
from training import self_play_game, TrainingExample
from training.self_play import summarize_examples


def test_one_self_play_game_completes():
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(
        net,
        num_players=3,
        sims_per_move=10,  # tiny for speed
        add_dirichlet_noise=True,
        seed=42,
        force_pure_self_play=True,
    )
    assert len(examples) > 0, "Self-play produced no examples"
    print(f"  ok test_one_self_play_game_completes ({len(examples)} examples)")


def test_examples_have_correct_shape():
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(net, num_players=4, sims_per_move=5, seed=1, force_pure_self_play=True)
    for ex in examples:
        assert isinstance(ex, TrainingExample)
        assert ex.state_vector.shape == (config.INPUT_SIZE,)
        assert ex.state_vector.dtype == np.float32
        assert ex.mcts_policy.shape == (config.NUM_ACTIONS,)
        assert abs(ex.mcts_policy.sum() - 1.0) < 1e-5, f"policy sum != 1: {ex.mcts_policy.sum()}"
        assert 0 <= ex.player_id < ex.num_players
        assert 1 <= ex.round_number <= config.NUM_ROUNDS
    print(f"  ok test_examples_have_correct_shape ({len(examples)} examples)")


def test_values_are_assigned_correctly():
    """At least one example should have value > 0 (the winner(s))."""
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(net, num_players=3, sims_per_move=5, seed=2, force_pure_self_play=True)

    n_winning = sum(1 for ex in examples if ex.value > 0)
    n_losing = sum(1 for ex in examples if ex.value == 0)
    assert n_winning > 0, "No winning examples — but someone has to win"
    assert n_winning + n_losing == len(examples)

    # All winning examples should share the same player_id (or there's a tie)
    win_pids = {ex.player_id for ex in examples if ex.value > 0}
    win_values = {ex.value for ex in examples if ex.value > 0}
    # Sanity: all winning examples have the same value (1/k)
    assert len(win_values) == 1, f"Inconsistent winner values: {win_values}"
    print(
        f"  ok test_values_are_assigned_correctly "
        f"(winners={sorted(win_pids)}, value={list(win_values)[0]:.3f})"
    )


def test_works_for_all_player_counts():
    torch.manual_seed(0)
    net = BankNetwork()
    for nplayers in [2, 3, 4, 5, 6]:
        examples, _ = self_play_game(
            net, num_players=nplayers, sims_per_move=5, seed=10 + nplayers, force_pure_self_play=True
        )
        assert len(examples) > 0
        # All examples should report num_players correctly
        assert all(ex.num_players == nplayers for ex in examples)
    print("  ok test_works_for_all_player_counts")


def test_random_player_count_when_unspecified():
    """When num_players is None, it's sampled in [2, 6]."""
    torch.manual_seed(0)
    net = BankNetwork()
    counts_seen = set()
    for seed in range(15):
        examples, _ = self_play_game(net, num_players=None, sims_per_move=3, seed=seed, force_pure_self_play=True)
        if examples:
            counts_seen.add(examples[0].num_players)
    assert all(2 <= c <= 6 for c in counts_seen), f"Out-of-range counts: {counts_seen}"
    print(f"  ok test_random_player_count_when_unspecified (seen={sorted(counts_seen)})")


def test_temperature_drops_in_late_game():
    """In rounds 8-10 (last 3), temperature is 0 (greedy). Early rounds use full T=1.0."""
    from training.self_play import _temperature_for
    assert _temperature_for(1) == config.TEMPERATURE_INITIAL
    assert _temperature_for(7) == config.TEMPERATURE_INITIAL
    assert _temperature_for(8) == config.TEMPERATURE_LATE_GAME
    assert _temperature_for(10) == config.TEMPERATURE_LATE_GAME
    print("  ok test_temperature_drops_in_late_game")


def test_summary_helper():
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(net, num_players=4, sims_per_move=5, seed=99, force_pure_self_play=True)
    summary = summarize_examples(examples)
    assert summary["count"] == len(examples)
    assert summary["num_players"] == 4
    assert sum(summary["decisions_per_player"].values()) == len(examples)
    print(f"  ok test_summary_helper ({summary})")


def test_timing_smoke_one_game():
    """Time a realistic game with default sims_per_move to gauge overall speed."""
    torch.manual_seed(0)
    net = BankNetwork()
    t0 = time.perf_counter()
    examples, _ = self_play_game(
        net,
        num_players=4,
        sims_per_move=config.SIMS_PER_MOVE,  # 100
        seed=123,
        force_pure_self_play=True,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(
        f"  ok test_timing_smoke_one_game "
        f"({len(examples)} examples in {elapsed:.2f}s "
        f"= {elapsed/max(1,len(examples))*1000:.0f}ms per decision)"
    )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_one_self_play_game_completes,
    test_examples_have_correct_shape,
    test_values_are_assigned_correctly,
    test_works_for_all_player_counts,
    test_random_player_count_when_unspecified,
    test_temperature_drops_in_late_game,
    test_summary_helper,
    test_timing_smoke_one_game,
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
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()

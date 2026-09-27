"""
Tests for Slice 5: evaluation module, frozen-network opponent mix, end-to-end with eval.

Run from bank-ai-v2-mcts:
    python -m tests.test_slice5
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from evaluation import (
    Evaluator,
    EvaluationResult,
    play_match,
    evaluate_against_bots,
    evaluate_against_checkpoint,
)
from network import BankNetwork
from training import (
    FrozenNet,
    ThresholdBot,
    save_checkpoint,
    self_play_game,
)


# ---------------------------------------------------------------- play_match

def test_play_match_ai_vs_threshold():
    torch.manual_seed(0)
    net = BankNetwork()
    bot = ThresholdBot(200)
    from evaluation.evaluator import _AICompetitor
    ai = _AICompetitor(net, name="AI", sims_per_move=10)
    mr = play_match(ai, bot, num_players=4, games=5, base_seed=0)
    assert mr.games == 5
    assert 0.0 <= mr.a_win_rate <= 1.0
    assert 0.0 <= mr.b_win_rate <= 1.0
    # Wins fractional sums should equal total games
    assert abs((mr.a_wins + mr.b_wins) - mr.games) < 1e-6, f"a_wins+b_wins should equal games"
    print(f"  ok test_play_match_ai_vs_threshold (a={mr.a_win_rate:.2f}, b={mr.b_win_rate:.2f})")


def test_play_match_two_bots():
    """Bot vs bot to sanity-check the match infrastructure without MCTS."""
    a = ThresholdBot(150)
    b = ThresholdBot(300)
    mr = play_match(a, b, num_players=4, games=20, base_seed=1)
    assert mr.games == 20
    print(f"  ok test_play_match_two_bots ({a.name}={mr.a_win_rate:.2f}, {b.name}={mr.b_win_rate:.2f})")


# ---------------------------------------------------------------- evaluate_against_bots

def test_evaluate_against_bots_full():
    torch.manual_seed(0)
    net = BankNetwork()
    result = evaluate_against_bots(
        network=net,
        sims_per_move=5,
        games_per_player_count=4,
        player_counts=[2, 4],
        bot_thresholds=[150, 300],
        iteration=0,
    )
    assert isinstance(result, EvaluationResult)
    assert "Threshold(150)" in result.win_rates
    assert "Threshold(300)" in result.win_rates
    assert 2 in result.win_rates["Threshold(150)"]
    assert 4 in result.win_rates["Threshold(300)"]
    print(f"  ok test_evaluate_against_bots_full (overall={result.overall_win_rate():.3f})")


def test_evaluation_result_to_dict_jsonable():
    torch.manual_seed(0)
    net = BankNetwork()
    result = evaluate_against_bots(
        net, sims_per_move=3, games_per_player_count=2,
        player_counts=[3], bot_thresholds=[200], iteration=42,
    )
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["iteration"] == 42
    assert "win_rates" in d
    assert "overall_win_rate" in d
    print("  ok test_evaluation_result_to_dict_jsonable")


# ---------------------------------------------------------------- frozen network mix

def test_frozen_net_used_in_self_play():
    """When a frozen_nets list is given, some games should use those seats and produce no examples for them."""
    torch.manual_seed(0)
    net_current = BankNetwork()

    # Create a frozen "older" version (just a different random init)
    torch.manual_seed(99)
    net_old = BankNetwork()
    frozen = [FrozenNet(network=net_old, name="old-iter-1", sims_per_move=5)]

    # Run 30 games, force a config that should occasionally insert frozen seats
    n_games = 30
    n_with_frozen = 0
    for g in range(n_games):
        examples, result = self_play_game(
            network=net_current,
            num_players=4,
            sims_per_move=5,
            seed=g,
            bot_pool=None,
            frozen_nets=frozen,
        )
        if any(isinstance(s, FrozenNet) for s in result.seat_assignments):
            n_with_frozen += 1
            # Confirm: the frozen seats produced no examples (only AI seats record)
            ai_seat_count = sum(1 for s in result.seat_assignments if s == "AI")
            unique_pids_in_examples = set(ex.player_id for ex in examples)
            # All recorded pids must correspond to AI seats only
            for pid in unique_pids_in_examples:
                assert result.seat_assignments[pid] == "AI", (
                    f"seat {pid} produced example but is {type(result.seat_assignments[pid]).__name__}"
                )
    # Given OLD_CHECKPOINT_MIX_FRACTION=0.05 and bot_pool=None (so other branches degenerate
    # to OLD_CHECKPOINT_MIX_FRACTION effective), we should see at least some frozen games.
    assert n_with_frozen > 0, "no frozen games in 30 trials"
    print(f"  ok test_frozen_net_used_in_self_play ({n_with_frozen}/{n_games} games used frozen)")


# ---------------------------------------------------------------- evaluate_against_checkpoint

def test_evaluate_against_checkpoint_with_saved_file():
    torch.manual_seed(0)
    current_net = BankNetwork()
    other_net = BankNetwork()
    # Touch other_net's weights so it differs from current
    with torch.no_grad():
        for p in other_net.parameters():
            p.add_(0.05)

    with tempfile.TemporaryDirectory() as td:
        ck_path = Path(td) / "old.pt"
        save_checkpoint(ck_path, other_net, trainer=None, iteration=5)

        result = evaluate_against_checkpoint(
            current_network=current_net,
            other_checkpoint_path=ck_path,
            sims_per_move=5,
            games_per_player_count=3,
            player_counts=[3],
            iteration=10,
        )
        assert isinstance(result, EvaluationResult)
        op_name = next(iter(result.win_rates.keys()))
        assert "checkpoint_iter_5" in op_name
        assert 3 in result.win_rates[op_name]
        print(f"  ok test_evaluate_against_checkpoint_with_saved_file (op={op_name})")


# ---------------------------------------------------------------- Evaluator coordinator

def test_evaluator_run_no_old_checkpoint():
    torch.manual_seed(0)
    net = BankNetwork()
    ev = Evaluator()
    result = ev.run(
        network=net,
        iteration=1,
        sims_per_move=4,
        games_per_player_count=2,
        player_counts=[3],
        bot_thresholds=[150, 250],
    )
    assert ev.last_result is result
    assert "Threshold(150)" in result.win_rates
    assert "Threshold(250)" in result.win_rates
    assert "checkpoint_iter" not in str(result.win_rates)
    print(f"  ok test_evaluator_run_no_old_checkpoint (overall={result.overall_win_rate():.3f})")


def test_evaluator_run_with_old_checkpoint():
    torch.manual_seed(0)
    net = BankNetwork()
    other = BankNetwork()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.1)

    with tempfile.TemporaryDirectory() as td:
        ck_path = Path(td) / "iter_3.pt"
        save_checkpoint(ck_path, other, trainer=None, iteration=3)

        ev = Evaluator()
        result = ev.run(
            network=net,
            iteration=10,
            sims_per_move=3,
            games_per_player_count=2,
            player_counts=[3],
            bot_thresholds=[200],
            old_checkpoint_path=str(ck_path),
        )
        assert any("checkpoint_iter_3" in name for name in result.win_rates.keys()), (
            f"old checkpoint not in results: {result.win_rates.keys()}"
        )
        print("  ok test_evaluator_run_with_old_checkpoint")


# ---------------------------------------------------------------- end-to-end with eval

def test_end_to_end_training_with_evaluation():
    """
    Tiny training run with eval after every iteration. Verifies metrics.json contains eval data.

    All config values that drive eval cost are aggressively shrunk so the test runs in seconds
    rather than minutes (real training uses much larger eval panels).
    """
    from train import run_training

    saved = {k: getattr(config, k) for k in (
        "EVAL_INTERVAL_ITERATIONS",
        "CHECKPOINT_INTERVAL_ITERATIONS",
        "EVAL_GAMES_PER_OPPONENT",
        "THRESHOLD_BOT_THRESHOLDS",
        "EVAL_PLAYER_COUNTS",
    )}
    try:
        config.EVAL_INTERVAL_ITERATIONS = 1
        config.CHECKPOINT_INTERVAL_ITERATIONS = 1
        config.EVAL_GAMES_PER_OPPONENT = 8
        config.THRESHOLD_BOT_THRESHOLDS = [200]
        config.EVAL_PLAYER_COUNTS = [3]

        with tempfile.TemporaryDirectory() as td:
            ck_dir = Path(td) / "models"
            metrics_path = Path(td) / "metrics.json"
            t0 = time.perf_counter()
            run_training(
                max_iterations=2,
                games_per_iteration=2,
                training_steps_per_iteration=5,
                sims_per_move=5,
                batch_size=16,
                checkpoint_dir=ck_dir,
                metrics_path=metrics_path,
                seed=0,
                workers=1,
            )
            elapsed = time.perf_counter() - t0

            with open(metrics_path) as f:
                data = json.load(f)
            assert "eval_overall_win_rate" in data["history"], "eval did not log to history"
            assert len(data["history"]["eval_overall_win_rate"]) >= 1
            latest = data["latest"]
            assert "eval_results" in latest, "latest snapshot missing eval_results"
            eval_results = latest["eval_results"]
            assert "win_rates" in eval_results
            print(
                f"  ok test_end_to_end_training_with_evaluation "
                f"({elapsed:.1f}s, overall_wr={latest.get('eval_overall_win_rate', 0):.3f})"
            )
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_play_match_ai_vs_threshold,
    test_play_match_two_bots,
    test_evaluate_against_bots_full,
    test_evaluation_result_to_dict_jsonable,
    test_frozen_net_used_in_self_play,
    test_evaluate_against_checkpoint_with_saved_file,
    test_evaluator_run_no_old_checkpoint,
    test_evaluator_run_with_old_checkpoint,
    test_end_to_end_training_with_evaluation,
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

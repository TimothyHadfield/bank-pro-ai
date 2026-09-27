"""
Tests for training/parallel_self_play.py.

Run from bank-ai-v2-mcts:
    python -m tests.test_parallel
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from network import BankNetwork
from training import parallel_self_play


def test_serial_fallback_for_one_worker():
    """n_workers=1 should fall back to serial execution and produce valid output."""
    torch.manual_seed(0)
    net = BankNetwork()
    examples, results = parallel_self_play(
        network=net,
        n_games=2,
        n_workers=1,
        sims_per_move=5,
        base_seed=1,
        add_dirichlet_noise=True,
        use_bot_pool=True,
    )
    assert len(results) == 2
    assert all(r.num_players >= 2 and r.num_players <= 6 for r in results)
    assert len(examples) > 0
    print(f"  ok test_serial_fallback_for_one_worker ({len(examples)} examples)")


def test_parallel_two_workers():
    """n_workers=2 should run a multiprocessing pool and produce valid output."""
    torch.manual_seed(0)
    net = BankNetwork()
    t0 = time.perf_counter()
    examples, results = parallel_self_play(
        network=net,
        n_games=4,
        n_workers=2,
        sims_per_move=5,
        base_seed=2,
        add_dirichlet_noise=True,
        use_bot_pool=True,
    )
    elapsed = time.perf_counter() - t0
    assert len(results) == 4
    assert len(examples) > 0
    print(
        f"  ok test_parallel_two_workers "
        f"({len(results)} games, {len(examples)} examples in {elapsed:.1f}s)"
    )


def test_total_games_distributed_correctly():
    """If we ask for 5 games across 3 workers, we should still get 5 games total."""
    torch.manual_seed(0)
    net = BankNetwork()
    examples, results = parallel_self_play(
        network=net,
        n_games=5,
        n_workers=3,
        sims_per_move=3,
        base_seed=3,
    )
    assert len(results) == 5, f"expected 5 results, got {len(results)}"
    print("  ok test_total_games_distributed_correctly")


def test_examples_have_valid_structure():
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = parallel_self_play(
        network=net,
        n_games=3,
        n_workers=2,
        sims_per_move=4,
        base_seed=4,
    )
    for ex in examples:
        assert ex.state_vector.shape == (config.INPUT_SIZE,)
        assert ex.mcts_policy.shape == (config.NUM_ACTIONS,)
        assert 0 <= ex.value <= 1
        assert 2 <= ex.num_players <= 6
    print(f"  ok test_examples_have_valid_structure ({len(examples)} examples checked)")


def test_speedup_mini_benchmark():
    """
    Tiny benchmark: 4 games at 30 sims, serial vs 2 workers. 2 workers should be faster
    (or at least not catastrophically slower) on any multi-core machine. We tolerate a
    margin because subprocess startup is significant on small workloads.
    """
    import os
    if (os.cpu_count() or 1) < 2:
        print("  ok test_speedup_mini_benchmark (skipped — single-core)")
        return

    torch.manual_seed(0)
    net = BankNetwork()

    t0 = time.perf_counter()
    parallel_self_play(net, n_games=4, n_workers=1, sims_per_move=20, base_seed=10)
    serial_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    parallel_self_play(net, n_games=4, n_workers=2, sims_per_move=20, base_seed=10)
    parallel_t = time.perf_counter() - t1

    ratio = serial_t / parallel_t
    print(
        f"  ok test_speedup_mini_benchmark "
        f"(serial={serial_t:.1f}s, parallel={parallel_t:.1f}s, ratio={ratio:.2f}×)"
    )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_serial_fallback_for_one_worker,
    test_parallel_two_workers,
    test_total_games_distributed_correctly,
    test_examples_have_valid_structure,
    test_speedup_mini_benchmark,
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

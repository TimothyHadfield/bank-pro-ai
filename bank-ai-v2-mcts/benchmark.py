"""
Benchmark script for Bank AI v2.

Measures self-play throughput for different worker counts and sims-per-move settings.
Useful for tuning the training loop before kicking off long runs.

Usage:
    python benchmark.py                        # quick default run
    python benchmark.py --workers 1 2 4 8      # specific worker counts
    python benchmark.py --sims 50 100 200      # specific sim counts
    python benchmark.py --games 20             # how many games per measurement
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import config
from network import BankNetwork
from training import parallel_self_play


def benchmark_one(
    network: BankNetwork,
    n_games: int,
    n_workers: int,
    sims_per_move: int,
) -> dict:
    t0 = time.perf_counter()
    examples, results = parallel_self_play(
        network=network,
        n_games=n_games,
        n_workers=n_workers,
        sims_per_move=sims_per_move,
        base_seed=12345,
        add_dirichlet_noise=True,
        use_bot_pool=True,
    )
    elapsed = time.perf_counter() - t0
    n_decisions = len(examples)
    games_per_sec = n_games / elapsed
    decisions_per_sec = n_decisions / elapsed
    return {
        "n_games": n_games,
        "n_workers": n_workers,
        "sims_per_move": sims_per_move,
        "elapsed_seconds": elapsed,
        "games_per_sec": games_per_sec,
        "games_per_min": games_per_sec * 60,
        "decisions_per_sec": decisions_per_sec,
        "decisions": n_decisions,
        "n_results": len(results),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, nargs="+", default=[1, max(2, os.cpu_count() // 2 or 1), os.cpu_count() or 1])
    p.add_argument("--sims", type=int, nargs="+", default=[50, 100])
    p.add_argument("--games", type=int, default=8)
    args = p.parse_args()

    print(f"CPU count: {os.cpu_count()}")
    print(f"Benchmarking {args.games} games per measurement")
    print(f"Worker counts: {args.workers}")
    print(f"Sims/move:     {args.sims}")
    print()

    torch.manual_seed(0)
    network = BankNetwork()

    rows = []
    for sims in args.sims:
        for w in args.workers:
            print(f"  workers={w:>2d} sims/move={sims:>3d} ...", end=" ", flush=True)
            r = benchmark_one(network, args.games, w, sims)
            rows.append(r)
            print(
                f"{r['elapsed_seconds']:6.2f}s  "
                f"{r['games_per_min']:6.1f} games/min  "
                f"{r['decisions_per_sec']:5.1f} dec/s"
            )

    print()
    print("Summary:")
    print(f"{'workers':>8} {'sims':>5} {'games/min':>11} {'speedup vs 1w':>14}")
    # Compute speedups within each sims setting
    by_sims = {}
    for r in rows:
        by_sims.setdefault(r["sims_per_move"], []).append(r)
    for sims, group in by_sims.items():
        # Find baseline (workers=1) for this sims
        base = next((g for g in group if g["n_workers"] == 1), None)
        for r in group:
            if base:
                speedup = r["games_per_min"] / base["games_per_min"]
                speedup_str = f"{speedup:.2f}×"
            else:
                speedup_str = "—"
            print(
                f"{r['n_workers']:>8} {r['sims_per_move']:>5} "
                f"{r['games_per_min']:>11.1f} {speedup_str:>14}"
            )


if __name__ == "__main__":
    main()

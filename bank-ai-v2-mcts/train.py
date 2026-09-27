"""
Bank AI v2 — main training loop.

One iteration = generate self-play games + train on minibatches sampled from the buffer.
Periodic checkpoint and (eventually) evaluation. Watchdog auto-halts on broken metrics.

Usage:
    python train.py                              # start fresh
    python train.py --resume models/latest.pt    # resume from a checkpoint
    python train.py --max-iterations 100         # stop after N iterations

Outputs:
    metrics.json            (rolling metrics for the dashboard)
    models/latest.pt        (most recent checkpoint)
    models/iter_<N>.pt      (periodic checkpoints)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from evaluation import Evaluator
from monitoring import MetricsLogger
from network import BankNetwork
from training import (
    ReplayBuffer,
    Trainer,
    Watchdog,
    build_default_bot_pool,
    load_checkpoint,
    parallel_self_play,
    save_checkpoint,
    self_play_game,
)
from training.parallel_self_play import shutdown_pool as _shutdown_worker_pool


# ---------------------------------------------------------------- helpers

def _policy_entropy_from_examples(examples) -> float:
    """Average entropy of MCTS visit distributions in this batch of examples (in nats)."""
    if not examples:
        return 0.0
    entropies = []
    for ex in examples:
        p = ex.mcts_policy
        # avoid log(0)
        e = -float((p * np.log(p + 1e-12)).sum())
        entropies.append(e)
    return float(np.mean(entropies))


# Rolling cache of the most-recent eval result per (opponent, player_count) so that
# rotation gives the dashboard accumulated coverage rather than only the panel just run.
_eval_panel_cache: dict[tuple[str, int], dict] = {}


def _merge_eval_into_logger(logger, eval_result) -> None:
    """
    Merge this eval round's panels into the running cache, then log the union as
    `eval_results` so the dashboard always shows the most recent value for every
    (opponent, player_count) pair we've ever evaluated.
    """
    for op_name, by_pc in eval_result.win_rates.items():
        for pc, rate in by_pc.items():
            _eval_panel_cache[(op_name, pc)] = {
                "win_rate": rate,
                "details": eval_result.details.get(op_name, {}).get(pc, {}),
                "iteration": eval_result.iteration,
            }
    # Reconstruct an EvaluationResult-like dict from the cache
    win_rates: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for (op_name, pc), info in _eval_panel_cache.items():
        win_rates.setdefault(op_name, {})[str(pc)] = info["win_rate"]
        details.setdefault(op_name, {})[str(pc)] = info["details"]

    # Compute overall win rate as the mean across all cached panels
    all_rates = [info["win_rate"] for info in _eval_panel_cache.values()]
    overall = sum(all_rates) / len(all_rates) if all_rates else 0.0

    # Per-player-count averages (averaged across bots for that player count)
    rates_by_pc: dict[int, list[float]] = {}
    for (op_name, pc), info in _eval_panel_cache.items():
        rates_by_pc.setdefault(pc, []).append(info["win_rate"])

    merged = {
        "iteration": eval_result.iteration,
        "win_rates": win_rates,
        "details": details,
        "overall_win_rate": overall,
    }
    log_payload = {
        "eval_overall_win_rate": overall,
        "eval_results": merged,
    }
    for pc, rates in rates_by_pc.items():
        log_payload[f"eval_win_rate_{pc}p"] = sum(rates) / len(rates)
    logger.log(eval_result.iteration, log_payload)


def _pick_frozen_opponents(checkpoint_dir: Path, current_iter: int, max_count: int = 4) -> list[str]:
    """
    Return a list of paths to past iter_*.pt checkpoints suitable for use as frozen opponents.
    Picks up to `max_count` of the most recent past checkpoints (excluding the current iteration).
    Returns [] if no past checkpoints exist yet.
    """
    if not checkpoint_dir.exists():
        return []
    candidates = []
    for p in checkpoint_dir.glob("iter_*.pt"):
        try:
            iter_num = int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if 0 < iter_num < current_iter:
            candidates.append((iter_num, p))
    candidates.sort(reverse=True)
    return [str(p) for _, p in candidates[:max_count]]


def _per_player_count_win_stats(results, ai_self_play_only: bool = True) -> dict:
    """
    Compute win rate per player count over a list of GameResult objects.
    If ai_self_play_only, only games where ALL seats were AI count toward win rate.
    """
    by_count: dict[int, dict] = {}
    for r in results:
        if ai_self_play_only and r.num_bot_seats > 0:
            continue
        bucket = by_count.setdefault(r.num_players, {"games": 0, "winners": 0})
        bucket["games"] += 1
        bucket["winners"] += len(r.winners)
    out = {}
    for n, b in by_count.items():
        if b["games"] == 0:
            continue
        # In an N-player AI-only game, the AI wins exactly len(winners) seats per game,
        # so "AI seats winning per game" / N is the per-seat win rate (which trivially
        # equals 1/N on average when all seats use the same network and play deterministically
        # over many games, but tracks above 1/N as the AI improves vs older versions).
        out[str(n)] = b["winners"] / (b["games"] * n)
    return out


# ---------------------------------------------------------------- training loop

def run_training(
    max_iterations: int = 1_000_000,
    games_per_iteration: int = config.GAMES_PER_ITERATION,
    training_steps_per_iteration: int = config.TRAINING_STEPS_PER_ITERATION,
    sims_per_move: int = config.SIMS_PER_MOVE,
    batch_size: int = config.BATCH_SIZE,
    checkpoint_dir: str | Path = "models",
    metrics_path: str | Path = "metrics.json",
    resume_path: str | None = None,
    seed: int = config.DEFAULT_SEED,
    workers: int = 1,
    eval_interval: int = config.EVAL_INTERVAL_ITERATIONS,
    eval_games_per_player_count: int = 5,
    eval_disabled: bool = False,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(seed)
    np_master_seed = seed

    # Network + trainer
    network = BankNetwork()
    trainer = Trainer(network)
    iteration = 0

    if resume_path:
        info = load_checkpoint(resume_path, network, trainer)
        iteration = info["iteration"]
        print(f"resumed from {resume_path} at iteration {iteration}")

    buffer = ReplayBuffer()
    watchdog = Watchdog()
    logger = MetricsLogger(metrics_path=metrics_path)
    logger.set_status("running", "training started")
    evaluator = Evaluator()

    bot_pool = build_default_bot_pool(seed=seed)

    print(
        f"starting training: max_iterations={max_iterations}, "
        f"games/iter={games_per_iteration}, steps/iter={training_steps_per_iteration}, "
        f"sims/move={sims_per_move}, batch={batch_size}"
    )

    try:
        while iteration < max_iterations:
            iteration += 1
            iter_start = time.perf_counter()

            # ---- pick frozen-checkpoint opponents from past iterations ----
            frozen_checkpoint_paths = _pick_frozen_opponents(checkpoint_dir, current_iter=iteration)

            # ---- self-play phase ----
            sp_start = time.perf_counter()
            if workers > 1:
                all_examples, results = parallel_self_play(
                    network=network,
                    n_games=games_per_iteration,
                    n_workers=workers,
                    sims_per_move=sims_per_move,
                    base_seed=np_master_seed + iteration * 1_000_000,
                    add_dirichlet_noise=True,
                    use_bot_pool=True,
                    frozen_checkpoint_paths=frozen_checkpoint_paths,
                )
            else:
                # Materialize frozen networks once for the iteration
                from training import FrozenNet
                frozen_nets = []
                for p in frozen_checkpoint_paths:
                    try:
                        fnet = BankNetwork()
                        load_checkpoint(p, fnet)
                        fnet.eval()
                        frozen_nets.append(FrozenNet(
                            network=fnet, name=Path(p).name, sims_per_move=sims_per_move,
                        ))
                    except Exception as e:
                        print(f"could not load frozen {p}: {e}")
                all_examples = []
                results = []
                for g in range(games_per_iteration):
                    examples, result = self_play_game(
                        network=network,
                        num_players=None,
                        sims_per_move=sims_per_move,
                        add_dirichlet_noise=True,
                        seed=np_master_seed + iteration * 10_000 + g,
                        bot_pool=bot_pool,
                        frozen_nets=frozen_nets if frozen_nets else None,
                    )
                    all_examples.extend(examples)
                    results.append(result)
            sp_time = time.perf_counter() - sp_start

            # Add to buffer (tag with iteration)
            buffer.extend(all_examples, iteration=iteration)
            buffer.set_iteration(iteration)

            # Update LR if we've crossed the decay boundary
            if iteration == config.LEARNING_RATE_DECAY_AT_ITER:
                trainer.set_learning_rate(config.LEARNING_RATE_DECAYED)
                print(f"learning rate decayed to {config.LEARNING_RATE_DECAYED}")

            # ---- training phase ----
            train_start = time.perf_counter()
            losses = []
            for _ in range(training_steps_per_iteration):
                if len(buffer) < batch_size:
                    bs = max(1, len(buffer))
                else:
                    bs = batch_size
                batch = buffer.sample(bs)
                step_result = trainer.train_step(batch)
                losses.append(step_result)
            train_time = time.perf_counter() - train_start

            avg_total = np.mean([s.total_loss for s in losses])
            avg_policy = np.mean([s.policy_loss for s in losses])
            avg_value = np.mean([s.value_loss for s in losses])
            avg_l2 = np.mean([s.l2_loss for s in losses])
            mcts_entropy = _policy_entropy_from_examples(all_examples)
            ai_only_win_rates = _per_player_count_win_stats(results)

            iter_time = time.perf_counter() - iter_start
            games_per_min = (games_per_iteration / iter_time) * 60.0

            metrics = {
                "total_loss": float(avg_total),
                "policy_loss": float(avg_policy),
                "value_loss": float(avg_value),
                "l2_loss": float(avg_l2),
                "mcts_visit_entropy": float(mcts_entropy),
                "buffer_size": len(buffer),
                "buffer_avg_age": buffer.stats().get("average_age") or 0.0,
                "self_play_seconds": sp_time,
                "training_seconds": train_time,
                "iteration_seconds": iter_time,
                "games_per_min": games_per_min,
                "ai_only_win_rate_by_player_count": ai_only_win_rates,
            }
            logger.log(iteration, metrics)

            # ---- watchdog ----
            status = watchdog.check(
                iteration=iteration,
                total_loss=float(avg_total),
                policy_entropy=float(mcts_entropy),
            )
            if status.halted:
                logger.set_status("halted", status.reason)
                print(f"WATCHDOG HALT @ iter {iteration}: {status.reason}")
                # Still save the checkpoint of what we had
                save_checkpoint(
                    checkpoint_dir / f"halt_iter_{iteration}.pt",
                    network,
                    trainer,
                    iteration,
                    metadata={"halt_reason": status.reason},
                )
                return

            # ---- checkpoints ----
            if iteration % config.CHECKPOINT_INTERVAL_ITERATIONS == 0:
                save_checkpoint(
                    checkpoint_dir / f"iter_{iteration}.pt",
                    network,
                    trainer,
                    iteration,
                    metadata=metrics,
                )
            # Always update the "latest" pointer
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                network,
                trainer,
                iteration,
                metadata=metrics,
            )

            # ---- periodic evaluation ----
            # Skipped if eval_disabled=True or eval_interval<=0.
            # Rotation: each eval covers ONE player count from EVAL_PLAYER_COUNTS, cycling
            # through them on successive eval calls. This keeps eval cost ~3-5% of training
            # time. Over a few evals, you see all player counts.
            if (not eval_disabled) and eval_interval > 0 and iteration % eval_interval == 0:
                # Pick which player count to evaluate this round (rotate)
                eval_round = iteration // eval_interval
                pc_list = config.EVAL_PLAYER_COUNTS
                rotated_pc = [pc_list[(eval_round - 1) % len(pc_list)]]

                # Pick the most recent OLDER checkpoint to compare against
                old_ck = None
                older_iter = iteration - eval_interval
                if older_iter > 0:
                    candidate = checkpoint_dir / f"iter_{older_iter}.pt"
                    if candidate.exists():
                        old_ck = str(candidate)

                # Estimate eval cost up front
                est_bot_games = (
                    len(config.THRESHOLD_BOT_THRESHOLDS)
                    * len(rotated_pc)
                    * eval_games_per_player_count
                )
                est_ck_games = (
                    len(rotated_pc) * max(2, eval_games_per_player_count // 2)
                ) if old_ck else 0
                print(
                    f"[iter {iteration}] starting evaluation: "
                    f"{est_bot_games} bot games (player count {rotated_pc[0]})"
                    + (f" + {est_ck_games} vs {Path(old_ck).name}" if old_ck else "")
                    + f" at sims={sims_per_move}",
                    flush=True,
                )
                eval_start = time.perf_counter()
                eval_result = evaluator.run(
                    network=network,
                    iteration=iteration,
                    sims_per_move=sims_per_move,
                    games_per_player_count=eval_games_per_player_count,
                    player_counts=rotated_pc,                       # one player count per call
                    bot_thresholds=config.THRESHOLD_BOT_THRESHOLDS,
                    old_checkpoint_path=old_ck,
                    base_seed=iteration * 17,
                )
                eval_time = time.perf_counter() - eval_start
                print(
                    f"[iter {iteration}] eval done in {eval_time:.1f}s, "
                    f"overall win rate {eval_result.overall_win_rate():.3f}",
                    flush=True,
                )
                # Merge into a per-player-count rolling result so the dashboard sees
                # accumulated coverage rather than only the latest single panel.
                _merge_eval_into_logger(logger, eval_result)

    except KeyboardInterrupt:
        print("\ntraining interrupted by user; saving final checkpoint")
        save_checkpoint(
            checkpoint_dir / "latest.pt", network, trainer, iteration, metadata={"interrupted": True}
        )
        logger.set_status("idle", "interrupted by user")
    except Exception as e:
        logger.set_status("error", f"{type(e).__name__}: {e}")
        raise
    finally:
        # Always tear down worker processes so they don't linger after training stops.
        _shutdown_worker_pool()

    logger.set_status("idle", "max iterations reached")


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Bank AI v2 training loop")
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--games-per-iter", type=int, default=config.GAMES_PER_ITERATION)
    parser.add_argument("--steps-per-iter", type=int, default=config.TRAINING_STEPS_PER_ITERATION)
    parser.add_argument("--sims-per-move", type=int, default=config.SIMS_PER_MOVE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS,
                        help="Number of parallel self-play workers (>=1). Use # of CPU cores.")
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL_ITERATIONS,
                        help="Run evaluation every N iterations. Set 0 to disable eval entirely.")
    parser.add_argument("--eval-games", type=int, default=config.EVAL_GAMES_PER_OPPONENT,
                        help="Games per bot per eval. Each eval rotates one player count.")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation entirely (fastest).")
    args = parser.parse_args()

    run_training(
        max_iterations=args.max_iterations,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.steps_per_iter,
        sims_per_move=args.sims_per_move,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        resume_path=args.resume,
        seed=args.seed,
        workers=args.workers,
        eval_interval=args.eval_interval,
        eval_games_per_player_count=args.eval_games,
        eval_disabled=args.no_eval,
    )


if __name__ == "__main__":
    main()

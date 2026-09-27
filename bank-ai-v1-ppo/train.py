"""
Entry point: run this to start training the Bank AI via self-play + PPO.

Usage:
    python train.py
    python train.py --resume models/checkpoint_500.pt
"""
import os
import sys
import time
import json
import argparse
import torch
import numpy as np

from config import Config
from game.environment import BankGame
from network.model import BankNetwork
from training.self_play import run_self_play, update_frozen_bots
from training.trainer import PPOTrainer
from training.watchdog import Watchdog
from monitoring.logger import TrainingLogger


OVERRIDES_PATH = "dashboard/overrides.json"


def load_overrides(config):
    """Load runtime overrides from the dashboard (batch_size, reward_win_bonus)."""
    if not os.path.exists(OVERRIDES_PATH):
        return
    try:
        with open(OVERRIDES_PATH, "r") as f:
            overrides = json.load(f)
        if "batch_size" in overrides:
            config.batch_size = int(overrides["batch_size"])
        if "entropy_coefficient" in overrides:
            config.entropy_coefficient = float(overrides["entropy_coefficient"])
        if "watchdog_enabled" in overrides:
            config.watchdog_enabled = bool(overrides["watchdog_enabled"])
    except (json.JSONDecodeError, OSError, ValueError):
        pass


def evaluate_vs_threshold(network, config: Config, threshold: int, num_games: int = 200):
    network.eval()
    ai_wins = 0

    for _ in range(num_games):
        game = BankGame(num_players=config.num_players, num_rounds=config.num_rounds)
        game.reset()

        while not game.game_over:
            if not game.waiting_for_decisions:
                break

            actions = {}
            for pid in game.players_in_round:
                if pid == 0:
                    state_vec = game.get_state_vector(pid)
                    with torch.no_grad():
                        action, _, _ = network.select_action(state_vec, deterministic=True)
                    actions[pid] = "bank" if action == 0 else "continue"
                else:
                    if game.pot > threshold:
                        actions[pid] = "bank"
                    else:
                        actions[pid] = "continue"

            game.step(actions)

        if 0 in game.get_winners():
            ai_wins += 1

    return ai_wins / num_games


def save_checkpoint(path, network, trainer, batch_num, watchdog, config):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": trainer.get_optimizer_state(),
        "batch_num": batch_num,
        "watchdog_state": watchdog.get_state(),
        "config": vars(config),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Bank AI")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    trainer = PPOTrainer(network, config)
    watchdog = Watchdog(config)
    logger = TrainingLogger(config)

    start_batch = 1

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        network.load_state_dict(checkpoint["network_state_dict"])
        trainer.load_optimizer_state(checkpoint["optimizer_state_dict"])
        start_batch = checkpoint["batch_num"] + 1
        if "watchdog_state" in checkpoint:
            watchdog.load_state(checkpoint["watchdog_state"])
        print(f"Resumed at batch {start_batch}")

    os.makedirs(config.model_dir, exist_ok=True)

    print(f"Starting training: {config.max_training_batches} batches, "
          f"{config.batch_size} games/batch, {config.num_players} players")
    print(f"Checkpoints every {config.checkpoint_interval} batches, "
          f"evaluations every {config.eval_interval} batches")
    print()

    batch_times = []

    for batch_num in range(start_batch, config.max_training_batches + 1):
        batch_start = time.time()

        # Load runtime overrides from dashboard (checked each batch)
        load_overrides(config)

        # Snapshot current AI as a frozen opponent periodically
        update_frozen_bots(network, batch_num, config)

        # Step 1: Self-play
        experience = run_self_play(network, config)

        # Step 2: PPO update
        update_metrics = trainer.update(
            experience["states"],
            experience["actions"],
            experience["log_probs"],
            experience["returns"],
            experience["advantages"],
        )

        # Step 3: Log metrics
        batch_elapsed = time.time() - batch_start
        metrics = {
            "avg_score": experience["avg_score"],
            "avg_reward": experience["avg_reward"],
            "win_rate": experience.get("win_rate", 0.0),
            "batch_time": round(batch_elapsed, 2),
            **update_metrics,
        }
        logger.log_batch(batch_num, metrics)

        # Step 4: Watchdog check (only halts if enabled)
        watchdog.check(network, metrics, experience["action_distribution"])
        logger.log_watchdog(watchdog.get_dashboard_status())
        if config.watchdog_enabled and watchdog.halted:
            save_checkpoint(
                os.path.join(config.model_dir, "checkpoint_halted.pt"),
                network, trainer, batch_num, watchdog, config,
            )
            print(f"Halted checkpoint saved to {config.model_dir}/checkpoint_halted.pt")
            break

        # Step 5: Periodic checkpoint
        if batch_num % config.checkpoint_interval == 0:
            path = os.path.join(config.model_dir, f"checkpoint_{batch_num}.pt")
            save_checkpoint(path, network, trainer, batch_num, watchdog, config)
            print(f"  Checkpoint saved: {path}")

        # Step 6: Periodic evaluation
        if batch_num % config.eval_interval == 0:
            wr100 = evaluate_vs_threshold(network, config, threshold=100)
            wr200 = evaluate_vs_threshold(network, config, threshold=200)
            logger.log_evaluation(batch_num, {
                "win_rate_vs_100": wr100,
                "win_rate_vs_200": wr200,
            })

        # Timing estimate
        batch_times.append(batch_elapsed)
        if len(batch_times) > 50:
            batch_times = batch_times[-50:]
        avg_time = sum(batch_times) / len(batch_times)
        remaining = (config.max_training_batches - batch_num) * avg_time
        if batch_num % 10 == 0:
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            print(f"  [{avg_time:.1f}s/batch, ~{hours}h {minutes}m remaining]")

    logger.close()
    final_path = os.path.join(config.model_dir, f"checkpoint_final_{batch_num}.pt")
    save_checkpoint(final_path, network, trainer, batch_num, watchdog, config)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

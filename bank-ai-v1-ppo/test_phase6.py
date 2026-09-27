"""
Phase 6 Test: Run training for 10 batches with small batch size,
verify checkpoints, logs, and no crashes.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from config import Config
from game.environment import BankGame
from network.model import BankNetwork
from training.self_play import run_self_play
from training.trainer import PPOTrainer
from training.watchdog import Watchdog
from monitoring.logger import TrainingLogger
from train import evaluate_vs_threshold, save_checkpoint


def test_training_loop():
    print("=" * 60)
    print("TEST: Training loop (10 batches, 20 games/batch)")
    print("=" * 60)

    config = Config()
    config.batch_size = 20  # Small for testing
    config.checkpoint_interval = 5
    config.eval_interval = 5

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    trainer = PPOTrainer(network, config)
    watchdog = Watchdog(config)
    logger = TrainingLogger(config)
    logger.config.log_dir = "runs/test_phase6"

    os.makedirs(config.model_dir, exist_ok=True)

    for batch_num in range(1, 11):
        experience = run_self_play(network, config)
        update_metrics = trainer.update(
            experience["states"],
            experience["actions"],
            experience["log_probs"],
            experience["returns"],
            experience["advantages"],
        )

        metrics = {
            "avg_score": experience["avg_score"],
            "avg_reward": experience["avg_reward"],
            **update_metrics,
        }
        logger.log_batch(batch_num, metrics)

        halted = watchdog.check(network, metrics, experience["action_distribution"])
        assert not halted, f"Watchdog halted unexpectedly at batch {batch_num}: {watchdog.halt_reason}"

        if batch_num % config.checkpoint_interval == 0:
            path = os.path.join(config.model_dir, f"checkpoint_test_{batch_num}.pt")
            save_checkpoint(path, network, trainer, batch_num, watchdog, config)
            print(f"  Checkpoint saved: {path}")

        if batch_num % config.eval_interval == 0:
            wr100 = evaluate_vs_threshold(network, config, threshold=100, num_games=20)
            wr200 = evaluate_vs_threshold(network, config, threshold=200, num_games=20)
            logger.log_evaluation(batch_num, {
                "win_rate_vs_100": wr100,
                "win_rate_vs_200": wr200,
            })

    logger.close()

    # Verify checkpoint exists and loads
    ckpt_path = os.path.join(config.model_dir, "checkpoint_test_5.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert "network_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "batch_num" in checkpoint
    assert checkpoint["batch_num"] == 5
    print(f"\n  Checkpoint verified: batch_num={checkpoint['batch_num']}")

    # Verify TensorBoard logs exist
    assert os.path.exists("runs/test_phase6") or os.path.exists("runs"), "TensorBoard logs not found"

    # Clean up test checkpoints
    for f in os.listdir(config.model_dir):
        if f.startswith("checkpoint_test_"):
            os.remove(os.path.join(config.model_dir, f))

    print("\nPASSED: Training loop runs correctly for 10 batches.\n")


def test_resume():
    print("=" * 60)
    print("TEST: Checkpoint resume")
    print("=" * 60)

    config = Config()
    config.batch_size = 10
    torch.manual_seed(config.seed)

    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    trainer = PPOTrainer(network, config)
    watchdog = Watchdog(config)

    # Save a checkpoint
    os.makedirs(config.model_dir, exist_ok=True)
    path = os.path.join(config.model_dir, "checkpoint_resume_test.pt")
    save_checkpoint(path, network, trainer, 42, watchdog, config)

    # Load into new network
    network2 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    checkpoint = torch.load(path, weights_only=False)
    network2.load_state_dict(checkpoint["network_state_dict"])

    # Verify weights match
    for p1, p2 in zip(network.parameters(), network2.parameters()):
        assert torch.equal(p1, p2), "Weights should match after loading"

    os.remove(path)
    print("PASSED: Checkpoint resume works correctly.\n")


if __name__ == "__main__":
    test_training_loop()
    test_resume()

    print("=" * 60)
    print("ALL PHASE 6 TESTS PASSED!")
    print("=" * 60)

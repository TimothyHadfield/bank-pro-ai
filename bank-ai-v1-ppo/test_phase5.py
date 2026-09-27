"""
Phase 5 Test: Verify logger and watchdog work correctly.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from config import Config
from network.model import BankNetwork
from game.environment import BankGame
from monitoring.logger import TrainingLogger
from training.watchdog import Watchdog


def test_logger():
    print("=" * 60)
    print("TEST 1: Logger")
    print("=" * 60)

    config = Config()
    config.log_dir = "runs/test"
    logger = TrainingLogger(config)

    # Log some dummy batches
    for i in range(3):
        logger.log_batch(i, {
            "avg_score": 100.0 + i * 10,
            "avg_reward": 0.5 + i * 0.1,
            "policy_loss": 0.1 - i * 0.01,
            "value_loss": 0.5 - i * 0.05,
            "entropy": 0.69 - i * 0.01,
        })

    logger.log_evaluation(3, {
        "win_rate_vs_100": 0.65,
        "win_rate_vs_200": 0.55,
    })

    logger.close()
    print("PASSED: Logger works correctly.\n")


def test_watchdog_normal():
    print("=" * 60)
    print("TEST 2: Watchdog - normal operation (no halt)")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    watchdog = Watchdog(config)

    for i in range(15):
        should_halt = watchdog.check(
            network,
            {"policy_loss": 0.1, "value_loss": 0.5, "avg_score": 200.0, "entropy": 0.6},
            {0: 500, 1: 500},
        )
        assert not should_halt, f"Should not halt on normal metrics (batch {i})"

    print("PASSED: No false halts on normal metrics.\n")


def test_watchdog_nan_weights():
    print("=" * 60)
    print("TEST 3: Watchdog - NaN in weights")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    watchdog = Watchdog(config)

    # Inject NaN into weights
    with torch.no_grad():
        list(network.parameters())[0][0][0] = float('nan')

    should_halt = watchdog.check(
        network,
        {"policy_loss": 0.1, "value_loss": 0.5, "avg_score": 200.0},
        {0: 500, 1: 500},
    )
    assert should_halt, "Should halt on NaN weights"
    print(f"  Halt reason: {watchdog.halt_reason}")
    print("PASSED: Detects NaN in weights.\n")


def test_watchdog_nan_loss():
    print("=" * 60)
    print("TEST 4: Watchdog - NaN in loss")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    watchdog = Watchdog(config)

    should_halt = watchdog.check(
        network,
        {"policy_loss": float('nan'), "value_loss": 0.5, "avg_score": 200.0},
        {0: 500, 1: 500},
    )
    assert should_halt, "Should halt on NaN loss"
    print(f"  Halt reason: {watchdog.halt_reason}")
    print("PASSED: Detects NaN in loss.\n")


def test_watchdog_policy_collapse():
    print("=" * 60)
    print("TEST 5: Watchdog - policy collapse")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    watchdog = Watchdog(config)

    # Simulate 10 batches where >95% of actions are the same
    for i in range(config.watchdog_collapse_patience):
        should_halt = watchdog.check(
            network,
            {"policy_loss": 0.1, "value_loss": 0.5, "avg_score": 200.0, "entropy": 0.01},
            {0: 960, 1: 40},  # 96% bank
        )
        if should_halt:
            break

    assert should_halt, "Should halt on policy collapse"
    print(f"  Halt reason: {watchdog.halt_reason}")
    print("PASSED: Detects policy collapse.\n")


def test_watchdog_state_persistence():
    print("=" * 60)
    print("TEST 6: Watchdog - state save/load")
    print("=" * 60)

    config = Config()
    watchdog = Watchdog(config)
    watchdog.peak_score = 500.0
    watchdog.below_peak_count = 5
    watchdog.policy_loss_history.append(0.1)

    state = watchdog.get_state()
    watchdog2 = Watchdog(config)
    watchdog2.load_state(state)

    assert watchdog2.peak_score == 500.0
    assert watchdog2.below_peak_count == 5
    assert len(watchdog2.policy_loss_history) == 1
    print("PASSED: Watchdog state persists correctly.\n")


if __name__ == "__main__":
    test_logger()
    test_watchdog_normal()
    test_watchdog_nan_weights()
    test_watchdog_nan_loss()
    test_watchdog_policy_collapse()
    test_watchdog_state_persistence()

    print("=" * 60)
    print("ALL PHASE 5 TESTS PASSED!")
    print("=" * 60)

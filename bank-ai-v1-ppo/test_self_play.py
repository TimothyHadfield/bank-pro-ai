"""
Phase 4 Test: Verify self-play produces valid experience data.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from game.environment import BankGame
from network.model import BankNetwork
from training.self_play import run_self_play
from config import Config


def test_self_play():
    print("=" * 60)
    print("TEST: Self-play data collection")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)

    # Run a small batch
    num_games = 10
    result = run_self_play(network, config, num_games=num_games)

    print(f"  Games played: {num_games}")
    print(f"  Total experiences collected: {len(result['states'])}")
    print(f"  States shape: {result['states'].shape}")
    print(f"  Actions shape: {result['actions'].shape}")
    print(f"  Log probs shape: {result['log_probs'].shape}")
    print(f"  Returns shape: {result['returns'].shape}")
    print(f"  Advantages shape: {result['advantages'].shape}")
    print(f"  Avg score: {result['avg_score']:.2f}")
    print(f"  Avg reward: {result['avg_reward']:.4f}")
    print(f"  Action distribution: bank={result['action_distribution'][0]}, "
          f"continue={result['action_distribution'][1]}")

    n = len(result['states'])
    assert n > 0, "Should have collected some experiences"
    assert result['states'].shape == (n, state_size), "States shape mismatch"
    assert result['actions'].shape == (n,), "Actions shape mismatch"
    assert result['log_probs'].shape == (n,), "Log probs shape mismatch"
    assert result['returns'].shape == (n,), "Returns shape mismatch"
    assert result['advantages'].shape == (n,), "Advantages shape mismatch"

    # Check actions are valid (0 or 1)
    assert (result['actions'] >= 0).all() and (result['actions'] <= 1).all(), "Invalid actions"

    # Check log probs are negative
    assert (result['log_probs'] <= 0).all(), "Log probs should be <= 0"

    # Check advantages are normalized (mean ~0, std ~1)
    adv_mean = result['advantages'].mean().item()
    adv_std = result['advantages'].std().item()
    print(f"  Advantages mean: {adv_mean:.4f}, std: {adv_std:.4f}")
    assert abs(adv_mean) < 0.1, f"Advantages should be ~0 mean, got {adv_mean}"
    assert abs(adv_std - 1.0) < 0.2, f"Advantages should be ~1 std, got {adv_std}"

    # Check no NaNs
    assert not torch.isnan(result['states']).any(), "NaN in states"
    assert not torch.isnan(result['returns']).any(), "NaN in returns"
    assert not torch.isnan(result['advantages']).any(), "NaN in advantages"

    print("\nPASSED: Self-play produces valid experience data.\n")

    # Run a slightly larger batch to verify stability
    print("Running 50 games to verify stability...")
    result2 = run_self_play(network, config, num_games=50)
    print(f"  50-game batch: {len(result2['states'])} experiences, "
          f"avg score: {result2['avg_score']:.2f}")
    assert len(result2['states']) > 0, "Should collect experiences"
    assert not torch.isnan(result2['returns']).any(), "NaN in larger batch"
    print("PASSED: Larger batch is stable.\n")


if __name__ == "__main__":
    test_self_play()
    print("=" * 60)
    print("ALL PHASE 4 TESTS PASSED!")
    print("=" * 60)

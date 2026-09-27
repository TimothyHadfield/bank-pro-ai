"""
Phase 2 Test: Verify BankNetwork outputs are correct shapes and behave properly.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from game.environment import BankGame
from network.model import BankNetwork


def test_output_shapes():
    """Verify policy and value output shapes."""
    print("=" * 60)
    print("TEST 1: Output shapes")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size, hidden_size=128)

    # Single state
    dummy_state = torch.randn(1, state_size)
    logits, value = net(dummy_state)
    print(f"  Single state - logits shape: {logits.shape}, value shape: {value.shape}")
    assert logits.shape == (1, 2), f"Expected (1,2), got {logits.shape}"
    assert value.shape == (1, 1), f"Expected (1,1), got {value.shape}"

    # Batch of states
    batch = torch.randn(32, state_size)
    logits, value = net(batch)
    print(f"  Batch of 32 - logits shape: {logits.shape}, value shape: {value.shape}")
    assert logits.shape == (32, 2), f"Expected (32,2), got {logits.shape}"
    assert value.shape == (32, 1), f"Expected (32,1), got {value.shape}"

    print("PASSED: Output shapes are correct.\n")


def test_policy_is_probability_distribution():
    """Verify softmax of policy logits sums to 1."""
    print("=" * 60)
    print("TEST 2: Policy is valid probability distribution")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size)

    states = torch.randn(10, state_size)
    logits, _ = net(states)
    probs = torch.softmax(logits, dim=-1)

    sums = probs.sum(dim=-1)
    print(f"  Probability sums: {sums.detach().numpy()}")
    assert torch.allclose(sums, torch.ones(10), atol=1e-5), "Probabilities don't sum to 1"
    assert (probs >= 0).all(), "Negative probabilities found"

    print("PASSED: Policy outputs valid probability distributions.\n")


def test_select_action():
    """Verify action selection returns correct types and ranges."""
    print("=" * 60)
    print("TEST 3: Action selection")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size)

    game = BankGame(num_players=4, seed=42)
    game.reset()
    state_vec = game.get_state_vector(0)

    # Stochastic
    action, log_prob, value = net.select_action(state_vec, deterministic=False)
    print(f"  Stochastic: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")
    assert action in (0, 1), f"Action should be 0 or 1, got {action}"
    assert isinstance(log_prob, float), "log_prob should be float"
    assert isinstance(value, float), "value should be float"

    # Deterministic
    action_d, log_prob_d, value_d = net.select_action(state_vec, deterministic=True)
    print(f"  Deterministic: action={action_d}, log_prob={log_prob_d:.4f}, value={value_d:.4f}")
    assert action_d in (0, 1)

    print("PASSED: Action selection works correctly.\n")


def test_evaluate_batch():
    """Verify evaluate returns correct shapes."""
    print("=" * 60)
    print("TEST 4: Evaluate batch")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size)

    batch_size = 16
    states = torch.randn(batch_size, state_size)
    actions = torch.randint(0, 2, (batch_size,))

    log_probs, entropy, values = net.evaluate(states, actions)

    print(f"  log_probs shape: {log_probs.shape}")
    print(f"  entropy shape: {entropy.shape}")
    print(f"  values shape: {values.shape}")

    assert log_probs.shape == (batch_size,), f"Expected ({batch_size},), got {log_probs.shape}"
    assert entropy.shape == (batch_size,), f"Expected ({batch_size},), got {entropy.shape}"
    assert values.shape == (batch_size,), f"Expected ({batch_size},), got {values.shape}"
    assert (log_probs <= 0).all(), "Log probabilities should be <= 0"
    assert (entropy >= 0).all(), "Entropy should be non-negative"

    print("PASSED: Evaluate returns correct shapes and values.\n")


def test_different_states_different_outputs():
    """Verify different states produce different outputs."""
    print("=" * 60)
    print("TEST 5: Different states produce different outputs")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size)

    state1 = torch.randn(1, state_size)
    state2 = torch.randn(1, state_size)

    logits1, val1 = net(state1)
    logits2, val2 = net(state2)

    logits_differ = not torch.allclose(logits1, logits2, atol=1e-6)
    values_differ = not torch.allclose(val1, val2, atol=1e-6)

    print(f"  State 1 logits: {logits1.detach().numpy()}, value: {val1.item():.4f}")
    print(f"  State 2 logits: {logits2.detach().numpy()}, value: {val2.item():.4f}")
    print(f"  Logits differ: {logits_differ}, Values differ: {values_differ}")

    assert logits_differ or values_differ, "Different inputs should produce different outputs"

    print("PASSED: Different states produce different outputs.\n")


def test_parameter_count():
    """Verify the network is the right size (not too big)."""
    print("=" * 60)
    print("TEST 6: Parameter count")
    print("=" * 60)

    state_size = BankGame.state_size()
    net = BankNetwork(state_size=state_size, hidden_size=128)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Total parameters: {total_params}")
    # Expected: 11*128 + 128 + 128*128 + 128 + 128*2 + 2 + 128*1 + 1
    #         = 1408 + 128 + 16384 + 128 + 256 + 2 + 128 + 1 = 18435
    assert total_params < 50000, "Network is too large"
    print(f"PASSED: Network has reasonable parameter count.\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    test_output_shapes()
    test_policy_is_probability_distribution()
    test_select_action()
    test_evaluate_batch()
    test_different_states_different_outputs()
    test_parameter_count()

    print("=" * 60)
    print("ALL PHASE 2 TESTS PASSED!")
    print("=" * 60)

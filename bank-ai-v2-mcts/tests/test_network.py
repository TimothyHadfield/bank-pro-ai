"""
Tests for network/model.py.

Run from bank-ai-v2-mcts directory:
    python -m tests.test_network
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from game import BankGame
from network import BankNetwork


def test_forward_single_state():
    net = BankNetwork()
    x = torch.randn(config.INPUT_SIZE)
    policy, value = net(x)
    assert policy.shape == (config.NUM_ACTIONS,), f"policy shape {policy.shape}"
    assert value.shape == (1,), f"value shape {value.shape}"
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    assert abs(policy.sum().item() - 1.0) < 1e-5, f"policy doesn't sum to 1: {policy.sum().item()}"
    assert 0.0 <= value.item() <= 1.0, f"value not in [0, 1]: {value.item()}"
    print("  ok test_forward_single_state")


def test_forward_batch():
    net = BankNetwork()
    batch_size = 7
    x = torch.randn(batch_size, config.INPUT_SIZE)
    policy, value = net(x)
    assert policy.shape == (batch_size, config.NUM_ACTIONS)
    assert value.shape == (batch_size, 1)
    sums = policy.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    assert (value >= 0).all() and (value <= 1).all()
    print("  ok test_forward_batch")


def test_predict_numpy():
    net = BankNetwork()
    g = BankGame(num_players=4, seed=42)
    g.reset()
    sv = g.get_state_vector(0)
    policy, value = net.predict(sv)
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (config.NUM_ACTIONS,)
    assert isinstance(value, float)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert 0.0 <= value <= 1.0
    print("  ok test_predict_numpy")


def test_predict_batch_numpy():
    net = BankNetwork()
    states = []
    for nplayers in [2, 3, 4, 5, 6]:
        g = BankGame(num_players=nplayers, seed=nplayers * 7)
        g.reset()
        states.append(g.get_state_vector(0))
    batch = np.stack(states)
    policy, value = net.predict_batch(batch)
    assert policy.shape == (5, config.NUM_ACTIONS)
    assert value.shape == (5,)
    print("  ok test_predict_batch_numpy")


def test_different_states_different_outputs():
    """Random init should give a network that responds to inputs."""
    net = BankNetwork()
    a = torch.randn(config.INPUT_SIZE)
    b = torch.randn(config.INPUT_SIZE)
    pa, va = net(a)
    pb, vb = net(b)
    assert not torch.allclose(pa, pb), "Different inputs gave identical policies — network broken?"
    assert va.item() != vb.item(), "Different inputs gave identical values — network broken?"
    print("  ok test_different_states_different_outputs")


def test_handles_all_player_counts():
    net = BankNetwork()
    for nplayers in [2, 3, 4, 5, 6]:
        g = BankGame(num_players=nplayers, seed=nplayers)
        g.reset()
        for pid in range(nplayers):
            sv = g.get_state_vector(pid)
            policy, value = net.predict(sv)
            assert np.isfinite(policy).all() and np.isfinite(value)
    print("  ok test_handles_all_player_counts")


def test_gradients_flow():
    """Check that loss.backward() updates both heads."""
    net = BankNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    x = torch.randn(8, config.INPUT_SIZE)
    target_policy = torch.tensor([[0.7, 0.3]] * 8)
    target_value = torch.tensor([[0.5]] * 8)

    # Baseline forward
    policy, value = net(x)
    initial_policy_loss = -(target_policy * torch.log(policy + 1e-9)).sum(dim=-1).mean()
    initial_value_loss = ((value - target_value) ** 2).mean()
    initial_total = initial_policy_loss + initial_value_loss

    for _ in range(50):
        policy, value = net(x)
        policy_loss = -(target_policy * torch.log(policy + 1e-9)).sum(dim=-1).mean()
        value_loss = ((value - target_value) ** 2).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final
    policy, value = net(x)
    final_policy_loss = -(target_policy * torch.log(policy + 1e-9)).sum(dim=-1).mean()
    final_value_loss = ((value - target_value) ** 2).mean()
    final_total = final_policy_loss + final_value_loss

    assert final_total < initial_total, (
        f"Loss did not decrease: initial={initial_total.item():.4f} final={final_total.item():.4f}"
    )
    print(
        f"  ok test_gradients_flow "
        f"(loss: {initial_total.item():.3f} -> {final_total.item():.3f})"
    )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_forward_single_state,
    test_forward_batch,
    test_predict_numpy,
    test_predict_batch_numpy,
    test_different_states_different_outputs,
    test_handles_all_player_counts,
    test_gradients_flow,
]


def main():
    failures = []
    for t in ALL_TESTS:
        print(f"running {t.__name__}")
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  FAIL: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(ALL_TESTS)} FAILED")
        for n, e in failures:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()

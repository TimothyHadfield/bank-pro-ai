"""
Slice 1e end-to-end test: self-play -> buffer -> training loop -> loss decreases.

Run from bank-ai-v2-mcts:
    python -m tests.test_training
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from network import BankNetwork
from training import (
    self_play_game,
    ReplayBuffer,
    Trainer,
)


def test_replay_buffer_basic():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    # Generate a small batch of examples
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(net, num_players=3, sims_per_move=5, seed=1, force_pure_self_play=True)
    buf.extend(examples, iteration=0)
    assert len(buf) == min(10, len(examples)), (
        f"buffer length {len(buf)} should equal min(capacity, examples)"
    )
    batch = buf.sample(batch_size=4)
    assert batch.state_vectors.shape == (4, config.INPUT_SIZE)
    assert batch.mcts_policies.shape == (4, config.NUM_ACTIONS)
    assert batch.values.shape == (4,)
    print("  ok test_replay_buffer_basic")


def test_one_training_step_runs():
    torch.manual_seed(0)
    net = BankNetwork()
    examples, _ = self_play_game(net, num_players=3, sims_per_move=5, seed=2, force_pure_self_play=True)
    buf = ReplayBuffer(capacity=1000)
    buf.extend(examples, iteration=0)
    trainer = Trainer(net)
    batch = buf.sample(batch_size=min(32, len(buf)))
    result = trainer.train_step(batch)
    assert np.isfinite(result.total_loss)
    assert np.isfinite(result.policy_loss)
    assert np.isfinite(result.value_loss)
    print(
        f"  ok test_one_training_step_runs "
        f"(total={result.total_loss:.4f} pol={result.policy_loss:.4f} "
        f"val={result.value_loss:.4f} l2={result.l2_loss:.4f})"
    )


def test_loss_decreases_on_repeated_batches():
    """
    Train for multiple steps on a fixed batch — loss should decrease (overfit a tiny dataset).
    This proves the training loop wires together correctly.
    """
    torch.manual_seed(0)
    net = BankNetwork()

    # Generate a couple of games for diverse data
    all_examples = []
    for seed in range(3):
        ex, _ = self_play_game(net, num_players=3, sims_per_move=5, seed=seed, force_pure_self_play=True)
        all_examples.extend(ex)

    buf = ReplayBuffer(capacity=10000)
    buf.extend(all_examples, iteration=0)

    trainer = Trainer(net, learning_rate=1e-3)

    # Sample one fixed batch and train repeatedly
    batch = buf.sample(batch_size=min(64, len(buf)))
    initial = trainer.train_step(batch)

    for _ in range(50):
        trainer.train_step(batch)

    final = trainer.train_step(batch)

    assert final.total_loss < initial.total_loss, (
        f"loss did not decrease: initial={initial.total_loss:.4f} final={final.total_loss:.4f}"
    )
    print(
        f"  ok test_loss_decreases_on_repeated_batches "
        f"(total {initial.total_loss:.4f} -> {final.total_loss:.4f}, "
        f"policy {initial.policy_loss:.4f} -> {final.policy_loss:.4f}, "
        f"value {initial.value_loss:.4f} -> {final.value_loss:.4f})"
    )


def test_full_mini_iteration():
    """
    A miniature version of one training iteration:
      - generate a few games
      - extend buffer
      - train a few steps
      - verify it all flows
    """
    torch.manual_seed(0)
    net = BankNetwork()
    buf = ReplayBuffer(capacity=10000)
    trainer = Trainer(net)

    n_games = 3
    n_train_steps = 10

    t0 = time.perf_counter()
    for g in range(n_games):
        examples, _ = self_play_game(
            net,
            num_players=None,    # random per game
            sims_per_move=20,
            seed=g + 100,
            force_pure_self_play=True,
        )
        buf.extend(examples, iteration=0)
    t1 = time.perf_counter()

    losses = []
    for _ in range(n_train_steps):
        batch = buf.sample(batch_size=min(64, len(buf)))
        result = trainer.train_step(batch)
        losses.append(result.total_loss)
    t2 = time.perf_counter()

    print(
        f"  ok test_full_mini_iteration "
        f"({n_games} games in {t1-t0:.1f}s, {len(buf)} buffered, "
        f"{n_train_steps} train steps in {t2-t1:.2f}s, "
        f"loss {losses[0]:.3f} -> {losses[-1]:.3f})"
    )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_replay_buffer_basic,
    test_one_training_step_runs,
    test_loss_decreases_on_repeated_batches,
    test_full_mini_iteration,
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

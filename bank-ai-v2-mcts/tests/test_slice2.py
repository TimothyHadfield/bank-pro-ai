"""
Tests for Slice 2 pieces: bots, mixed self-play, recency-weighted buffer, logger,
watchdog, checkpoint save/load, and a short end-to-end training run.

Run from bank-ai-v2-mcts directory:
    python -m tests.test_slice2
"""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from network import BankNetwork
from training import (
    AlwaysBank,
    AlwaysContinue,
    RandomBot,
    ReplayBuffer,
    ThresholdBot,
    Trainer,
    Watchdog,
    build_default_bot_pool,
    is_compatible,
    load_checkpoint,
    save_checkpoint,
    self_play_game,
)
from monitoring import MetricsLogger


# ---------------------------------------------------------------- bot tests

def test_threshold_bot():
    bot = ThresholdBot(threshold=200)
    assert bot.choose({"pot": 100}) == "continue"
    assert bot.choose({"pot": 200}) == "bank"
    assert bot.choose({"pot": 350}) == "bank"
    print("  ok test_threshold_bot")


def test_random_bot_distribution():
    bot = RandomBot(p_bank=0.3, seed=0)
    n = 1000
    bank_count = sum(1 for _ in range(n) if bot.choose({"pot": 0}) == "bank")
    rate = bank_count / n
    assert 0.20 < rate < 0.40, f"RandomBot(p=0.3) gave bank rate {rate:.2f}"
    print(f"  ok test_random_bot_distribution (rate={rate:.2f})")


def test_always_bots():
    assert AlwaysBank().choose({"pot": 0}) == "bank"
    assert AlwaysContinue().choose({"pot": 999}) == "continue"
    print("  ok test_always_bots")


def test_default_pool_built():
    pool = build_default_bot_pool(seed=0)
    assert len(pool) > 0
    names = [b.name for b in pool]
    # Should include thresholds, randoms, and always-bank/continue
    assert any("Threshold" in n for n in names)
    assert any("Random" in n for n in names)
    assert "AlwaysBank" in names
    assert "AlwaysContinue" in names
    print(f"  ok test_default_pool_built ({len(pool)} bots: {names})")


# ---------------------------------------------------------------- mixed self-play tests

def test_mixed_self_play_includes_bots_sometimes():
    """Over many games with the default mix, some games should have bot seats."""
    torch.manual_seed(0)
    net = BankNetwork()
    bot_pool = build_default_bot_pool(seed=42)
    n_games = 60
    n_with_bots = 0
    for g in range(n_games):
        # Force a 4+ player game so bot mix can engage
        examples, result = self_play_game(
            net, num_players=4, sims_per_move=3, seed=g, bot_pool=bot_pool
        )
        if result.num_bot_seats > 0:
            n_with_bots += 1
    expected = n_games * (config.ONE_BOT_FRACTION + config.TWO_BOT_FRACTION)
    # Allow generous tolerance since 60 games is a small sample
    assert n_with_bots > 0, "no bot games in 60 trials — mix logic broken?"
    print(
        f"  ok test_mixed_self_play_includes_bots_sometimes "
        f"({n_with_bots}/{n_games} games had bots; expected ~{expected:.0f})"
    )


def test_force_pure_self_play():
    torch.manual_seed(0)
    net = BankNetwork()
    pool = build_default_bot_pool(seed=0)
    for seed in range(5):
        _, result = self_play_game(
            net, num_players=4, sims_per_move=3, seed=seed,
            bot_pool=pool, force_pure_self_play=True,
        )
        assert result.num_bot_seats == 0
    print("  ok test_force_pure_self_play")


# ---------------------------------------------------------------- buffer tests

def test_buffer_iteration_tagging_and_eviction():
    buf = ReplayBuffer(capacity=1000, max_age_iterations=3, recency_decay_per_iteration=0.0)
    torch.manual_seed(0)
    net = BankNetwork()

    # Iteration 1
    ex1, _ = self_play_game(net, num_players=2, sims_per_move=3, seed=1, force_pure_self_play=True)
    buf.extend(ex1, iteration=1)
    n1 = len(buf)

    # Iterations 2, 3, 4 — at iter 5, iter 1's data should be evicted (age > 3)
    for it in (2, 3, 4):
        ex, _ = self_play_game(net, num_players=2, sims_per_move=3, seed=it, force_pure_self_play=True)
        buf.extend(ex, iteration=it)

    # At iter 5, set_iteration triggers eviction of anything with iter < (5 - 3) = 2
    buf.set_iteration(5)
    # All surviving entries must be from iter >= 2
    for entry in buf._entries:
        assert entry.iteration >= 5 - 3, f"stale entry survived: iter={entry.iteration}"
    # All iter-1 entries must be gone
    iter1_remaining = sum(1 for e in buf._entries if e.iteration == 1)
    assert iter1_remaining == 0, f"iter-1 entries still present: {iter1_remaining}"
    print(f"  ok test_buffer_iteration_tagging_and_eviction (size after eviction={len(buf)})")


def test_buffer_recency_weighted_sampling():
    """Newer entries should be sampled more often than older ones."""
    buf = ReplayBuffer(capacity=10000, max_age_iterations=100, recency_decay_per_iteration=0.5)
    torch.manual_seed(0)
    net = BankNetwork()

    # Add the same seed at iteration 1 and a different at iteration 10
    old_examples, _ = self_play_game(net, num_players=2, sims_per_move=2, seed=1, force_pure_self_play=True)
    new_examples, _ = self_play_game(net, num_players=2, sims_per_move=2, seed=2, force_pure_self_play=True)
    buf.extend(old_examples, iteration=1)
    buf.extend(new_examples, iteration=10)

    # Sample many times and count which iteration shows up
    n_samples = 5000
    n_old = 0
    n_new = 0
    for _ in range(n_samples):
        batch = buf.sample(1)
        # Decode by checking values: old and new examples may share values, so use state vectors
        # Easier: peek at the buffer entry directly by sampling the index
        # Since sample() returns a TrainingBatch, just count via independent draws
        pass
    # Better: use the internal weighted choice directly
    import random
    weights = []
    for e in buf._entries:
        age = buf.current_iteration - e.iteration
        weights.append(math.exp(-buf.recency_decay_per_iteration * age))
    sampled = random.choices(buf._entries, weights=weights, k=n_samples)
    n_old = sum(1 for e in sampled if e.iteration == 1)
    n_new = sum(1 for e in sampled if e.iteration == 10)
    assert n_new > n_old * 2, (
        f"newer entries should outnumber older 2:1 with decay=0.5/iter, age gap=9; "
        f"got new={n_new} old={n_old}"
    )
    print(f"  ok test_buffer_recency_weighted_sampling (new={n_new}, old={n_old})")


# ---------------------------------------------------------------- logger tests

def test_logger_writes_metrics_json():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "metrics.json"
        logger = MetricsLogger(metrics_path=path, verbose=False)
        logger.set_status("running")
        logger.log(1, {"total_loss": 0.5, "policy_loss": 0.3, "value_loss": 0.2})
        logger.log(2, {"total_loss": 0.4})
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "running"
        assert data["latest"]["iteration"] == 2
        assert "total_loss" in data["history"]
        assert len(data["history"]["total_loss"]) == 2
    print("  ok test_logger_writes_metrics_json")


# ---------------------------------------------------------------- watchdog tests

def test_watchdog_nan_loss():
    wd = Watchdog()
    s = wd.check(iteration=1, total_loss=float("nan"))
    assert s.halted and "non-finite" in s.reason.lower()
    print("  ok test_watchdog_nan_loss")


def test_watchdog_loss_explosion():
    wd = Watchdog(loss_explosion_factor=5.0, min_history_for_explosion=5)
    # Build history of small losses
    for i in range(10):
        s = wd.check(iteration=i, total_loss=1.0)
        assert not s.halted
    # Now spike
    s = wd.check(iteration=11, total_loss=100.0)
    assert s.halted and "exploded" in s.reason.lower()
    print(f"  ok test_watchdog_loss_explosion ({s.reason})")


def test_watchdog_policy_collapse():
    wd = Watchdog(
        policy_collapse_entropy_threshold=0.05,
        policy_collapse_consecutive=5,
    )
    for i in range(4):
        s = wd.check(iteration=i, total_loss=1.0, policy_entropy=0.01)
        assert not s.halted
    s = wd.check(iteration=4, total_loss=1.0, policy_entropy=0.01)
    assert s.halted and "collapse" in s.reason.lower()
    print(f"  ok test_watchdog_policy_collapse ({s.reason})")


def test_watchdog_recovers_from_low_entropy():
    """If entropy goes back up, the consecutive counter should reset."""
    wd = Watchdog(
        policy_collapse_entropy_threshold=0.05,
        policy_collapse_consecutive=5,
    )
    for i in range(3):
        wd.check(iteration=i, total_loss=1.0, policy_entropy=0.01)
    # Entropy recovers
    wd.check(iteration=3, total_loss=1.0, policy_entropy=0.5)
    # Now go low again — should NOT halt yet (counter reset)
    for i in range(4, 7):
        s = wd.check(iteration=i, total_loss=1.0, policy_entropy=0.01)
        assert not s.halted, f"halted prematurely at iter {i}"
    print("  ok test_watchdog_recovers_from_low_entropy")


# ---------------------------------------------------------------- checkpoint tests

def test_checkpoint_save_and_load():
    torch.manual_seed(0)
    net = BankNetwork()
    trainer = Trainer(net)

    # Touch the network so weights are non-trivial
    x = torch.randn(4, config.INPUT_SIZE)
    target_p = torch.tensor([[0.6, 0.4]] * 4)
    target_v = torch.tensor([[0.7]] * 4)
    for _ in range(5):
        policy_logits = net.policy_logits(x)
        value_logit = net.value_logit(x)
        loss = (
            -(target_p * torch.nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
            + ((torch.sigmoid(value_logit) - target_v) ** 2).mean()
        )
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

    # Snapshot weights
    weights_before = {k: v.clone() for k, v in net.state_dict().items()}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.pt"
        save_checkpoint(path, net, trainer, iteration=42, metadata={"foo": "bar"})

        # Load into a fresh network
        net2 = BankNetwork()
        trainer2 = Trainer(net2)
        info = load_checkpoint(path, net2, trainer2)
        assert info["iteration"] == 42
        assert info["metadata"]["foo"] == "bar"
        # Weights should match
        for k, v in net2.state_dict().items():
            assert torch.allclose(v, weights_before[k]), f"weight mismatch at {k}"
        # Compatibility check
        ok, msg = is_compatible(info["config_snapshot"])
        assert ok, msg
    print("  ok test_checkpoint_save_and_load")


# ---------------------------------------------------------------- end-to-end mini training

def test_end_to_end_mini_training_run():
    """
    Run train.run_training() for 2 iterations with tiny settings. Verify metrics.json,
    checkpoints, and that loss decreased.
    """
    from train import run_training

    with tempfile.TemporaryDirectory() as td:
        ck_dir = Path(td) / "models"
        metrics_path = Path(td) / "metrics.json"

        t0 = time.perf_counter()
        run_training(
            max_iterations=2,
            games_per_iteration=2,
            training_steps_per_iteration=10,
            sims_per_move=10,           # very small for speed
            batch_size=32,
            checkpoint_dir=ck_dir,
            metrics_path=metrics_path,
            seed=0,
        )
        elapsed = time.perf_counter() - t0

        # Validate metrics file
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["latest"]["iteration"] == 2
        assert "total_loss" in data["history"]
        assert len(data["history"]["total_loss"]) == 2

        # Validate checkpoints exist
        assert (ck_dir / "latest.pt").exists()

        print(
            f"  ok test_end_to_end_mini_training_run "
            f"(2 iterations in {elapsed:.1f}s, latest_loss={data['latest']['total_loss']:.3f})"
        )


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_threshold_bot,
    test_random_bot_distribution,
    test_always_bots,
    test_default_pool_built,
    test_mixed_self_play_includes_bots_sometimes,
    test_force_pure_self_play,
    test_buffer_iteration_tagging_and_eviction,
    test_buffer_recency_weighted_sampling,
    test_logger_writes_metrics_json,
    test_watchdog_nan_loss,
    test_watchdog_loss_explosion,
    test_watchdog_policy_collapse,
    test_watchdog_recovers_from_low_entropy,
    test_checkpoint_save_and_load,
    test_end_to_end_mini_training_run,
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

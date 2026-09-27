"""
Replay buffer for Bank AI v2.

Slice 2: per-iteration tagging, age-based eviction, recency-weighted sampling.

Each example is stored with the iteration number it was generated in. On every add(),
entries older than `max_age_iterations` (relative to current iteration) are dropped.
Sampling weights newer entries higher: weight ∝ exp(-age * decay_rate).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

import config
from training.self_play import TrainingExample


@dataclass
class TrainingBatch:
    """A minibatch ready for the trainer."""
    state_vectors: np.ndarray   # (B, INPUT_SIZE)
    mcts_policies: np.ndarray   # (B, NUM_ACTIONS)
    values: np.ndarray          # (B,)


@dataclass
class _Entry:
    example: TrainingExample
    iteration: int


class ReplayBuffer:
    """
    Bounded replay buffer with iteration tagging and age-based eviction.

    Behavior:
      - Stores up to `capacity` entries (drops oldest first when full).
      - On every extend(), entries older than `max_age_iterations` (relative to the
        last-seen iteration) are dropped before inserting new ones.
      - sample() weights newer entries more heavily (exp decay by age).
    """

    def __init__(
        self,
        capacity: int = config.REPLAY_BUFFER_SIZE,
        max_age_iterations: int = config.MAX_BUFFER_AGE_ITERATIONS,
        recency_decay_per_iteration: float = 0.05,
    ):
        self.capacity = capacity
        self.max_age_iterations = max_age_iterations
        self.recency_decay_per_iteration = recency_decay_per_iteration
        self._entries: list[_Entry] = []
        self._current_iteration: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    def set_iteration(self, iteration: int) -> None:
        """Update the current iteration counter (used for age-based eviction)."""
        self._current_iteration = max(self._current_iteration, iteration)
        self._evict_old()

    def add(self, example: TrainingExample, iteration: int) -> None:
        self._current_iteration = max(self._current_iteration, iteration)
        self._entries.append(_Entry(example, iteration))
        self._evict_old()
        self._enforce_capacity()

    def extend(self, examples: list[TrainingExample], iteration: int) -> None:
        self._current_iteration = max(self._current_iteration, iteration)
        for ex in examples:
            self._entries.append(_Entry(ex, iteration))
        self._evict_old()
        self._enforce_capacity()

    def _evict_old(self) -> None:
        cutoff = self._current_iteration - self.max_age_iterations
        if cutoff <= 0:
            return
        self._entries = [e for e in self._entries if e.iteration >= cutoff]

    def _enforce_capacity(self) -> None:
        if len(self._entries) > self.capacity:
            # drop oldest first
            self._entries = self._entries[-self.capacity :]

    def sample(
        self, batch_size: int, rng: random.Random | None = None
    ) -> TrainingBatch:
        if not self._entries:
            raise ValueError("Cannot sample from empty buffer")
        rng = rng or random

        # Compute recency weights
        weights = []
        for e in self._entries:
            age = self._current_iteration - e.iteration
            w = math.exp(-self.recency_decay_per_iteration * age)
            weights.append(w)

        sampled = rng.choices(self._entries, weights=weights, k=batch_size)
        return TrainingBatch(
            state_vectors=np.stack([s.example.state_vector for s in sampled]),
            mcts_policies=np.stack([s.example.mcts_policy for s in sampled]),
            values=np.array([s.example.value for s in sampled], dtype=np.float32),
        )

    # --- diagnostics for the dashboard ---

    def stats(self) -> dict:
        if not self._entries:
            return {
                "size": 0,
                "current_iteration": self._current_iteration,
                "oldest_age": None,
                "newest_age": None,
                "average_age": None,
            }
        ages = [self._current_iteration - e.iteration for e in self._entries]
        return {
            "size": len(self._entries),
            "current_iteration": self._current_iteration,
            "oldest_age": max(ages),
            "newest_age": min(ages),
            "average_age": sum(ages) / len(ages),
        }

"""
Bot strategies and the opponent pool used to vary self-play opposition.

Each bot exposes a `choose(state: dict) -> "bank" | "continue"` method that the self-play
loop calls. Bots see the SAME public state dict that an AI player sees — they get no
special access to other players' same-turn decisions (no peeking).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from game.environment import BANK, CONTINUE


class Bot(ABC):
    """Base class for non-network strategies."""

    name: str = "Bot"

    @abstractmethod
    def choose(self, state: dict) -> str:
        ...


class ThresholdBot(Bot):
    """Banks when the pot meets or exceeds `threshold`. Otherwise continues."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.name = f"Threshold({threshold})"

    def choose(self, state: dict) -> str:
        return BANK if state["pot"] >= self.threshold else CONTINUE


class RandomBot(Bot):
    """Banks with probability `p_bank` each decision."""

    def __init__(self, p_bank: float, seed: int | None = None):
        self.p_bank = p_bank
        self.name = f"Random(p={p_bank:.2f})"
        self._rng = random.Random(seed)

    def choose(self, state: dict) -> str:
        return BANK if self._rng.random() < self.p_bank else CONTINUE


class AlwaysContinue(Bot):
    """Never banks — extreme adversarial training case."""

    name = "AlwaysContinue"

    def choose(self, state: dict) -> str:
        return CONTINUE


class AlwaysBank(Bot):
    """Banks at the very first decision point each round."""

    name = "AlwaysBank"

    def choose(self, state: dict) -> str:
        return BANK


# ---------------------------------------------------------------- pool

def build_default_bot_pool(seed: int | None = None) -> list[Bot]:
    """
    Construct the default mix of bots used in self-play and evaluation.
    Per PLAN.md §11.
    """
    import config
    rng = random.Random(seed)
    bots: list[Bot] = []
    for t in config.THRESHOLD_BOT_THRESHOLDS:
        bots.append(ThresholdBot(t))
    for p in config.RANDOM_BOT_PROBABILITIES:
        bots.append(RandomBot(p, seed=rng.randint(0, 2**31 - 1)))
    bots.append(AlwaysBank())
    bots.append(AlwaysContinue())
    return bots

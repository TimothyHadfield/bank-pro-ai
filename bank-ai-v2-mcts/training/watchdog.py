"""
Watchdog for Bank AI v2 training.

Monitors training metrics each iteration and signals halt if something looks broken.

Halt conditions (per PLAN/v1 patterns):
  1. NaN or Inf in any reported loss -> halt
  2. Total loss exceeds 10x running average for the past N iterations -> halt
  3. Policy collapse: MCTS visit-distribution entropy near 0 for many iterations -> halt
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class WatchdogStatus:
    halted: bool
    reason: str = ""
    triggered_metric: str = ""
    triggered_value: float = 0.0


class Watchdog:
    def __init__(
        self,
        history_window: int = 50,
        loss_explosion_factor: float = 10.0,
        min_history_for_explosion: int = 20,
        policy_collapse_entropy_threshold: float = 0.05,
        policy_collapse_consecutive: int = 20,
    ):
        self.history_window = history_window
        self.loss_explosion_factor = loss_explosion_factor
        self.min_history_for_explosion = min_history_for_explosion
        self.policy_collapse_entropy_threshold = policy_collapse_entropy_threshold
        self.policy_collapse_consecutive = policy_collapse_consecutive

        self._loss_history: deque[float] = deque(maxlen=history_window)
        self._consecutive_low_entropy = 0

    def check(
        self,
        iteration: int,
        total_loss: float,
        policy_entropy: float | None = None,
    ) -> WatchdogStatus:
        # 1. NaN / Inf check
        if not math.isfinite(total_loss):
            return WatchdogStatus(
                halted=True,
                reason="non-finite loss (NaN or Inf)",
                triggered_metric="total_loss",
                triggered_value=total_loss,
            )

        # 2. Loss explosion: only check after we have enough history
        if len(self._loss_history) >= self.min_history_for_explosion:
            avg = sum(self._loss_history) / len(self._loss_history)
            if avg > 0 and total_loss > self.loss_explosion_factor * avg:
                return WatchdogStatus(
                    halted=True,
                    reason=f"loss exploded: {total_loss:.4f} > {self.loss_explosion_factor}× running avg ({avg:.4f})",
                    triggered_metric="total_loss",
                    triggered_value=total_loss,
                )
        self._loss_history.append(total_loss)

        # 3. Policy collapse: low entropy for many consecutive iterations
        if policy_entropy is not None:
            if not math.isfinite(policy_entropy):
                return WatchdogStatus(
                    halted=True,
                    reason="non-finite policy entropy",
                    triggered_metric="policy_entropy",
                    triggered_value=policy_entropy,
                )
            if policy_entropy < self.policy_collapse_entropy_threshold:
                self._consecutive_low_entropy += 1
                if self._consecutive_low_entropy >= self.policy_collapse_consecutive:
                    return WatchdogStatus(
                        halted=True,
                        reason=(
                            f"policy collapse: entropy < {self.policy_collapse_entropy_threshold} "
                            f"for {self._consecutive_low_entropy} iterations"
                        ),
                        triggered_metric="policy_entropy",
                        triggered_value=policy_entropy,
                    )
            else:
                self._consecutive_low_entropy = 0

        return WatchdogStatus(halted=False)

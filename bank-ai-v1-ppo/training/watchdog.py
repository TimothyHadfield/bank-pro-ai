"""
Training watchdog: monitors metrics and auto-halts on anomalies.
"""
import os
import torch
import numpy as np
from collections import deque
from config import Config


class Watchdog:
    """
    Monitors training health and triggers a halt if something goes wrong.

    Halt conditions:
    1. NaN in network weights or loss values
    2. Policy loss exceeds 10x its running average
    3. Average score drops below 50% of peak for 20 consecutive batches
    4. Policy collapse: same action > 95% of decisions for 10 consecutive batches
    """

    def __init__(self, config: Config):
        self.config = config
        self.policy_loss_history = deque(maxlen=100)
        self.peak_score = 0.0
        self.below_peak_count = 0
        self.collapse_count = 0
        self.halted = False
        self.halt_reason = ""

    def check(self, network, metrics: dict, action_distribution: dict) -> bool:
        """
        Check all halt conditions.

        Args:
            network: the BankNetwork model
            metrics: dict with keys like policy_loss, value_loss, avg_score, entropy
            action_distribution: dict with action counts {0: bank_count, 1: continue_count}

        Returns:
            True if training should halt, False otherwise.
        """
        if self.halted:
            return True

        # Condition 1: NaN in weights
        for name, param in network.named_parameters():
            if torch.isnan(param).any():
                self._halt(f"NaN detected in network parameter '{name}'")
                return True

        # Condition 1b: NaN in losses
        for key in ("policy_loss", "value_loss", "entropy"):
            val = metrics.get(key)
            if val is not None and (np.isnan(val) or np.isinf(val)):
                self._halt(f"NaN/Inf detected in {key}: {val}")
                return True

        # Condition 2: Policy loss spike
        policy_loss = metrics.get("policy_loss", 0.0)
        self.policy_loss_history.append(abs(policy_loss))
        if len(self.policy_loss_history) > 10:
            running_avg = np.mean(list(self.policy_loss_history)[:-1])
            if running_avg > 0 and abs(policy_loss) > self.config.watchdog_policy_loss_multiplier * running_avg:
                self._halt(
                    f"Policy loss spike: {abs(policy_loss):.6f} > "
                    f"{self.config.watchdog_policy_loss_multiplier}x running avg {running_avg:.6f}"
                )
                return True

        # Condition 3: Score drop
        avg_score = metrics.get("avg_score", 0.0)
        if avg_score > self.peak_score:
            self.peak_score = avg_score
            self.below_peak_count = 0
        elif self.peak_score > 0 and avg_score < self.config.watchdog_score_drop_ratio * self.peak_score:
            self.below_peak_count += 1
            if self.below_peak_count >= self.config.watchdog_score_drop_patience:
                self._halt(
                    f"Score collapse: avg_score={avg_score:.1f} < "
                    f"{self.config.watchdog_score_drop_ratio*100:.0f}% of peak {self.peak_score:.1f} "
                    f"for {self.below_peak_count} consecutive batches"
                )
                return True
        else:
            self.below_peak_count = 0

        # Condition 4: Policy collapse
        total_actions = sum(action_distribution.values())
        if total_actions > 0:
            max_action_frac = max(action_distribution.values()) / total_actions
            if max_action_frac > self.config.watchdog_collapse_threshold:
                self.collapse_count += 1
                if self.collapse_count >= self.config.watchdog_collapse_patience:
                    self._halt(
                        f"Policy collapse: {max_action_frac:.1%} of actions are the same "
                        f"for {self.collapse_count} consecutive batches"
                    )
                    return True
            else:
                self.collapse_count = 0

        return False

    def _halt(self, reason: str):
        """Record halt reason and set halted flag."""
        self.halted = True
        self.halt_reason = reason
        print(f"\n{'='*60}")
        print(f"WATCHDOG HALT: {reason}")
        print(f"{'='*60}\n")

    def get_dashboard_status(self) -> dict:
        """Return current watchdog status for the dashboard."""
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "peak_score": self.peak_score,
            "below_peak_count": self.below_peak_count,
            "collapse_count": self.collapse_count,
        }

    def get_state(self) -> dict:
        """Return watchdog state for checkpointing."""
        return {
            "policy_loss_history": list(self.policy_loss_history),
            "peak_score": self.peak_score,
            "below_peak_count": self.below_peak_count,
            "collapse_count": self.collapse_count,
        }

    def load_state(self, state: dict):
        """Restore watchdog state from checkpoint."""
        self.policy_loss_history = deque(state.get("policy_loss_history", []), maxlen=100)
        self.peak_score = state.get("peak_score", 0.0)
        self.below_peak_count = state.get("below_peak_count", 0)
        self.collapse_count = state.get("collapse_count", 0)

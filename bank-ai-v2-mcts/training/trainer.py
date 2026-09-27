"""
Network training step for Bank AI v2.

Loss = policy_loss (cross-entropy with MCTS visit dist) + value_loss (MSE) + L2 regularization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import config
from network import BankNetwork
from training.replay_buffer import TrainingBatch


@dataclass
class TrainStepResult:
    total_loss: float
    policy_loss: float
    value_loss: float
    l2_loss: float


class Trainer:
    def __init__(
        self,
        network: BankNetwork,
        learning_rate: float = config.LEARNING_RATE,
        l2_weight: float = config.L2_WEIGHT,
    ):
        self.network = network
        self.l2_weight = l2_weight
        # Adam includes L2 via weight_decay, but we compute it separately for monitoring
        # so we use weight_decay=0 here and add L2 manually to keep it visible.
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

    def set_learning_rate(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train_step(self, batch: TrainingBatch) -> TrainStepResult:
        self.network.train()

        x = torch.from_numpy(batch.state_vectors)             # (B, INPUT_SIZE)
        target_policy = torch.from_numpy(batch.mcts_policies)  # (B, NUM_ACTIONS)
        target_value = torch.from_numpy(batch.values).unsqueeze(-1)  # (B, 1)

        # Forward
        policy_logits = self.network.policy_logits(x)         # (B, NUM_ACTIONS)
        value_logit = self.network.value_logit(x)             # (B, 1)

        # Policy loss: cross-entropy between target distribution and predicted distribution
        # log_softmax + sum(target * log_p) is more numerically stable than softmax + log
        log_pred = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policy * log_pred).sum(dim=-1).mean()

        # Value loss: BCE between sigmoid(value_logit) and target value (already in [0,1])
        # Using BCE-with-logits avoids redundant sigmoid in the loss path.
        value_loss = F.binary_cross_entropy_with_logits(value_logit, target_value)

        # L2 regularization (manual, for monitoring)
        l2 = sum((p ** 2).sum() for p in self.network.parameters())
        l2_loss = self.l2_weight * l2

        total = policy_loss + value_loss + l2_loss

        self.optimizer.zero_grad()
        total.backward()
        # Optional: gradient clipping (not in PLAN but useful)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return TrainStepResult(
            total_loss=float(total.item()),
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            l2_loss=float(l2_loss.item()),
        )

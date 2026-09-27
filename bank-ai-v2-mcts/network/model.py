"""
Two-headed neural network for Bank AI v2.

Architecture (per PLAN.md §3.1):
    Input (28 floats)
      -> Linear(28 -> 128) -> ReLU
      -> Linear(128 -> 128) -> ReLU       (shared trunk)
      |
      +-- Policy head:
      |     -> Linear(128 -> 64) -> ReLU
      |     -> Linear(64 -> 2)
      |     -> Softmax (probability for [bank, continue])
      |
      +-- Value head:
            -> Linear(128 -> 64) -> ReLU
            -> Linear(64 -> 1)
            -> Sigmoid (probability that THIS player wins from here)

The same network handles all player counts (2-6). Player count is encoded in the input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BankNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = config.INPUT_SIZE,
        hidden_size: int = config.HIDDEN_SIZE,
        head_size: int = config.HEAD_SIZE,
        num_actions: int = config.NUM_ACTIONS,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head -> logits over actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, num_actions),
        )

        # Value head -> single scalar in [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (batch, input_size) OR (input_size,)

        Returns:
            policy_probs: shape (batch, num_actions) summing to 1 along last dim
            value: shape (batch, 1), each in [0, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        trunk_out = self.trunk(x)
        policy_logits = self.policy_head(trunk_out)
        policy_probs = F.softmax(policy_logits, dim=-1)
        value_logits = self.value_head(trunk_out)
        value = torch.sigmoid(value_logits)

        if squeeze_back:
            return policy_probs.squeeze(0), value.squeeze(0)
        return policy_probs, value

    # ---------------------------------------------------------------- inference helpers

    @torch.no_grad()
    def predict(self, state_vector: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Single-state inference. Returns (policy_probs as numpy [num_actions], value as float).
        """
        self.eval()
        x = torch.from_numpy(state_vector.astype(np.float32))
        policy, value = self.forward(x)
        return policy.cpu().numpy(), float(value.item())

    @torch.no_grad()
    def predict_batch(self, state_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Batched inference. state_vectors shape (N, input_size).
        Returns (policy_probs (N, num_actions), values (N,)).
        """
        self.eval()
        x = torch.from_numpy(state_vectors.astype(np.float32))
        policy, value = self.forward(x)
        return policy.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    # ---------------------------------------------------------------- training-time helpers

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw policy logits (before softmax). Used in cross-entropy loss for stability."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        trunk_out = self.trunk(x)
        return self.policy_head(trunk_out)

    def value_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw value logit (pre-sigmoid)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        trunk_out = self.trunk(x)
        return self.value_head(trunk_out)

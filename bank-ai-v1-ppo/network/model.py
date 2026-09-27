import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Policy network: takes a state vector and outputs action logits (bank, continue).
    Own hidden layers — shares zero parameters with the value network.
    """

    def __init__(self, state_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    Value network: takes a state vector and outputs a scalar value estimate.
    Own hidden layers — shares zero parameters with the policy network.
    """

    def __init__(self, state_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BankNetwork:
    """
    Combined policy + value interface for the Bank AI.
    Wraps two fully separate networks with zero shared parameters.
    Not an nn.Module itself — just a convenience wrapper.
    """

    def __init__(self, state_size: int, hidden_size: int = 128):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.policy = PolicyNetwork(state_size, hidden_size)
        self.value = ValueNetwork(state_size, hidden_size)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select an action given a single state.

        Returns:
            action: 0 (bank) or 1 (continue)
            log_prob: log probability of the selected action
            value: value estimate for this state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        val = self.value(state_tensor)
        return action.item(), log_prob.item(), val.item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate a batch of state-action pairs.

        Returns:
            log_probs: shape (batch,)
            entropy: shape (batch,)
            values: shape (batch,)
        """
        logits = self.policy(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(states).squeeze(-1)

        return log_probs, entropy, values

    def eval(self):
        """Set both networks to eval mode."""
        self.policy.eval()
        self.value.eval()
        return self

    def train(self):
        """Set both networks to train mode."""
        self.policy.train()
        self.value.train()
        return self

    def state_dict(self):
        """Return combined state dict for checkpointing."""
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        self.policy.load_state_dict(state_dict["policy"])
        self.value.load_state_dict(state_dict["value"])

    def parameters(self):
        """Yield all parameters from both networks (for watchdog NaN check)."""
        yield from self.policy.parameters()
        yield from self.value.parameters()

    def named_parameters(self):
        """Yield all named parameters from both networks."""
        for name, p in self.policy.named_parameters():
            yield f"policy.{name}", p
        for name, p in self.value.named_parameters():
            yield f"value.{name}", p

    @staticmethod
    def state_size_for(num_opponent_slots: int = 5) -> int:
        return 6 + num_opponent_slots

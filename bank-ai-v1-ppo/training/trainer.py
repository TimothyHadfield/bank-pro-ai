"""
PPO trainer: implements the Proximal Policy Optimization update step.
Uses separate optimizers for the policy and value networks.
Supports entropy annealing to maintain exploration.
"""
import torch
import torch.nn as nn
from config import Config


class PPOTrainer:
    """Performs PPO updates on the separate policy and value networks."""

    def __init__(self, network, config: Config):
        self.network = network
        self.config = config
        self.policy_optimizer = torch.optim.Adam(
            network.policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            network.value.parameters(), lr=config.learning_rate
        )
        self.batch_count = 0

    def _get_entropy_coeff(self):
        """Anneal entropy coefficient: start high (0.05), decay to config value over 2000 batches."""
        start = 0.05
        end = self.config.entropy_coefficient
        decay_batches = 2000
        if self.batch_count >= decay_batches:
            return end
        progress = self.batch_count / decay_batches
        return start + (end - start) * progress

    def update(self, states, actions, old_log_probs, returns, advantages):
        self.network.train()
        self.batch_count += 1

        entropy_coeff = self._get_entropy_coeff()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.config.ppo_epochs):
            new_log_probs, entropy, values = self.network.evaluate(states, actions)

            # --- Policy update ---
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_mean = entropy.mean()

            policy_total = policy_loss - entropy_coeff * entropy_mean

            self.policy_optimizer.zero_grad()
            policy_total.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(
                self.network.policy.parameters(), self.config.max_grad_norm
            )
            self.policy_optimizer.step()

            # --- Value update ---
            _, _, values = self.network.evaluate(states, actions)
            value_loss = nn.functional.mse_loss(values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.value.parameters(), self.config.max_grad_norm
            )
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()

        num_epochs = self.config.ppo_epochs
        return {
            "policy_loss": total_policy_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "entropy": total_entropy / num_epochs,
        }

    def get_optimizer_state(self):
        return {
            "policy": self.policy_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict(),
            "batch_count": self.batch_count,
        }

    def load_optimizer_state(self, state_dict):
        self.policy_optimizer.load_state_dict(state_dict["policy"])
        self.value_optimizer.load_state_dict(state_dict["value"])
        self.batch_count = state_dict.get("batch_count", 0)

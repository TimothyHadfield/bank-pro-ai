"""
Central configuration for all hyperparameters.
Every other file imports from here — never hardcode values elsewhere.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # Game settings
    num_players: int = 4
    num_rounds: int = 10

    # Network architecture
    hidden_size: int = 128

    # Optimizer
    learning_rate: float = 3e-4

    # PPO hyperparameters
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    ppo_clip: float = 0.2         # Clipping parameter
    ppo_epochs: int = 4           # Epochs per PPO update
    batch_size: int = 256         # Number of games per training batch
    max_training_batches: int = 100000

    # Checkpointing and evaluation
    checkpoint_interval: int = 100   # Save every N batches
    eval_interval: int = 50          # Evaluate every N batches

    # Loss coefficients
    entropy_coefficient: float = 0.03
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # Reward: pure rank-based (no score component)
    # 1st: +1.0, 2nd: +0.33, 3rd: -0.33, 4th: -1.0

    # Watchdog thresholds (watchdog is OFF by default — enable via dashboard)
    watchdog_enabled: bool = False
    watchdog_policy_loss_multiplier: float = 20.0
    watchdog_score_drop_ratio: float = 0.3
    watchdog_score_drop_patience: int = 50
    watchdog_collapse_threshold: float = 0.98
    watchdog_collapse_patience: int = 25

    # Reproducibility
    seed: int = 42

    # Directories
    model_dir: str = "models"
    log_dir: str = "runs"
    dashboard_metrics_path: str = "dashboard/metrics.json"

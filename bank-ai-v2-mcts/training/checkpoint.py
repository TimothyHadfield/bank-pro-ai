"""
Checkpoint save/load for Bank AI v2.

A checkpoint bundles:
  - network state_dict
  - optimizer state_dict
  - current iteration number
  - any extra metadata (last metrics, replay buffer summary, etc.)

Atomic writes (write to .tmp, then os.replace) prevent corruption from interrupted saves.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import config
from network import BankNetwork
from training.trainer import Trainer


def save_checkpoint(
    path: str | Path,
    network: BankNetwork,
    trainer: Trainer | None,
    iteration: int,
    metadata: dict | None = None,
) -> None:
    """Atomically save a checkpoint to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict() if trainer else None,
        "iteration": iteration,
        "metadata": metadata or {},
        "config_snapshot": _config_snapshot(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    network: BankNetwork,
    trainer: Trainer | None = None,
) -> dict:
    """Load a checkpoint from `path` into `network` (and `trainer` if provided).

    Returns the metadata dict and the loaded iteration number as
    {"iteration": int, "metadata": dict, "config_snapshot": dict}.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(payload["network_state_dict"])
    if trainer is not None and payload.get("optimizer_state_dict") is not None:
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {
        "iteration": int(payload.get("iteration", 0)),
        "metadata": payload.get("metadata", {}),
        "config_snapshot": payload.get("config_snapshot", {}),
    }


def _config_snapshot() -> dict:
    """Capture key config values so a loaded checkpoint can be sanity-checked."""
    return {
        "INPUT_SIZE": config.INPUT_SIZE,
        "HIDDEN_SIZE": config.HIDDEN_SIZE,
        "HEAD_SIZE": config.HEAD_SIZE,
        "NUM_ACTIONS": config.NUM_ACTIONS,
        "MAX_OPPONENTS": config.MAX_OPPONENTS,
        "NUM_PLAYERS_RANGE": list(config.NUM_PLAYERS_RANGE),
        "NUM_ROUNDS": config.NUM_ROUNDS,
    }


def is_compatible(loaded_snapshot: dict) -> tuple[bool, str]:
    """
    Check whether a loaded checkpoint's architecture matches the current config.

    Returns (compatible, message). If not compatible, message describes the mismatch.
    """
    current = _config_snapshot()
    for key in ("INPUT_SIZE", "HIDDEN_SIZE", "HEAD_SIZE", "NUM_ACTIONS"):
        if loaded_snapshot.get(key) != current[key]:
            return (
                False,
                f"checkpoint {key}={loaded_snapshot.get(key)} != current {key}={current[key]}",
            )
    return True, "ok"

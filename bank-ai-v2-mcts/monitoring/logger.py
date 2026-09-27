"""
Metrics logger for Bank AI v2.

Writes a metrics.json file atomically (write to .tmp then os.replace) so the dashboard
browser-side polling never reads a partially-written file.

Also accumulates a rolling history of per-iteration metrics that the dashboard charts.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path


class MetricsLogger:
    def __init__(
        self,
        metrics_path: str | Path = "metrics.json",
        history_size: int = 5000,
        verbose: bool = True,
        resume: bool = True,
    ):
        self.metrics_path = Path(metrics_path)
        self.tmp_path = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
        self.verbose = verbose
        self.start_time = time.perf_counter()

        # Rolling histories per metric (deque of (iteration, value))
        self.history: dict[str, deque[tuple[int, float]]] = {}
        self.history_size = history_size

        # Latest snapshot
        self.latest: dict = {}

        # Status flags
        self.status: str = "idle"   # "idle", "running", "halted", "error"
        self.status_message: str = ""

        # If resuming, preserve existing chart history from disk
        if resume and self.metrics_path.exists():
            try:
                import json as _json
                with open(self.metrics_path) as f:
                    existing = _json.load(f)
                hist = existing.get("history", {})
                for k, points in hist.items():
                    self.history[k] = deque(maxlen=history_size)
                    for entry in points:
                        # Stored as [iteration, value]
                        if isinstance(entry, (list, tuple)) and len(entry) == 2:
                            self.history[k].append((int(entry[0]), float(entry[1])))
                self.latest = existing.get("latest", {})
            except (OSError, ValueError, TypeError):
                # If the metrics file is corrupt, just start fresh
                self.history.clear()
                self.latest = {}

    # ---------------------------------------------------------------- API

    def log(self, iteration: int, metrics: dict) -> None:
        """
        Record a snapshot of metrics for this iteration. Each value should be a number;
        nested dicts are stored as-is (e.g. for per-player-count win rates).

        Side effects: appends to history (for numeric values) and writes metrics.json.
        """
        timestamp = time.perf_counter() - self.start_time
        full = {
            "iteration": iteration,
            "timestamp_seconds": timestamp,
            **metrics,
        }
        self.latest = full

        # Append numeric values to history
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k not in self.history:
                    self.history[k] = deque(maxlen=self.history_size)
                self.history[k].append((iteration, float(v)))

        self._write_metrics_file()

        if self.verbose:
            self._print_terminal_line(iteration, metrics)

    def set_status(self, status: str, message: str = "") -> None:
        self.status = status
        self.status_message = message
        self._write_metrics_file()

    # ---------------------------------------------------------------- internals

    def _write_metrics_file(self) -> None:
        payload = {
            "status": self.status,
            "status_message": self.status_message,
            "latest": self.latest,
            "history": {k: list(v) for k, v in self.history.items()},
        }
        # Write atomically: write to .tmp, then rename
        try:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(self.tmp_path, self.metrics_path)
        except OSError as e:
            # Don't crash training if disk write fails (e.g. file lock)
            if self.verbose:
                print(f"[logger] WARN: could not write metrics.json: {e}")

    def _print_terminal_line(self, iteration: int, metrics: dict) -> None:
        # One-line summary for the terminal
        parts = [f"iter={iteration:>5d}"]
        for k in ("total_loss", "policy_loss", "value_loss"):
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.4f}")
        for k in ("games_per_min", "buffer_size", "win_rate"):
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.3f}")
                else:
                    parts.append(f"{k}={v}")
        print(" | ".join(parts))

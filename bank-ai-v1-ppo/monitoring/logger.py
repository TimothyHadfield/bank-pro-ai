"""
Training logger: TensorBoard + terminal output + dashboard JSON.
"""
import os
import json
import time
import tempfile
from torch.utils.tensorboard import SummaryWriter
from config import Config


class TrainingLogger:
    """Logs training metrics to TensorBoard, terminal, and a JSON file for the dashboard."""

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Dashboard data accumulator — load existing if present (for resume)
        self._dashboard_path = config.dashboard_metrics_path
        os.makedirs(os.path.dirname(self._dashboard_path) or ".", exist_ok=True)
        self._history_keys = ["avg_score", "avg_reward", "win_rate", "policy_loss", "value_loss", "entropy", "batch_time"]
        self._dashboard = self._load_existing_dashboard()

    def _load_existing_dashboard(self):
        """Load existing metrics.json if present, so resumed training keeps previous data."""
        default = {
            "last_updated": 0,
            "current_batch": 0,
            "max_batches": self.config.max_training_batches,
            "latest": {},
            "history": {"batch_nums": [], **{k: [] for k in self._history_keys}},
            "evaluations": {"batch_nums": [], "win_rate_vs_100": [], "win_rate_vs_200": []},
            "watchdog": {"halted": False, "halt_reason": "", "peak_score": 0, "below_peak_count": 0, "collapse_count": 0},
        }
        if os.path.exists(self._dashboard_path):
            try:
                with open(self._dashboard_path, "r") as f:
                    existing = json.load(f)
                # Validate it has the right structure
                if "history" in existing and "batch_nums" in existing["history"]:
                    # Clear the halted flag so training can continue
                    if "watchdog" in existing:
                        existing["watchdog"]["halted"] = False
                        existing["watchdog"]["halt_reason"] = ""
                    # Ensure all history keys exist
                    for k in self._history_keys:
                        if k not in existing["history"]:
                            existing["history"][k] = [0] * len(existing["history"]["batch_nums"])
                    return existing
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        return default

    def log_batch(self, batch_num: int, metrics: dict):
        """Log per-batch metrics to TensorBoard, terminal, and dashboard JSON."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"training/{key}", value, batch_num)

        # Terminal summary
        print(
            f"[Batch {batch_num:>6d}] "
            f"score={metrics.get('avg_score', 0):.1f}  "
            f"reward={metrics.get('avg_reward', 0):.3f}  "
            f"p_loss={metrics.get('policy_loss', 0):.4f}  "
            f"v_loss={metrics.get('value_loss', 0):.4f}  "
            f"entropy={metrics.get('entropy', 0):.4f}",
            flush=True,
        )

        # Dashboard JSON
        self._dashboard["current_batch"] = batch_num
        self._dashboard["latest"] = {k: metrics.get(k, 0) for k in self._history_keys}
        self._dashboard["history"]["batch_nums"].append(batch_num)
        for k in self._history_keys:
            self._dashboard["history"][k].append(round(metrics.get(k, 0), 6))

        # Decimate history to keep JSON small (keep last 2000 points)
        max_points = 2000
        h = self._dashboard["history"]
        if len(h["batch_nums"]) > max_points:
            step = len(h["batch_nums"]) // max_points + 1
            h["batch_nums"] = h["batch_nums"][::step] + h["batch_nums"][-200:]
            for k in self._history_keys:
                h[k] = h[k][::step] + h[k][-200:]

        self._write_json()

    def log_evaluation(self, batch_num: int, eval_metrics: dict):
        """Log evaluation metrics."""
        for key, value in eval_metrics.items():
            self.writer.add_scalar(f"evaluation/{key}", value, batch_num)

        print(
            f"  [Eval] "
            f"vs_threshold100={eval_metrics.get('win_rate_vs_100', 0):.1%}  "
            f"vs_threshold200={eval_metrics.get('win_rate_vs_200', 0):.1%}",
            flush=True,
        )

        # Dashboard
        self._dashboard["evaluations"]["batch_nums"].append(batch_num)
        self._dashboard["evaluations"]["win_rate_vs_100"].append(
            round(eval_metrics.get("win_rate_vs_100", 0), 4))
        self._dashboard["evaluations"]["win_rate_vs_200"].append(
            round(eval_metrics.get("win_rate_vs_200", 0), 4))
        self._write_json()

    def log_watchdog(self, status: dict):
        """Update watchdog status in dashboard JSON."""
        self._dashboard["watchdog"] = status
        self._write_json()

    def _write_json(self):
        """Write dashboard JSON atomically (temp file + rename)."""
        self._dashboard["last_updated"] = time.time()
        try:
            dir_path = os.path.dirname(self._dashboard_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self._dashboard, f)
            os.replace(tmp_path, self._dashboard_path)
        except OSError:
            pass  # Non-critical — dashboard just won't update this tick

    def close(self):
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()

"""
Dashboard server — serves the dashboard UI and manages training.

Usage:
    python dashboard/server.py

Then open http://localhost:5001 in your browser.
"""
import os
import sys
import json
import glob
import time
import shutil
import subprocess
import threading
import argparse
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import urllib.parse as urlparse
import torch
from config import Config
from network.model import BankNetwork
from game.environment import BankGame

# --- V2 (MCTS) project paths ---
# v2 lives at ../../bank-ai-v2-mcts/ relative to this file.
V2_PROJECT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "bank-ai-v2-mcts"))
V2_AVAILABLE = os.path.isdir(V2_PROJECT_DIR) and os.path.exists(os.path.join(V2_PROJECT_DIR, "train.py"))
V2_MODELS_DIR = os.path.join(V2_PROJECT_DIR, "models")
V2_METRICS_PATH = os.path.join(V2_PROJECT_DIR, "metrics.json")
V2_SAVES_DIR = os.path.join(V2_PROJECT_DIR, "saves")

_config = Config()
_process = None
_process_lock = threading.Lock()
_log_lines = deque(maxlen=500)
_log_lock = threading.Lock()
_last_resume = None
_server_ref = None  # Reference to HTTPServer for shutdown

# --- v2 process state (separate from v1) ---
_v2_process = None
_v2_process_lock = threading.Lock()
_v2_log_lines = deque(maxlen=500)
_v2_log_lock = threading.Lock()

# --- unified Play vs AI session (single active game, local-only) ---
_play_session_lock = threading.Lock()
_play_session = {
    "game": None,           # v2 BankGame instance
    "ai_competitor": None,  # Competitor wrapping v1 net OR v2 net+MCTS OR custom code
    "ai_type": None,        # "v1" | "v2" | "custom"
    "ai_label": None,       # human-readable name
    "human_seats": set(),
    "history": [],          # list of {round, actions, dice}
    "rng_seed": 0,
}


def _read_v2_output(proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            with _v2_log_lock:
                _v2_log_lines.append(line.rstrip("\n\r"))
    except (ValueError, OSError):
        pass


def _v2_is_running():
    global _v2_process
    if _v2_process is None:
        return False
    if _v2_process.poll() is not None:
        _v2_process = None
        return False
    return True


def _read_v2_metrics():
    if not os.path.exists(V2_METRICS_PATH):
        return None
    try:
        with open(V2_METRICS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _serialize_play_state(game, human_seats, revealed_actions=None):
    """JSON-safe snapshot of a play game's state."""
    return {
        "num_players": game.num_players,
        "round_number": game.round_number,
        "roll_number": game.roll_number,
        "pot": game.pot,
        "banked_scores": list(game.banked_scores),
        "players_in_round": sorted(game.players_in_round),
        "last_dice": list(game.last_dice) if game.last_dice else None,
        "immunity_active": game.roll_number <= 3,
        "rolls_since_immunity_ended": game.rolls_since_immunity_ended,
        "game_over": game.game_over,
        "round_over": game.round_over,
        "waiting_for_decisions": game.waiting_for_decisions,
        "winners": game.winners() if game.game_over else [],
        "human_seats": sorted(human_seats),
        "ai_seats": sorted(set(range(game.num_players)) - human_seats),
        "revealed_actions": revealed_actions,
    }


def _read_output(proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            with _log_lock:
                _log_lines.append(line.rstrip("\n\r"))
    except (ValueError, OSError):
        pass


def _is_running():
    global _process
    if _process is None:
        return False
    if _process.poll() is not None:
        _process = None
        return False
    return True


class DashboardHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def end_headers(self):
        # Disable caching on ALL responses so the browser always gets fresh files
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api_get()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            try:
                self._handle_api_post()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle_api_get(self):
        path = self.path.split("?")[0]

        if path == "/api/status":
            running = _is_running()
            metrics = _read_metrics()
            wd = metrics.get("watchdog", {}) if metrics else {}
            self._json_response({
                "running": running,
                "halted": wd.get("halted", False),
                "halt_reason": wd.get("halt_reason", ""),
            })

        elif path == "/api/checkpoints":
            models_dir = os.path.join(PROJECT_DIR, _config.model_dir)
            checkpoints = []
            if os.path.exists(models_dir):
                for f in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
                    if "._deleted_" in f:
                        continue
                    stat = os.stat(f)
                    checkpoints.append({
                        "filename": os.path.basename(f),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    })
            self._json_response({"checkpoints": checkpoints})

        elif path == "/api/logs":
            with _log_lock:
                lines = list(_log_lines)
            self._json_response({"lines": lines})

        elif path == "/api/config":
            self._json_response(vars(Config()))

        elif path == "/api/overrides":
            overrides = _read_overrides()
            self._json_response(overrides)

        elif path == "/api/saves":
            saves_dir = os.path.join(PROJECT_DIR, "saves")
            saves = []
            if os.path.exists(saves_dir):
                for d in sorted(os.listdir(saves_dir)):
                    if "._deleted_" in d:
                        continue
                    full = os.path.join(saves_dir, d)
                    if not os.path.isdir(full):
                        continue
                    # Find the .pt file and metrics
                    pts = glob.glob(os.path.join(full, "*.pt"))
                    has_metrics = os.path.exists(os.path.join(full, "metrics.json"))
                    has_log = os.path.exists(os.path.join(full, "training_log.txt"))
                    batch_num = 0
                    if pts:
                        # Extract batch num from folder name
                        parts = d.rsplit("_batch", 1)
                        if len(parts) == 2:
                            try: batch_num = int(parts[1])
                            except ValueError: pass
                    saves.append({
                        "name": d,
                        "batch_num": batch_num,
                        "has_checkpoint": len(pts) > 0,
                        "has_metrics": has_metrics,
                        "has_log": has_log,
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(full))),
                    })
            self._json_response({"saves": saves})

        elif path.startswith("/api/weights/"):
            filename = urlparse.unquote(path.split("/api/weights/", 1)[1])
            ckpt_path = os.path.join(PROJECT_DIR, _config.model_dir, filename)
            if not os.path.exists(ckpt_path):
                self._json_response({"error": "Checkpoint not found: " + filename}, 404)
                return
            try:
                state_size = BankGame.state_size()
                net = BankNetwork(state_size=state_size, hidden_size=_config.hidden_size)
                ckpt = torch.load(ckpt_path, weights_only=False)
                net.load_state_dict(ckpt["network_state_dict"])
                weights = {}
                for name, param in net.policy.named_parameters():
                    weights[name] = param.detach().numpy().tolist()
                self._json_response({
                    "batch_num": ckpt.get("batch_num", 0),
                    "hidden_size": _config.hidden_size,
                    "state_size": state_size,
                    "weights": weights,
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response({"error": str(e)}, 500)

        # ============================================================
        # V2 (MCTS) endpoints
        # ============================================================
        elif path == "/api/v2/status":
            running = _v2_is_running()
            metrics = _read_v2_metrics() or {}
            self._json_response({
                "available": V2_AVAILABLE,
                "running": running,
                "status": metrics.get("status", "idle"),
                "status_message": metrics.get("status_message", ""),
            })

        elif path == "/api/v2/metrics":
            self._json_response(_read_v2_metrics() or {"status": "idle", "history": {}, "latest": {}})

        elif path == "/api/v2/logs":
            with _v2_log_lock:
                lines = list(_v2_log_lines)
            self._json_response({"lines": lines})

        elif path == "/api/v2/checkpoints":
            checkpoints = []
            if os.path.isdir(V2_MODELS_DIR):
                for f in sorted(glob.glob(os.path.join(V2_MODELS_DIR, "*.pt"))):
                    if "._deleted_" in f:
                        continue
                    stat = os.stat(f)
                    checkpoints.append({
                        "filename": os.path.basename(f),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    })
            self._json_response({"checkpoints": checkpoints})

        elif path == "/api/play_unified/state":
            with _play_session_lock:
                game = _play_session["game"]
                if game is None:
                    self._json_response({"ok": False, "error": "no active game"})
                    return
                self._json_response({
                    "ok": True,
                    "ai_type": _play_session["ai_type"],
                    "ai_label": _play_session["ai_label"],
                    "state": _serialize_play_state(game, _play_session["human_seats"]),
                    "history": _play_session["history"][-30:],
                })

        elif path == "/api/v2/saves":
            saves = []
            if os.path.isdir(V2_SAVES_DIR):
                for d in sorted(os.listdir(V2_SAVES_DIR)):
                    full = os.path.join(V2_SAVES_DIR, d)
                    if not os.path.isdir(full) or "._deleted_" in d:
                        continue
                    saves.append({
                        "name": d,
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(full))),
                    })
            self._json_response({"saves": saves})

        else:
            self.send_error(404)

    def _handle_api_post(self):
        global _process, _last_resume, _v2_process
        path = self.path.split("?")[0]
        print(f"[API POST] {path}", flush=True)
        body = self._read_body()

        if path == "/api/start":
            with _process_lock:
                if _is_running():
                    self._json_response({"error": "Training is already running."}, 400)
                    return

                resume = body.get("resume") if body else None
                if not resume and _last_resume:
                    resume = _last_resume

                cmd = [sys.executable, "-u", os.path.join(PROJECT_DIR, "train.py")]
                if resume:
                    resume_path = resume if os.path.isabs(resume) else os.path.join(PROJECT_DIR, "models", resume)
                    if os.path.exists(resume_path):
                        cmd += ["--resume", resume_path]
                        _last_resume = resume

                with _log_lock:
                    _log_lines.append(f"[Dashboard] Starting: {' '.join(cmd)}")

                try:
                    _process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=PROJECT_DIR,
                    )
                    threading.Thread(target=_read_output, args=(_process,), daemon=True).start()
                    self._json_response({"status": "started", "pid": _process.pid})
                except Exception as e:
                    self._json_response({"error": str(e)}, 500)

        elif path == "/api/stop":
            # Pause — saves checkpoint automatically via train.py's final save
            with _process_lock:
                if not _is_running():
                    self._json_response({"status": "not_running"})
                    return
                _process.terminate()
                with _log_lock:
                    _log_lines.append("[Dashboard] Training paused.")
                # Find latest checkpoint for auto-resume
                self._update_last_resume()
                self._json_response({"status": "paused"})

        elif path == "/api/reset":
            # Full reset — stop training, clear metrics
            with _process_lock:
                if _is_running():
                    _process.terminate()
                    time.sleep(1)
            _last_resume = None
            # Clear metrics file
            metrics_path = os.path.join(PROJECT_DIR, _config.dashboard_metrics_path)
            if os.path.exists(metrics_path):
                os.remove(metrics_path)
            with _log_lock:
                _log_lines.clear()
                _log_lines.append("[Dashboard] Training reset. All progress cleared.")
            self._json_response({"status": "reset"})

        elif path == "/api/save":
            # Save current training state with a user-provided name
            save_name = body.get("name", "save") if body else "save"
            save_name = "".join(c for c in save_name if c.isalnum() or c in "-_").strip() or "save"

            # Find the latest checkpoint
            models_dir = os.path.join(PROJECT_DIR, _config.model_dir)
            pts = sorted(glob.glob(os.path.join(models_dir, "*.pt")), key=os.path.getmtime)
            if not pts:
                self._json_response({"error": "No checkpoints to save"}, 400)
                return

            latest_ckpt = pts[-1]
            # Determine batch number from checkpoint
            try:
                import torch
                ckpt = torch.load(latest_ckpt, weights_only=False)
                batch_num = ckpt.get("batch_num", 0)
            except Exception:
                batch_num = 0

            # Create save directory
            saves_dir = os.path.join(PROJECT_DIR, "saves", f"{save_name}_batch{batch_num}")
            os.makedirs(saves_dir, exist_ok=True)

            # Copy checkpoint
            dest_ckpt = os.path.join(saves_dir, f"{save_name}_batch{batch_num}.pt")
            shutil.copy2(latest_ckpt, dest_ckpt)

            # Save training log
            with _log_lock:
                log_text = "\n".join(_log_lines)
            with open(os.path.join(saves_dir, "training_log.txt"), "w") as f:
                f.write(log_text)

            # Copy metrics
            metrics_path = os.path.join(PROJECT_DIR, _config.dashboard_metrics_path)
            if os.path.exists(metrics_path):
                shutil.copy2(metrics_path, os.path.join(saves_dir, "metrics.json"))

            with _log_lock:
                _log_lines.append(f"[Dashboard] Saved as '{save_name}' at batch {batch_num} -> {saves_dir}")

            self._json_response({
                "status": "saved",
                "path": saves_dir,
                "checkpoint": f"{save_name}_batch{batch_num}.pt",
                "batch_num": batch_num,
            })

        elif path == "/api/export":
            checkpoint = body.get("checkpoint") if body else None
            if not checkpoint:
                self._json_response({"error": "No checkpoint specified"}, 400)
                return
            ckpt_path = os.path.join(PROJECT_DIR, _config.model_dir, checkpoint)
            if not os.path.exists(ckpt_path):
                self._json_response({"error": "Checkpoint not found"}, 404)
                return
            try:
                result = subprocess.run(
                    [sys.executable, os.path.join(PROJECT_DIR, "export_weights.py"), ckpt_path],
                    capture_output=True, text=True, timeout=30, cwd=PROJECT_DIR,
                )
                self._json_response({"status": "ok", "output": result.stdout + result.stderr})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif path == "/api/ai_vs_ai":
            model1 = body.get("model1") if body else None
            model2 = body.get("model2") if body else None
            num_games = int(body.get("games", 500)) if body else 500
            if not model1 or not model2:
                self._json_response({"error": "Specify model1 and model2"}, 400)
                return
            try:
                from evaluation.ai_vs_ai import run_match
                state_size = BankGame.state_size()

                net1 = BankNetwork(state_size=state_size, hidden_size=_config.hidden_size)
                ckpt1 = torch.load(os.path.join(PROJECT_DIR, _config.model_dir, model1), weights_only=False)
                net1.load_state_dict(ckpt1["network_state_dict"])
                net1.eval()

                net2 = BankNetwork(state_size=state_size, hidden_size=_config.hidden_size)
                ckpt2 = torch.load(os.path.join(PROJECT_DIR, _config.model_dir, model2), weights_only=False)
                net2.load_state_dict(ckpt2["network_state_dict"])
                net2.eval()

                results = run_match(net1, net2, _config, num_games=num_games)
                results["model1_name"] = model1
                results["model2_name"] = model2
                results["model1_batch"] = ckpt1.get("batch_num", "?")
                results["model2_batch"] = ckpt2.get("batch_num", "?")
                self._json_response(results)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif path == "/api/load_save":
            # Restore a saved session: copy metrics + log back, set resume checkpoint
            save_name = body.get("name") if body else None
            if not save_name:
                self._json_response({"error": "No save name specified"}, 400)
                return
            save_dir = os.path.join(PROJECT_DIR, "saves", save_name)
            if not os.path.isdir(save_dir):
                self._json_response({"error": f"Save not found: {save_name}"}, 404)
                return

            # Stop training if running
            with _process_lock:
                if _is_running():
                    _process.terminate()
                    time.sleep(1)

            # Restore metrics.json
            saved_metrics = os.path.join(save_dir, "metrics.json")
            metrics_dest = os.path.join(PROJECT_DIR, _config.dashboard_metrics_path)
            if os.path.exists(saved_metrics):
                os.makedirs(os.path.dirname(metrics_dest) or ".", exist_ok=True)
                shutil.copy2(saved_metrics, metrics_dest)

            # Restore training log
            saved_log = os.path.join(save_dir, "training_log.txt")
            if os.path.exists(saved_log):
                with open(saved_log, "r") as f:
                    lines = f.read().strip().split("\n")
                with _log_lock:
                    _log_lines.clear()
                    for line in lines:
                        _log_lines.append(line)
                    _log_lines.append(f"[Dashboard] Restored session: {save_name}")

            # Copy checkpoint to models/ and set as resume target
            pts = glob.glob(os.path.join(save_dir, "*.pt"))
            if pts:
                ckpt_src = pts[0]
                ckpt_dest = os.path.join(PROJECT_DIR, _config.model_dir, os.path.basename(ckpt_src))
                if not os.path.exists(ckpt_dest):
                    shutil.copy2(ckpt_src, ckpt_dest)
                _last_resume = os.path.basename(ckpt_dest)

            self._json_response({
                "status": "loaded",
                "name": save_name,
                "resume_checkpoint": _last_resume,
            })

        elif path == "/api/delete_all_checkpoints":
            models_dir = os.path.join(PROJECT_DIR, _config.model_dir)
            deleted = 0
            if os.path.exists(models_dir):
                for f in glob.glob(os.path.join(models_dir, "*.pt")):
                    if "._deleted_" in f:
                        continue
                    try:
                        os.remove(f)
                        deleted += 1
                    except (PermissionError, OSError):
                        try:
                            os.rename(f, f + "._deleted_" + str(int(time.time())))
                            deleted += 1
                        except:
                            pass
            self._json_response({"status": "deleted", "count": deleted})

        elif path == "/api/delete_checkpoint":
            filename = body.get("filename") if body else None
            if not filename:
                self._json_response({"error": "No filename specified"}, 400)
                return
            filepath = os.path.join(PROJECT_DIR, _config.model_dir, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    self._json_response({"status": "deleted", "filename": filename})
                except (PermissionError, OSError):
                    try:
                        os.rename(filepath, filepath + "._deleted_" + str(int(time.time())))
                        self._json_response({"status": "deleted", "filename": filename})
                    except Exception as e:
                        self._json_response({"error": str(e)}, 500)
            else:
                self._json_response({"error": "File not found"}, 404)

        elif path == "/api/delete_save":
            save_name = body.get("name") if body else None
            if not save_name:
                self._json_response({"error": "No name specified"}, 400)
                return
            save_dir = os.path.join(PROJECT_DIR, "saves", save_name)
            if os.path.isdir(save_dir):
                try:
                    # Try normal delete first
                    shutil.rmtree(save_dir)
                    self._json_response({"status": "deleted", "name": save_name})
                except (PermissionError, OSError):
                    try:
                        # OneDrive lock — rename to .deleted so it disappears from the list
                        trash = save_dir + "._deleted_" + str(int(time.time()))
                        os.rename(save_dir, trash)
                        self._json_response({"status": "deleted", "name": save_name})
                    except Exception as e:
                        self._json_response({"error": str(e)}, 500)
            else:
                self._json_response({"error": "Save not found"}, 404)

        elif path == "/api/export_for_play":
            # Export a checkpoint's policy weights to dashboard/play_weights.json
            checkpoint = body.get("checkpoint") if body else None
            if not checkpoint:
                self._json_response({"error": "No checkpoint specified"}, 400)
                return
            ckpt_path = os.path.join(PROJECT_DIR, _config.model_dir, checkpoint)
            if not os.path.exists(ckpt_path):
                self._json_response({"error": f"Checkpoint not found: {checkpoint}"}, 404)
                return
            try:
                # Use export_weights.py to write to play/, then copy to dashboard/
                result = subprocess.run(
                    [sys.executable, os.path.join(PROJECT_DIR, "export_weights.py"), ckpt_path],
                    capture_output=True, text=True, timeout=30, cwd=PROJECT_DIR,
                )
                if result.returncode != 0:
                    self._json_response({"error": result.stderr or result.stdout}, 500)
                    return
                # Copy play/weights.json to dashboard/play_weights.json
                src = os.path.join(PROJECT_DIR, "play", "weights.json")
                dst = os.path.join(DASHBOARD_DIR, "play_weights.json")
                shutil.copy2(src, dst)
                # Read batch_num from the file
                with open(dst) as f:
                    batch_num = json.load(f).get("batch_num", 0)
                self._json_response({"status": "ok", "batch_num": batch_num})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif path == "/api/overrides":
            # Save runtime overrides that train.py reads each batch
            overrides = body or {}
            overrides_path = os.path.join(PROJECT_DIR, "dashboard", "overrides.json")
            with open(overrides_path, "w") as f:
                json.dump(overrides, f)
            with _log_lock:
                _log_lines.append(f"[Dashboard] Settings updated: {overrides}")
            self._json_response({"status": "saved", "overrides": overrides})

        elif path == "/api/shutdown":
            self._json_response({"status": "shutting_down"})
            # Stop training if running
            with _process_lock:
                if _is_running():
                    _process.terminate()
            with _v2_process_lock:
                if _v2_is_running():
                    _v2_process.terminate()
            # Shut down server after a short delay
            threading.Thread(target=_delayed_shutdown, daemon=True).start()

        # ============================================================
        # V2 (MCTS) POST endpoints — control v2's train.py subprocess
        # ============================================================
        elif path == "/api/v2/start":
            if not V2_AVAILABLE:
                self._json_response({"error": "v2 project not found at " + V2_PROJECT_DIR}, 400)
                return
            with _v2_process_lock:
                if _v2_is_running():
                    self._json_response({"error": "v2 training is already running"}, 400)
                    return
                args = (body or {}).get("args", {}) or {}
                cmd = [sys.executable, "-u", os.path.join(V2_PROJECT_DIR, "train.py")]
                allowed = {"max_iterations", "games_per_iter", "steps_per_iter",
                           "sims_per_move", "batch_size", "workers", "resume", "seed",
                           "eval_interval", "eval_games"}
                for k, v in args.items():
                    if k in allowed and v not in (None, ""):
                        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
                with _v2_log_lock:
                    _v2_log_lines.append(f"[Dashboard] Starting v2: {' '.join(cmd)}")
                try:
                    _v2_process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=V2_PROJECT_DIR,
                    )
                    threading.Thread(target=_read_v2_output, args=(_v2_process,), daemon=True).start()
                    self._json_response({"status": "started", "pid": _v2_process.pid})
                except Exception as e:
                    self._json_response({"error": str(e)}, 500)

        elif path == "/api/v2/stop":
            with _v2_process_lock:
                if not _v2_is_running():
                    self._json_response({"status": "not_running"})
                    return
                _v2_process.terminate()
                with _v2_log_lock:
                    _v2_log_lines.append("[Dashboard] v2 training paused.")
                self._json_response({"status": "paused"})

        elif path == "/api/v2/reset":
            with _v2_process_lock:
                if _v2_is_running():
                    _v2_process.terminate()
                    time.sleep(1)
            deleted = []
            if os.path.isdir(V2_MODELS_DIR):
                for f in glob.glob(os.path.join(V2_MODELS_DIR, "*.pt")):
                    if "._deleted_" in f:
                        continue
                    try:
                        os.remove(f); deleted.append(os.path.basename(f))
                    except OSError:
                        try:
                            os.rename(f, f + "._deleted_" + str(int(time.time())))
                            deleted.append(os.path.basename(f) + " (renamed)")
                        except OSError:
                            pass
            if os.path.exists(V2_METRICS_PATH):
                try: os.remove(V2_METRICS_PATH)
                except OSError: pass
            with _v2_log_lock:
                _v2_log_lines.clear()
                _v2_log_lines.append("[Dashboard] v2 training reset.")
            self._json_response({"status": "reset", "deleted": deleted})

        elif path == "/api/v2/save":
            save_name = (body or {}).get("name", "save")
            save_name = "".join(c for c in save_name if c.isalnum() or c in "-_").strip() or "save"
            if not os.path.isdir(V2_MODELS_DIR):
                self._json_response({"error": "no v2 models directory"}, 400)
                return
            os.makedirs(V2_SAVES_DIR, exist_ok=True)
            target = os.path.join(V2_SAVES_DIR, save_name)
            if os.path.exists(target):
                self._json_response({"error": f"save '{save_name}' already exists"}, 400)
                return
            os.makedirs(target)
            ckpt_dest = os.path.join(target, "models")
            os.makedirs(ckpt_dest)
            for p in glob.glob(os.path.join(V2_MODELS_DIR, "*.pt")):
                shutil.copy2(p, os.path.join(ckpt_dest, os.path.basename(p)))
            if os.path.exists(V2_METRICS_PATH):
                shutil.copy2(V2_METRICS_PATH, os.path.join(target, "metrics.json"))
            self._json_response({"status": "saved", "name": save_name})

        elif path == "/api/v2/load_save":
            save_name = (body or {}).get("name")
            if not save_name:
                self._json_response({"error": "no save name"}, 400)
                return
            src = os.path.join(V2_SAVES_DIR, save_name)
            if not os.path.isdir(src):
                self._json_response({"error": "save not found"}, 404)
                return
            with _v2_process_lock:
                if _v2_is_running():
                    _v2_process.terminate()
                    time.sleep(1)
            # Wipe current v2 models and metrics
            if os.path.isdir(V2_MODELS_DIR):
                for f in glob.glob(os.path.join(V2_MODELS_DIR, "*.pt")):
                    try: os.remove(f)
                    except OSError: pass
            if os.path.exists(V2_METRICS_PATH):
                try: os.remove(V2_METRICS_PATH)
                except OSError: pass
            saved_models = os.path.join(src, "models")
            if os.path.isdir(saved_models):
                os.makedirs(V2_MODELS_DIR, exist_ok=True)
                for p in glob.glob(os.path.join(saved_models, "*.pt")):
                    shutil.copy2(p, os.path.join(V2_MODELS_DIR, os.path.basename(p)))
            saved_metrics = os.path.join(src, "metrics.json")
            if os.path.exists(saved_metrics):
                shutil.copy2(saved_metrics, V2_METRICS_PATH)
            self._json_response({"status": "loaded", "name": save_name})

        elif path == "/api/play_unified/new":
            # Body:
            #   {
            #     "ai_type": "v1" | "v2" | "custom",
            #     "ai_checkpoint": "..."  (for v1/v2),
            #     "ai_code":  "..."  (for custom),
            #     "ai_label": "Pretty name",
            #     "num_players": 4,
            #     "human_seats": [0],
            #     "sims_per_move": 100  (v2 only),
            #   }
            try:
                from dashboard.unified_tournament import (
                    make_v1_competitor, make_v2_competitor, make_custom_competitor,
                    V2BankGame,
                )
                spec = body or {}
                ai_type = spec.get("ai_type")
                num_players = int(spec.get("num_players", 4))
                if not (2 <= num_players <= 6):
                    self._json_response({"ok": False, "error": "num_players must be 2..6"}, 400)
                    return
                human_seats = set(int(s) for s in spec.get("human_seats", [0]) if 0 <= int(s) < num_players)
                if not human_seats:
                    self._json_response({"ok": False, "error": "at least one human seat required"}, 400)
                    return

                if ai_type == "v1":
                    ckpt = spec.get("ai_checkpoint")
                    if not ckpt:
                        self._json_response({"ok": False, "error": "v1 type requires ai_checkpoint"}, 400)
                        return
                    ck_path = os.path.join(PROJECT_DIR, _config.model_dir, ckpt)
                    if not os.path.exists(ck_path):
                        self._json_response({"ok": False, "error": f"v1 checkpoint not found: {ckpt}"}, 400)
                        return
                    competitor = make_v1_competitor(ck_path, name=f"v1:{ckpt}")
                elif ai_type == "v2":
                    ckpt = spec.get("ai_checkpoint")
                    if not ckpt:
                        self._json_response({"ok": False, "error": "v2 type requires ai_checkpoint"}, 400)
                        return
                    ck_path = os.path.join(V2_MODELS_DIR, ckpt)
                    if not os.path.exists(ck_path):
                        self._json_response({"ok": False, "error": f"v2 checkpoint not found: {ckpt}"}, 400)
                        return
                    sims = int(spec.get("sims_per_move", 100))
                    competitor = make_v2_competitor(ck_path, name=f"v2:{ckpt}", sims_per_move=sims)
                elif ai_type == "custom":
                    code = spec.get("ai_code")
                    label = spec.get("ai_label", "BuildAI")
                    if not code:
                        self._json_response({"ok": False, "error": "custom type requires ai_code"}, 400)
                        return
                    try:
                        competitor = make_custom_competitor(code, name=label)
                    except (ValueError, RuntimeError) as e:
                        self._json_response({"ok": False, "error": f"compile failed: {e}"}, 400)
                        return
                else:
                    self._json_response({"ok": False, "error": f"Unknown ai_type: {ai_type!r}"}, 400)
                    return

                seed = int(time.time() * 1000) & 0x7FFFFFFF
                game = V2BankGame(num_players=num_players, seed=seed)
                game.reset()
                with _play_session_lock:
                    _play_session["game"] = game
                    _play_session["ai_competitor"] = competitor
                    _play_session["ai_type"] = ai_type
                    _play_session["ai_label"] = competitor.name
                    _play_session["human_seats"] = human_seats
                    _play_session["history"] = []
                    _play_session["rng_seed"] = seed
                self._json_response({
                    "ok": True,
                    "ai_type": ai_type,
                    "ai_label": competitor.name,
                    "state": _serialize_play_state(game, human_seats),
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

        elif path == "/api/play_unified/move":
            try:
                import numpy as np
                from dashboard.unified_tournament import BANK, CONTINUE
                with _play_session_lock:
                    game = _play_session["game"]
                    competitor = _play_session["ai_competitor"]
                    if game is None or competitor is None:
                        self._json_response({"ok": False, "error": "no active game"}, 400)
                        return
                    if game.is_terminal():
                        self._json_response({"ok": False, "error": "game over"}, 400)
                        return
                    if not game.waiting_for_decisions:
                        self._json_response({"ok": False, "error": "engine not waiting"}, 400)
                        return

                    active = game.players_in_round
                    human_seats = _play_session["human_seats"]
                    actions = {}

                    # Human-supplied actions
                    raw = (body or {}).get("actions", {}) or {}
                    for pid_str, val in raw.items():
                        pid = int(pid_str)
                        if pid in active and pid in human_seats:
                            if val not in (BANK, CONTINUE):
                                self._json_response({"ok": False, "error": f"invalid action {val!r}"}, 400)
                                return
                            actions[pid] = val
                    missing = [pid for pid in active if pid in human_seats and pid not in actions]
                    if missing:
                        self._json_response({"ok": False, "error": f"missing human action for seats {missing}"}, 400)
                        return

                    # AI seats decide via the unified competitor
                    rng = np.random.default_rng(seed=_play_session["rng_seed"] + int(time.time() * 1000))
                    for pid in sorted(active):
                        if pid in human_seats:
                            continue
                        actions[pid] = competitor.decide(game, pid, rng)

                    revealed = dict(actions)
                    game.step(actions)
                    _play_session["history"].append({
                        "round": game.round_number,
                        "actions": revealed,
                        "dice": list(game.last_dice) if game.last_dice else None,
                    })

                    self._json_response({
                        "ok": True,
                        "state": _serialize_play_state(game, human_seats, revealed_actions=revealed),
                        "history": _play_session["history"][-30:],
                    })
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

        elif path == "/api/aivai_combined":
            # Unified AI vs AI: v1 + v2 + Build AI custom strategies in any combination.
            # Body: {
            #   "num_players": 4,
            #   "games_per_matchup": 30,
            #   "competitors": [
            #     {"type": "v1", "checkpoint": "filename.pt", "name": "v1-foo"},
            #     {"type": "v2", "checkpoint": "iter_500.pt", "name": "v2-500", "sims_per_move": 100},
            #     {"type": "custom", "code": "if pot >= 200: bank\nelse: continue", "name": "MyBot"}
            #   ]
            # }
            try:
                from dashboard.unified_tournament import (
                    make_v1_competitor, make_v2_competitor, make_custom_competitor, run_tournament,
                )
                spec = body or {}
                num_players = int(spec.get("num_players", 4))
                games_per_matchup = int(spec.get("games_per_matchup", 30))
                competitor_specs = spec.get("competitors", [])
                if len(competitor_specs) < num_players:
                    self._json_response(
                        {"error": f"Need at least {num_players} competitors for {num_players}-player games (got {len(competitor_specs)})"},
                        400,
                    )
                    return

                competitors = []
                for cs in competitor_specs:
                    t = cs.get("type")
                    name = cs.get("name") or t
                    if t == "v1":
                        ckpt = cs.get("checkpoint")
                        if not ckpt:
                            self._json_response({"error": f"{name}: v1 type requires checkpoint"}, 400)
                            return
                        ck_path = os.path.join(PROJECT_DIR, _config.model_dir, ckpt)
                        if not os.path.exists(ck_path):
                            self._json_response({"error": f"{name}: v1 checkpoint not found: {ckpt}"}, 400)
                            return
                        competitors.append(make_v1_competitor(ck_path, name))
                    elif t == "v2":
                        ckpt = cs.get("checkpoint")
                        if not ckpt:
                            self._json_response({"error": f"{name}: v2 type requires checkpoint"}, 400)
                            return
                        ck_path = os.path.join(V2_MODELS_DIR, ckpt)
                        if not os.path.exists(ck_path):
                            self._json_response({"error": f"{name}: v2 checkpoint not found: {ckpt}"}, 400)
                            return
                        sims = int(cs.get("sims_per_move", 100))
                        competitors.append(make_v2_competitor(ck_path, name, sims_per_move=sims))
                    elif t == "custom":
                        code = cs.get("code")
                        if not code:
                            self._json_response({"error": f"{name}: custom type requires code"}, 400)
                            return
                        try:
                            competitors.append(make_custom_competitor(code, name))
                        except (ValueError, RuntimeError) as e:
                            self._json_response({"error": f"{name}: compile failed: {e}"}, 400)
                            return
                    else:
                        self._json_response({"error": f"Unknown competitor type: {t}"}, 400)
                        return

                results = run_tournament(
                    competitors=competitors,
                    num_players=num_players,
                    games_per_matchup=games_per_matchup,
                )
                self._json_response({"status": "ok", **results})
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response({"error": f"{type(e).__name__}: {e}"}, 500)

        elif path == "/api/v2/delete_save":
            save_name = (body or {}).get("name")
            if not save_name:
                self._json_response({"error": "no save name"}, 400)
                return
            target = os.path.join(V2_SAVES_DIR, save_name)
            if not os.path.isdir(target):
                self._json_response({"error": "save not found"}, 404)
                return
            try:
                shutil.rmtree(target)
                self._json_response({"status": "deleted", "name": save_name})
            except OSError:
                try:
                    os.rename(target, target + "._deleted_" + str(int(time.time())))
                    self._json_response({"status": "deleted", "name": save_name})
                except OSError as e:
                    self._json_response({"error": str(e)}, 500)

        else:
            print(f"[API POST] Unknown route: {path}", flush=True)
            self._json_response({"error": f"Unknown API route: {path}"}, 404)

    def _update_last_resume(self):
        global _last_resume
        models_dir = os.path.join(PROJECT_DIR, _config.model_dir)
        pts = sorted(glob.glob(os.path.join(models_dir, "*.pt")), key=os.path.getmtime)
        if pts:
            _last_resume = os.path.basename(pts[-1])

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            try:
                return json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _json_response(self, data, code=200):
        try:
            body = json.dumps(data).encode("utf-8")
        except (TypeError, ValueError):
            body = b'{"error":"serialization error"}'
            code = 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _delayed_shutdown():
    """Shut down the server after a brief delay so the response can be sent."""
    time.sleep(0.5)
    if _server_ref:
        _server_ref.shutdown()
    os._exit(0)


def _read_metrics():
    path = os.path.join(PROJECT_DIR, _config.dashboard_metrics_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_overrides():
    path = os.path.join(PROJECT_DIR, "dashboard", "overrides.json")
    if not os.path.exists(path):
        return {"batch_size": _config.batch_size, "entropy_coefficient": _config.entropy_coefficient, "watchdog_enabled": _config.watchdog_enabled}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="Bank AI Dashboard")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    print(f"\n  Bank AI Training Dashboard")
    print(f"  Open your browser to: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    global _server_ref
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    _server_ref = server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        with _process_lock:
            if _is_running():
                _process.terminate()
                print("v1 training process stopped.")
        with _v2_process_lock:
            if _v2_is_running():
                _v2_process.terminate()
                print("v2 training process stopped.")
        server.server_close()


if __name__ == "__main__":
    main()

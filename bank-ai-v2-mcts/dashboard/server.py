"""
Bank AI v2 dashboard server.

Local-only HTTP server that:
  - Serves the static dashboard UI (index.html, JS, CSS)
  - Exposes a JSON API to control training (start/stop/reset/save/load)
  - Exposes a JSON API to play games against the trained AI (server-side MCTS)
  - Exposes utility APIs (dice statistics, tournament runs, etc.)

Per PLAN.md §0 rule 2: this is local-only. No network hosting.

Usage:
    python -m dashboard.server
    # Then open http://localhost:5002/ in your browser

Default port: 5002 (v1 used 5001).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Make the parent directory importable so we can import config, network, training
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

PORT = 5002
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
METRICS_PATH = PROJECT_ROOT / "metrics.json"
MODELS_DIR = PROJECT_ROOT / "models"
SAVES_DIR = PROJECT_ROOT / "saves"
TRAIN_LOG_PATH = PROJECT_ROOT / "train.log"


# ============================================================================
# Training subprocess management
# ============================================================================

class TrainingProcess:
    """Wraps a training subprocess so we can start/stop/check it."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.start_time: float | None = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def start(self, args: dict) -> dict:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return {"ok": False, "error": "training already running"}

            cmd = [sys.executable, "-u", str(PROJECT_ROOT / "train.py")]
            for k, v in args.items():
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

            # Truncate the log file so the dashboard shows only this run
            self.log_file = open(TRAIN_LOG_PATH, "w", buffering=1)
            self.log_file.write(f"# training started at {datetime.now().isoformat()}\n")
            self.log_file.write(f"# command: {' '.join(cmd)}\n\n")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=self.log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if os.name == "nt"
                        else 0
                    ),
                )
            except Exception as e:
                if self.log_file:
                    self.log_file.close()
                    self.log_file = None
                return {"ok": False, "error": f"failed to start: {e}"}

            self.start_time = time.time()
            return {"ok": True, "pid": self.process.pid}

    def stop(self) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return {"ok": False, "error": "no training running"}

            try:
                if os.name == "nt":
                    # Send Ctrl+Break to the process group so train.py's KeyboardInterrupt fires
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)

                # Wait briefly for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            except Exception as e:
                return {"ok": False, "error": f"failed to stop: {e}"}
            finally:
                if self.log_file:
                    self.log_file.close()
                    self.log_file = None

            return {"ok": True}

    def status(self) -> dict:
        running = self.is_running()
        return {
            "running": running,
            "pid": self.process.pid if running and self.process else None,
            "start_time": self.start_time if running else None,
            "elapsed_seconds": (time.time() - self.start_time) if running and self.start_time else 0,
        }


training = TrainingProcess()


# ============================================================================
# Lazy-loaded state for Play vs AI
# ============================================================================

class PlaySession:
    """One active human-vs-AI game. Server-side MCTS runs synchronously."""

    def __init__(self):
        self.lock = threading.Lock()
        self.game = None              # BankGame
        self.network = None           # BankNetwork
        self.checkpoint_path: str | None = None
        self.human_seats: set[int] = set()
        self.action_history: list[dict] = []  # for client display
        self.sims_per_move: int = config.SIMS_PER_MOVE_PLAY_QUALITY


play_session = PlaySession()


def _load_network(checkpoint_path: str | None):
    """Lazily import torch/network/training only when actually needed."""
    from network import BankNetwork
    from training import load_checkpoint
    net = BankNetwork()
    if checkpoint_path:
        load_checkpoint(checkpoint_path, net)
    return net


def _serialize_game_state(game, human_seats: set[int], reveal_actions: dict | None = None) -> dict:
    """
    Build a JSON-friendly snapshot of the current game for the client.
    `reveal_actions` is shown in the response only AFTER all players have committed.
    """
    return {
        "num_players": game.num_players,
        "round_number": game.round_number,
        "roll_number": game.roll_number,
        "pot": game.pot,
        "banked_scores": list(game.banked_scores),
        "players_in_round": sorted(game.players_in_round),
        "last_dice": list(game.last_dice) if game.last_dice else None,
        "immunity_active": game.roll_number <= config.IMMUNITY_ROLLS,
        "rolls_since_immunity_ended": game.rolls_since_immunity_ended,
        "game_over": game.game_over,
        "round_over": game.round_over,
        "waiting_for_decisions": game.waiting_for_decisions,
        "winners": game.winners() if game.game_over else [],
        "human_seats": sorted(human_seats),
        "ai_seats": sorted(set(range(game.num_players)) - human_seats),
        "revealed_actions": reveal_actions,
    }


# ============================================================================
# Request handler
# ============================================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logs (BaseHTTPRequestHandler is very chatty)
        if "favicon" in (args[0] if args else ""):
            return
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    # ------------------------------------------------------------ helpers

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Disable caching so the dashboard always sees fresh data
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        ctype = self._guess_ctype(path.suffix)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _guess_ctype(self, ext: str) -> str:
        return {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }.get(ext.lower(), "application/octet-stream")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------ GET

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API
        if path == "/api/status":
            self._send_json(self._build_status())
            return

        if path == "/api/metrics":
            self._send_json(self._read_metrics())
            return

        if path == "/api/checkpoints":
            self._send_json(self._list_checkpoints())
            return

        if path == "/api/log":
            self._send_json(self._read_log_tail())
            return

        if path == "/api/saves":
            self._send_json(self._list_saves())
            return

        if path == "/api/play/state":
            self._send_json(self._play_state())
            return

        if path == "/api/dice_stats":
            self._send_json(self._dice_stats())
            return

        # Static files
        if path == "/" or path == "":
            self._send_file(DASHBOARD_DIR / "index.html")
            return

        # Restrict to files inside dashboard/ to avoid path traversal
        rel = path.lstrip("/")
        target = (DASHBOARD_DIR / rel).resolve()
        try:
            target.relative_to(DASHBOARD_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return
        self._send_file(target)

    # ------------------------------------------------------------ POST

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        if path == "/api/train/start":
            args = body.get("args", {})
            res = training.start(args)
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/train/stop":
            res = training.stop()
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/train/reset":
            res = self._reset_training(
                delete_models=body.get("delete_models", True),
                delete_metrics=body.get("delete_metrics", True),
            )
            self._send_json(res)
            return

        if path == "/api/save":
            res = self._save_session(body.get("name", "save"))
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/load":
            res = self._load_session(body.get("name", ""))
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/play/new":
            res = self._play_new(
                num_players=int(body.get("num_players", 4)),
                human_seats=list(body.get("human_seats", [0])),
                checkpoint=body.get("checkpoint", "models/latest.pt"),
                sims_per_move=int(body.get("sims_per_move", config.SIMS_PER_MOVE_PLAY_QUALITY)),
            )
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/play/move":
            res = self._play_move(body.get("actions", {}))
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/tournament":
            res = self._run_tournament(body)
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/build_ai/run":
            res = self._build_ai_run(body)
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        self.send_error(404)

    # ============================================================
    # Status / metrics / checkpoints
    # ============================================================

    def _build_status(self) -> dict:
        return {
            "training": training.status(),
            "metrics_exists": METRICS_PATH.exists(),
            "models_dir": str(MODELS_DIR),
            "checkpoints": self._list_checkpoints(),
        }

    def _read_metrics(self) -> dict:
        if not METRICS_PATH.exists():
            return {"status": "idle", "latest": {}, "history": {}}
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return {"status": "error", "latest": {}, "history": {}, "error": str(e)}

    def _list_checkpoints(self) -> list[dict]:
        if not MODELS_DIR.exists():
            return []
        entries = []
        for p in sorted(MODELS_DIR.glob("*.pt")):
            try:
                stat = p.stat()
                entries.append({
                    "name": p.name,
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                })
            except OSError:
                continue
        return entries

    def _read_log_tail(self, n_lines: int = 200) -> dict:
        if not TRAIN_LOG_PATH.exists():
            return {"lines": []}
        try:
            with open(TRAIN_LOG_PATH, errors="replace") as f:
                lines = f.readlines()
            return {"lines": lines[-n_lines:]}
        except OSError as e:
            return {"lines": [], "error": str(e)}

    def _list_saves(self) -> list[dict]:
        if not SAVES_DIR.exists():
            return []
        out = []
        for p in sorted(SAVES_DIR.iterdir()):
            if p.is_dir():
                out.append({"name": p.name, "modified": p.stat().st_mtime})
        return out

    def _reset_training(self, delete_models: bool, delete_metrics: bool) -> dict:
        if training.is_running():
            return {"ok": False, "error": "stop training first"}
        deleted = []
        if delete_models and MODELS_DIR.exists():
            for p in MODELS_DIR.glob("*.pt"):
                try:
                    p.unlink()
                    deleted.append(p.name)
                except OSError as e:
                    # OneDrive/Windows file locks: rename instead
                    try:
                        p.rename(p.with_suffix(p.suffix + f".deleted_{int(time.time())}"))
                        deleted.append(p.name + " (renamed)")
                    except OSError:
                        return {"ok": False, "error": f"could not delete {p.name}: {e}"}
        if delete_metrics and METRICS_PATH.exists():
            try:
                METRICS_PATH.unlink()
            except OSError:
                pass
        if TRAIN_LOG_PATH.exists():
            try:
                TRAIN_LOG_PATH.unlink()
            except OSError:
                pass
        return {"ok": True, "deleted": deleted}

    def _save_session(self, name: str) -> dict:
        if training.is_running():
            return {"ok": False, "error": "stop training first"}
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        target = SAVES_DIR / name
        if target.exists():
            return {"ok": False, "error": "save name already exists"}
        target.mkdir()
        # Copy models and metrics
        if MODELS_DIR.exists():
            (target / "models").mkdir()
            for p in MODELS_DIR.glob("*.pt"):
                shutil.copy2(p, target / "models" / p.name)
        if METRICS_PATH.exists():
            shutil.copy2(METRICS_PATH, target / "metrics.json")
        return {"ok": True, "name": name}

    def _load_session(self, name: str) -> dict:
        if training.is_running():
            return {"ok": False, "error": "stop training first"}
        src = SAVES_DIR / name
        if not src.exists() or not src.is_dir():
            return {"ok": False, "error": "save not found"}
        # Wipe current
        self._reset_training(delete_models=True, delete_metrics=True)
        if (src / "models").exists():
            MODELS_DIR.mkdir(exist_ok=True)
            for p in (src / "models").glob("*.pt"):
                shutil.copy2(p, MODELS_DIR / p.name)
        if (src / "metrics.json").exists():
            shutil.copy2(src / "metrics.json", METRICS_PATH)
        return {"ok": True, "name": name}

    # ============================================================
    # Play vs AI
    # ============================================================

    def _play_new(
        self,
        num_players: int,
        human_seats: list[int],
        checkpoint: str,
        sims_per_move: int,
    ) -> dict:
        from game import BankGame

        if not (2 <= num_players <= 6):
            return {"ok": False, "error": "num_players must be 2..6"}
        human_seats = [s for s in human_seats if 0 <= s < num_players]

        # Resolve checkpoint path (relative to project root if not absolute)
        ck_path: str | None = None
        if checkpoint:
            cp = Path(checkpoint)
            if not cp.is_absolute():
                cp = PROJECT_ROOT / cp
            if cp.exists():
                ck_path = str(cp)

        with play_session.lock:
            try:
                play_session.network = _load_network(ck_path)
            except Exception as e:
                return {"ok": False, "error": f"could not load checkpoint: {e}"}
            play_session.checkpoint_path = ck_path
            play_session.game = BankGame(num_players=num_players, seed=int(time.time()))
            play_session.game.reset()
            play_session.human_seats = set(human_seats)
            play_session.action_history = []
            play_session.sims_per_move = sims_per_move

        return {
            "ok": True,
            "checkpoint": ck_path or "(untrained)",
            "state": _serialize_game_state(play_session.game, play_session.human_seats),
        }

    def _play_state(self) -> dict:
        with play_session.lock:
            if play_session.game is None:
                return {"ok": False, "error": "no active game"}
            return {
                "ok": True,
                "checkpoint": play_session.checkpoint_path or "(untrained)",
                "state": _serialize_game_state(play_session.game, play_session.human_seats),
                "history": play_session.action_history,
            }

    def _play_move(self, human_actions: dict) -> dict:
        """Apply a step: human submits decisions for human seats; server-side MCTS picks AI seats."""
        from mcts import MCTS
        from game.environment import BANK, CONTINUE

        with play_session.lock:
            game = play_session.game
            net = play_session.network
            if game is None or net is None:
                return {"ok": False, "error": "no active game"}
            if game.is_terminal():
                return {"ok": False, "error": "game is over"}
            if not game.waiting_for_decisions:
                return {"ok": False, "error": "engine is not waiting"}

            active = game.players_in_round
            actions: dict[int, str] = {}

            # Human seat actions (must be supplied for any human seat that is in the round)
            for pid_str, val in (human_actions or {}).items():
                pid = int(pid_str)
                if pid not in active or pid not in play_session.human_seats:
                    continue
                if val not in (BANK, CONTINUE):
                    return {"ok": False, "error": f"invalid action {val!r} for seat {pid}"}
                actions[pid] = val

            # Verify every human seat in round is provided
            missing_humans = [
                pid for pid in active
                if pid in play_session.human_seats and pid not in actions
            ]
            if missing_humans:
                return {"ok": False, "error": f"missing human action for seats {missing_humans}"}

            # Run MCTS for each AI seat (independently — no peeking)
            import numpy as np
            for pid in sorted(active):
                if pid in play_session.human_seats:
                    continue
                mcts = MCTS(
                    network=net,
                    root_player=pid,
                    rng=np.random.default_rng(seed=int(time.time() * 1000) + pid),
                )
                visits = mcts.run(
                    game,
                    num_simulations=play_session.sims_per_move,
                    add_dirichlet_noise=False,
                )
                # Greedy choose
                action = max(visits, key=visits.get)
                actions[pid] = action

            # Validate: every active player has an action now
            if set(actions.keys()) != active:
                return {
                    "ok": False,
                    "error": f"action set mismatch (got {sorted(actions)}, expected {sorted(active)})",
                }

            # Snapshot decisions before applying (so we can show what each seat decided)
            revealed = dict(actions)

            game.step(actions)
            play_session.action_history.append({
                "round": revealed and game.round_number,  # the round the actions affected
                "actions": revealed,
                "post_state_dice": list(game.last_dice) if game.last_dice else None,
            })

            return {
                "ok": True,
                "state": _serialize_game_state(game, play_session.human_seats, reveal_actions=revealed),
                "history": play_session.action_history[-10:],
            }

    # ============================================================
    # Dice stats
    # ============================================================

    def _dice_stats(self) -> dict:
        """
        Probability tables for two-d6 outcomes.
        Returns sum probabilities, per-outcome probabilities, and special-event probabilities.
        """
        sums: dict[int, int] = {}
        doubles = 0
        sevens = 0
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                s = d1 + d2
                sums[s] = sums.get(s, 0) + 1
                if d1 == d2:
                    doubles += 1
                if s == 7:
                    sevens += 1
        total = 36
        return {
            "sums": [
                {"sum": s, "count": sums[s], "probability": sums[s] / total}
                for s in sorted(sums)
            ],
            "doubles_probability": doubles / total,
            "seven_probability": sevens / total,
            "bust_or_double_post_immunity_probability": (sevens + doubles) / total,
            "expected_pot_change_per_roll_post_immunity": _expected_pot_change(),
        }

    # ============================================================
    # Tournament
    # ============================================================

    def _run_tournament(self, body: dict) -> dict:
        """
        Quick tournament run. Body:
          {
            "num_players": 4,
            "games_per_matchup": 50,
            "competitors": [
              {"type": "checkpoint", "path": "models/latest.pt", "sims_per_move": 100, "name": "AI-latest"},
              {"type": "threshold", "threshold": 200, "name": "Thresh-200"},
              {"type": "random", "p_bank": 0.3, "name": "Rand-30"}
            ]
          }
        Returns wins per competitor, broken down per player count.
        """
        from training import (
            self_play_game, ThresholdBot, RandomBot,
            AlwaysBank, AlwaysContinue,
        )

        try:
            num_players = int(body.get("num_players", 4))
            games_per_matchup = int(body.get("games_per_matchup", 30))
            competitors_spec = body.get("competitors", [])
            if not competitors_spec:
                return {"ok": False, "error": "no competitors specified"}
        except Exception as e:
            return {"ok": False, "error": f"bad request: {e}"}

        # Materialize each competitor as either a (network, sims) pair or a Bot
        competitors = []
        for spec in competitors_spec:
            t = spec.get("type")
            name = spec.get("name") or t
            if t == "checkpoint":
                cp = spec.get("path")
                cp_path = (PROJECT_ROOT / cp) if cp and not Path(cp).is_absolute() else (Path(cp) if cp else None)
                try:
                    net = _load_network(str(cp_path) if cp_path else None)
                except Exception as e:
                    return {"ok": False, "error": f"could not load {cp}: {e}"}
                competitors.append({
                    "name": name,
                    "kind": "ai",
                    "network": net,
                    "sims_per_move": int(spec.get("sims_per_move", 100)),
                })
            elif t == "threshold":
                competitors.append({
                    "name": name, "kind": "bot",
                    "bot": ThresholdBot(int(spec.get("threshold", 200))),
                })
            elif t == "random":
                competitors.append({
                    "name": name, "kind": "bot",
                    "bot": RandomBot(float(spec.get("p_bank", 0.5))),
                })
            elif t == "always_bank":
                competitors.append({"name": name, "kind": "bot", "bot": AlwaysBank()})
            elif t == "always_continue":
                competitors.append({"name": name, "kind": "bot", "bot": AlwaysContinue()})
            else:
                return {"ok": False, "error": f"unknown competitor type: {t}"}

        # Run round-robin: every distinct unordered subset of competitors of size num_players
        # For simplicity: sample matchups by cycling competitors into seats
        results = self._tournament_run(competitors, num_players, games_per_matchup)
        return {"ok": True, **results}

    def _tournament_run(self, competitors, num_players, games_per_matchup):
        """Distribute games so each competitor plays a roughly equal number of seats."""
        from itertools import combinations
        from game import BankGame
        from game.environment import BANK, CONTINUE
        import numpy as np
        from mcts import MCTS

        n_comp = len(competitors)
        if n_comp < num_players:
            return {"error": f"need at least {num_players} competitors for {num_players}-player games"}

        win_counts = {c["name"]: 0 for c in competitors}
        games_played = {c["name"]: 0 for c in competitors}

        # Iterate every combination of competitors of size num_players
        for combo in combinations(range(n_comp), num_players):
            seats = [competitors[i] for i in combo]
            for g in range(games_per_matchup):
                game = BankGame(num_players=num_players, seed=hash((combo, g)) & 0x7FFFFFFF)
                game.reset()
                while not game.is_terminal():
                    if not game.waiting_for_decisions:
                        break
                    actions = {}
                    for pid in sorted(game.players_in_round):
                        seat = seats[pid]
                        if seat["kind"] == "bot":
                            actions[pid] = seat["bot"].choose(game.get_state(pid))
                        else:
                            mcts = MCTS(
                                network=seat["network"],
                                root_player=pid,
                                rng=np.random.default_rng(seed=g * 1000 + pid),
                            )
                            visits = mcts.run(
                                game,
                                num_simulations=seat["sims_per_move"],
                                add_dirichlet_noise=False,
                            )
                            actions[pid] = max(visits, key=visits.get)
                    game.step(actions)
                winners = game.winners()
                k = len(winners)
                share = 1.0 / k
                for pid in winners:
                    win_counts[seats[pid]["name"]] += share
                for s in seats:
                    games_played[s["name"]] += 1

        # Win rates
        leaderboard = []
        for c in competitors:
            n = games_played[c["name"]]
            wr = win_counts[c["name"]] / n if n > 0 else 0.0
            leaderboard.append({
                "name": c["name"],
                "games": n,
                "wins": round(win_counts[c["name"]], 2),
                "win_rate": wr,
            })
        leaderboard.sort(key=lambda x: -x["win_rate"])
        return {
            "leaderboard": leaderboard,
            "num_players": num_players,
            "games_per_matchup": games_per_matchup,
        }

    # ============================================================
    # Build AI runner (very minimal: evaluate a Python expression)
    # ============================================================

    def _build_ai_run(self, body: dict) -> dict:
        """
        Run a hand-coded strategy against a bot or AI in a quick eval.
        Body: {"code": "...", "num_players": 4, "num_games": 100, "opponent": {...}}
        The user code defines a function `decide(state)` returning 'bank' or 'continue'.
        """
        code = body.get("code", "")
        if not code.strip():
            return {"ok": False, "error": "empty code"}

        # Compile the user function in a restricted namespace
        ns: dict = {"__builtins__": {
            "abs": abs, "min": min, "max": max, "len": len,
            "int": int, "float": float, "bool": bool, "round": round,
            "sum": sum, "True": True, "False": False, "None": None,
        }}
        try:
            exec(code, ns)
        except Exception as e:
            return {"ok": False, "error": f"compile error: {type(e).__name__}: {e}"}
        decide = ns.get("decide")
        if not callable(decide):
            return {"ok": False, "error": "code must define decide(state)"}

        # Wrap as a bot
        from training import Bot
        from game.environment import BANK, CONTINUE

        class UserBot(Bot):
            name = "UserBot"
            def choose(self, state: dict) -> str:
                try:
                    out = decide(state)
                except Exception as e:
                    raise RuntimeError(f"decide() raised: {type(e).__name__}: {e}")
                if out not in (BANK, CONTINUE):
                    raise RuntimeError(f"decide() must return 'bank' or 'continue', got {out!r}")
                return out

        # Set up opponents (default: ThresholdBot(200))
        from training import ThresholdBot
        opp = body.get("opponent", {"type": "threshold", "threshold": 200})
        if opp.get("type") == "threshold":
            opp_bot = ThresholdBot(int(opp.get("threshold", 200)))
        else:
            opp_bot = ThresholdBot(200)

        num_players = int(body.get("num_players", 4))
        num_games = int(body.get("num_games", 100))

        # Run num_games where seat 0 is UserBot, others are opp_bot
        from game import BankGame
        wins = 0
        scores = []
        try:
            user_bot = UserBot()
            for g in range(num_games):
                game = BankGame(num_players=num_players, seed=g + 1000)
                game.reset()
                while not game.is_terminal():
                    if not game.waiting_for_decisions:
                        break
                    actions = {}
                    for pid in sorted(game.players_in_round):
                        if pid == 0:
                            actions[pid] = user_bot.choose(game.get_state(pid))
                        else:
                            actions[pid] = opp_bot.choose(game.get_state(pid))
                    game.step(actions)
                if 0 in game.winners():
                    wins += 1
                scores.append(game.banked_scores[0])
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        return {
            "ok": True,
            "wins": wins,
            "games": num_games,
            "win_rate": wins / num_games,
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }


# ============================================================================
# Helpers
# ============================================================================

def _expected_pot_change() -> dict:
    """Expected pot multiplier per post-immunity roll (assuming pot > 0)."""
    # P(7) = 6/36 -> pot becomes 0
    # P(doubles, not 7) = 6/36 -> pot doubles (×2)
    # P(other) = 24/36 -> pot += avg(other sums)
    # Expected sum across non-7-non-double outcomes:
    other_sum_total = 0
    other_count = 0
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            if d1 == d2:
                continue
            if d1 + d2 == 7:
                continue
            other_sum_total += d1 + d2
            other_count += 1
    avg_other = other_sum_total / other_count if other_count else 0
    return {
        "p_bust": 6 / 36,
        "p_double_pot": 6 / 36,
        "p_increment": 24 / 36,
        "avg_increment": avg_other,
    }


# ============================================================================
# main
# ============================================================================

def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Bank AI v2 dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
        # Best effort: stop child training process
        if training.is_running():
            training.stop()
        server.server_close()


if __name__ == "__main__":
    main()

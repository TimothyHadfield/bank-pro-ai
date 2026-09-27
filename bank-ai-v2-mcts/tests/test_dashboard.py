"""
Smoke tests for dashboard/server.py.

We start the server in a thread and hit each API endpoint.
"""

import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _start_server(port=15123):
    """Spin up the dashboard handler bound to a random port."""
    from dashboard import server as srv
    srv.PORT = port
    httpd = ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    # Give the server a moment
    time.sleep(0.3)
    return httpd, port


def _get(port, path):
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def _post(port, path, body):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # Read the JSON body even on 4xx/5xx so callers can inspect it
        return e.code, json.loads(e.read().decode())


# ---------------------------------------------------------------- tests

def test_static_index_served():
    httpd, port = _start_server(15123)
    try:
        url = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url, timeout=5) as r:
            body = r.read().decode()
        assert "Bank AI v2" in body, "index.html missing branding"
        print("  ok test_static_index_served")
    finally:
        httpd.shutdown()


def test_status_endpoint():
    httpd, port = _start_server(15124)
    try:
        s, payload = _get(port, "/api/status")
        assert s == 200
        assert "training" in payload and isinstance(payload["training"]["running"], bool)
        print("  ok test_status_endpoint")
    finally:
        httpd.shutdown()


def test_metrics_endpoint_when_no_file():
    httpd, port = _start_server(15125)
    try:
        s, payload = _get(port, "/api/metrics")
        assert s == 200
        assert "history" in payload
        print("  ok test_metrics_endpoint_when_no_file")
    finally:
        httpd.shutdown()


def test_dice_stats_endpoint():
    httpd, port = _start_server(15126)
    try:
        s, payload = _get(port, "/api/dice_stats")
        assert s == 200
        assert "sums" in payload
        assert len(payload["sums"]) == 11  # sums 2..12
        # 7 should be the most likely sum
        max_p = max(s["probability"] for s in payload["sums"])
        seven_p = next(x["probability"] for x in payload["sums"] if x["sum"] == 7)
        assert seven_p == max_p
        assert abs(payload["seven_probability"] - 6/36) < 1e-6
        assert abs(payload["doubles_probability"] - 6/36) < 1e-6
        print("  ok test_dice_stats_endpoint")
    finally:
        httpd.shutdown()


def test_play_vs_ai_full_round():
    """Start a 4-player game, take a few human turns, verify the engine works through the API."""
    httpd, port = _start_server(15127)
    try:
        # Use no checkpoint -> server will start with random network
        s, r = _post(port, "/api/play/new", {
            "num_players": 3,
            "human_seats": [0],
            "checkpoint": "",         # untrained
            "sims_per_move": 5,       # tiny for speed
        })
        assert s == 200 and r["ok"], f"play/new failed: {r}"
        state = r["state"]
        assert state["num_players"] == 3
        assert state["waiting_for_decisions"]

        # Play a few rounds
        n_steps = 0
        max_steps = 60
        while not state["game_over"] and n_steps < max_steps:
            actions = {}
            if 0 in state["players_in_round"]:
                # Simple strategy: bank if pot >= 200, else continue
                actions["0"] = "bank" if state["pot"] >= 200 else "continue"
            s, r = _post(port, "/api/play/move", {"actions": actions})
            assert s == 200 and r["ok"], f"play/move failed: {r}"
            state = r["state"]
            n_steps += 1
        # Either game ended or we ran out of steps
        print(f"  ok test_play_vs_ai_full_round (game_over={state['game_over']} after {n_steps} steps)")
    finally:
        httpd.shutdown()


def test_build_ai_run():
    httpd, port = _start_server(15128)
    try:
        code = """
def decide(state):
    return "bank" if state["pot"] >= 150 else "continue"
"""
        s, r = _post(port, "/api/build_ai/run", {
            "code": code,
            "num_players": 3,
            "num_games": 20,
            "opponent": {"type": "threshold", "threshold": 200},
        })
        assert s == 200 and r["ok"], f"build_ai/run failed: {r}"
        assert "win_rate" in r
        assert 0.0 <= r["win_rate"] <= 1.0
        print(f"  ok test_build_ai_run (win_rate={r['win_rate']:.2f})")
    finally:
        httpd.shutdown()


def test_build_ai_handles_bad_code():
    httpd, port = _start_server(15129)
    try:
        bad = "def decide(state):\n    return 'banzai'"
        s, r = _post(port, "/api/build_ai/run", {"code": bad, "num_games": 5})
        # Server returns 400 with ok=False and an error message
        assert not r.get("ok", True), f"expected error, got {r}"
        assert "error" in r and "banzai" in r["error"]
        print("  ok test_build_ai_handles_bad_code")
    finally:
        httpd.shutdown()


def test_tournament_endpoint():
    httpd, port = _start_server(15130)
    try:
        s, r = _post(port, "/api/tournament", {
            "num_players": 3,
            "games_per_matchup": 5,
            "competitors": [
                {"type": "threshold", "threshold": 150, "name": "T-150"},
                {"type": "threshold", "threshold": 250, "name": "T-250"},
                {"type": "random", "p_bank": 0.5, "name": "Rand"},
            ],
        })
        assert s == 200 and r["ok"], f"tournament failed: {r}"
        assert "leaderboard" in r and len(r["leaderboard"]) == 3
        print(f"  ok test_tournament_endpoint (top={r['leaderboard'][0]['name']})")
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_static_index_served,
    test_status_endpoint,
    test_metrics_endpoint_when_no_file,
    test_dice_stats_endpoint,
    test_play_vs_ai_full_round,
    test_build_ai_run,
    test_build_ai_handles_bad_code,
    test_tournament_endpoint,
]


def main():
    failures = []
    for t in ALL_TESTS:
        print(f"running {t.__name__}")
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((t.__name__, e))
            print(f"  FAIL: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(ALL_TESTS)} FAILED")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()

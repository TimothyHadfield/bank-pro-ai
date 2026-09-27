"""
Phase 7b Test: Verify the web app API works end-to-end.
"""
import sys
import os
import json
import threading
import time

sys.path.insert(0, ".")

# Start the Flask app in a background thread
os.environ["FLASK_ENV"] = "testing"
from web.app import app, _load_network

_load_network(os.path.join("models", "checkpoint_0.pt"))

client = app.test_client()


def test_index_page():
    print("=" * 60)
    print("TEST 1: Index page loads")
    print("=" * 60)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "BANK" in html
    assert "Start Game" in html
    print("  Status: 200, page contains expected content")
    print("PASSED\n")


def test_new_game():
    print("=" * 60)
    print("TEST 2: New game API")
    print("=" * 60)
    resp = client.post("/api/new_game", json={
        "num_ai": 3,
        "checkpoint": "checkpoint_0.pt",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "game_id" in data
    assert data["round_number"] == 1
    assert data["roll_number"] == 1
    assert data["pot"] > 0
    assert data["waiting_for_decision"] == True
    assert len(data["ai_scores"]) == 3
    print(f"  Game ID: {data['game_id']}")
    print(f"  Initial state: pot={data['pot']}, dice={data['dice']}")
    print("PASSED\n")
    return data["game_id"]


def test_game_flow(game_id):
    print("=" * 60)
    print("TEST 3: Full game flow via API")
    print("=" * 60)

    steps = 0
    game_over = False

    while not game_over and steps < 200:
        # Alternate bank/continue
        action = "continue" if steps % 3 != 0 else "bank"
        resp = client.post("/api/action", json={
            "game_id": game_id,
            "action": action,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        steps += 1

        if data.get("game_over"):
            game_over = True
            print(f"  Game ended after {steps} steps")
            print(f"  Final scores: {data.get('final_scores')}")
            print(f"  Human won: {data.get('human_won')}")

        # If we're no longer waiting for a decision and game isn't over,
        # we need to check if the round auto-advanced
        if not data.get("waiting_for_decision") and not game_over:
            # The round ended (7 bust or we banked) — next call will be in new round
            pass

    assert game_over, f"Game should have ended within 200 steps, got {steps}"
    print("PASSED\n")


def test_checkpoints_api():
    print("=" * 60)
    print("TEST 4: Checkpoints API")
    print("=" * 60)
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "checkpoints" in data
    print(f"  Available checkpoints: {data['checkpoints']}")
    print("PASSED\n")


if __name__ == "__main__":
    test_index_page()
    game_id = test_new_game()
    test_game_flow(game_id)
    test_checkpoints_api()

    print("=" * 60)
    print("ALL WEB APP TESTS PASSED!")
    print("=" * 60)

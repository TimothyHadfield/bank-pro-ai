"""
Web-based UI for playing Bank against trained AI opponents.

Usage:
    python web/app.py
    python web/app.py --checkpoint models/checkpoint_1000.pt --num_ai 3 --port 5000

Completely standalone — does not affect training or project structure.
"""
import os
import sys
import json
import argparse
import glob

# Add parent directory to path so we can import game/network/config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from flask import Flask, render_template, jsonify, request, session

from config import Config
from game.environment import BankGame
from network.model import BankNetwork

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global state
_network = None
_config = Config()
_games = {}  # session_id -> game state dict


def _load_network(checkpoint_path):
    """Load a network from a checkpoint file."""
    global _network
    state_size = BankGame.state_size()
    _network = BankNetwork(state_size=state_size, hidden_size=_config.hidden_size)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        _network.load_state_dict(ckpt["network_state_dict"])
        _network.eval()
        return ckpt.get("batch_num", "?")
    _network.eval()
    return 0


def _get_game_state(game_id):
    """Get or create game state for a session."""
    return _games.get(game_id)


def _serialize_game(game, num_ai, extra=None):
    """Convert game state to a JSON-friendly dict for the frontend."""
    human_state = game.get_state(0)
    ai_scores = []
    for i in range(1, 1 + num_ai):
        ai_scores.append({
            "id": i,
            "score": int(game.banked_scores[i]),
            "in_round": i in game.players_in_round,
        })

    result = {
        "pot": int(game.pot),
        "roll_number": game.roll_number,
        "round_number": game.round_number if not game.game_over else game.num_rounds,
        "immunity_active": game.roll_number <= 3 if not game.game_over else False,
        "dice": list(game.last_dice) if game.last_dice else [0, 0],
        "dice_sum": sum(game.last_dice) if game.last_dice else 0,
        "is_doubles": game.last_dice[0] == game.last_dice[1] if game.last_dice else False,
        "is_seven": sum(game.last_dice) == 7 if game.last_dice else False,
        "human_score": int(game.banked_scores[0]),
        "human_in_round": 0 in game.players_in_round,
        "ai_scores": ai_scores,
        "players_still_in": len(game.players_in_round),
        "game_over": game.game_over,
        "round_over": game.round_over,
        "waiting_for_decision": game.waiting_for_decisions and 0 in game.players_in_round,
        "num_rounds": game.num_rounds,
    }

    if game.game_over:
        winners = game.get_winners()
        result["winners"] = winners
        result["human_won"] = 0 in winners
        result["final_scores"] = {
            "You": int(game.banked_scores[0]),
        }
        for i in range(1, 1 + num_ai):
            result["final_scores"][f"AI {i}"] = int(game.banked_scores[i])

    if extra:
        result.update(extra)

    return result


@app.route("/")
def index():
    """Serve the main game page."""
    # Find available checkpoints
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    checkpoints = []
    if os.path.exists(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
            checkpoints.append(os.path.basename(f))
    return render_template("game.html", checkpoints=checkpoints)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Start a new game."""
    data = request.get_json()
    num_ai = int(data.get("num_ai", 3))
    checkpoint = data.get("checkpoint", None)

    if not 1 <= num_ai <= 5:
        return jsonify({"error": "num_ai must be between 1 and 5"}), 400

    # Load checkpoint if specified
    if checkpoint:
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", checkpoint)
        batch_num = _load_network(ckpt_path)
    else:
        batch_num = _load_network(None)

    num_players = 1 + num_ai
    game = BankGame(num_players=num_players, num_rounds=_config.num_rounds)
    game.reset()

    game_id = os.urandom(8).hex()
    _games[game_id] = {
        "game": game,
        "num_ai": num_ai,
        "history": [],
    }

    state = _serialize_game(game, num_ai)
    state["game_id"] = game_id
    state["checkpoint_batch"] = batch_num
    state["message"] = f"Game started! You vs {num_ai} AI opponent(s). Round 1 begins."
    return jsonify(state)


@app.route("/api/action", methods=["POST"])
def player_action():
    """Process the human player's bank/continue decision."""
    data = request.get_json()
    game_id = data.get("game_id")
    human_action = data.get("action")  # "bank" or "continue"

    if game_id not in _games:
        return jsonify({"error": "Game not found. Start a new game."}), 404

    gs = _games[game_id]
    game = gs["game"]
    num_ai = gs["num_ai"]

    if game.game_over:
        return jsonify(_serialize_game(game, num_ai, {"message": "Game is already over."}))

    if not game.waiting_for_decisions:
        return jsonify(_serialize_game(game, num_ai, {"message": "Not waiting for decisions."}))

    if human_action not in ("bank", "continue"):
        return jsonify({"error": "Action must be 'bank' or 'continue'"}), 400

    # Collect all actions
    actions = {}
    ai_decisions = {}

    for pid in game.players_in_round:
        if pid == 0:
            actions[pid] = human_action
        else:
            # AI decision
            state_vec = game.get_state_vector(pid)
            with torch.no_grad():
                action_idx, _, _ = _network.select_action(state_vec, deterministic=True)
            ai_action = "bank" if action_idx == 0 else "continue"
            actions[pid] = ai_action
            ai_decisions[pid] = ai_action

    pot_before = game.pot
    states, round_done, game_done, info = game.step(actions)

    # Build message
    messages = []
    for pid, dec in ai_decisions.items():
        messages.append(f"AI {pid} chose to {dec.upper()}")

    if info["banked_players"]:
        for pid in info["banked_players"]:
            label = "You" if pid == 0 else f"AI {pid}"
            messages.append(f"{label} banked {pot_before} points!")

    if info.get("round_ended_by_seven"):
        messages.append("A 7 was rolled after immunity! Pot wiped! Round over!")

    if round_done and not game_done:
        messages.append(f"Round ended. Starting round {game.round_number}.")

    # Record history
    gs["history"].append({
        "human_action": human_action,
        "ai_decisions": ai_decisions,
        "pot_before": pot_before,
        "banked": info["banked_players"],
        "seven_bust": info.get("round_ended_by_seven", False),
    })

    # If human already banked but round continues, auto-advance until
    # either the round ends or human is back in a new round
    auto_messages = []
    while (not game.game_over
           and game.waiting_for_decisions
           and 0 not in game.players_in_round):
        # Only AI players remain — auto-play their decisions
        actions = {}
        ai_decs = {}
        for pid in game.players_in_round:
            state_vec = game.get_state_vector(pid)
            with torch.no_grad():
                action_idx, _, _ = _network.select_action(state_vec, deterministic=True)
            ai_action = "bank" if action_idx == 0 else "continue"
            actions[pid] = ai_action
            ai_decs[pid] = ai_action

        pot_b = game.pot
        states, rd, gd, inf = game.step(actions)

        for pid, dec in ai_decs.items():
            auto_messages.append(f"AI {pid}: {dec.upper()}")
        if inf["banked_players"]:
            for pid in inf["banked_players"]:
                auto_messages.append(f"AI {pid} banked {pot_b}")
        if inf.get("round_ended_by_seven"):
            auto_messages.append("7 rolled! Pot wiped!")
        if rd:
            auto_messages.append("Round ended.")

    if auto_messages:
        messages.append("--- While you were out ---")
        messages.extend(auto_messages)

    state = _serialize_game(game, num_ai)
    state["game_id"] = game_id
    state["message"] = "\n".join(messages)
    state["ai_decisions"] = {str(k): v for k, v in ai_decisions.items()}
    return jsonify(state)


@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    """List available model checkpoints."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    checkpoints = []
    if os.path.exists(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
            checkpoints.append(os.path.basename(f))
    return jsonify({"checkpoints": checkpoints})


def main():
    parser = argparse.ArgumentParser(description="Bank AI Web Interface")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Default checkpoint to load")
    parser.add_argument("--num_ai", type=int, default=3,
                        help="Default number of AI opponents")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    if args.checkpoint:
        _load_network(args.checkpoint)
    else:
        # Load any available checkpoint, or initialize randomly
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        pts = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
        if pts:
            _load_network(pts[-1])
            print(f"Auto-loaded checkpoint: {pts[-1]}")
        else:
            _load_network(None)
            print("No checkpoint found — using randomly initialized network.")

    print(f"\n  Bank AI Web Interface")
    print(f"  Open your browser to: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

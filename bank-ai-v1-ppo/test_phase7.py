"""
Phase 7 Test: Verify play_vs_ai loads and runs without crashing.
We can't test the full interactive loop, but we test the components.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from config import Config
from game.environment import BankGame
from network.model import BankNetwork


def test_play_vs_ai_components():
    print("=" * 60)
    print("TEST: play_vs_ai components")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    network.eval()

    # Save a test checkpoint
    os.makedirs(config.model_dir, exist_ok=True)
    path = os.path.join(config.model_dir, "checkpoint_play_test.pt")
    torch.save({
        "network_state_dict": network.state_dict(),
        "batch_num": 0,
    }, path)

    # Load it back
    checkpoint = torch.load(path, weights_only=False)
    network2 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    network2.load_state_dict(checkpoint["network_state_dict"])
    network2.eval()

    # Simulate a full game with AI making all decisions (including "human" player)
    num_ai = 3
    num_players = 1 + num_ai
    game = BankGame(num_players=num_players, num_rounds=config.num_rounds)
    game.reset()

    steps = 0
    while not game.game_over:
        if not game.waiting_for_decisions:
            break

        actions = {}
        for pid in game.players_in_round:
            state_vec = game.get_state_vector(pid)
            with torch.no_grad():
                action, _, _ = network2.select_action(state_vec, deterministic=True)
            actions[pid] = "bank" if action == 0 else "continue"

        game.step(actions)
        steps += 1

    print(f"  Simulated game completed in {steps} steps")
    print(f"  Final scores: {game.get_final_scores()}")
    print(f"  Winners: {game.get_winners()}")

    os.remove(path)
    print("PASSED: play_vs_ai components work correctly.\n")


if __name__ == "__main__":
    test_play_vs_ai_components()
    print("=" * 60)
    print("ALL PHASE 7 TESTS PASSED!")
    print("=" * 60)

"""
Pit two checkpoints against each other over N games.

Usage:
    python -m evaluation.ai_vs_ai --model1 models/checkpoint_1000.pt --model2 models/checkpoint_5000.pt --games 10000
"""
import argparse
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config
from game.environment import BankGame
from network.model import BankNetwork


def run_match(network1, network2, config: Config, num_games: int = 10000):
    model1_wins = 0
    model2_wins = 0
    ties = 0
    model1_scores = []
    model2_scores = []

    for game_idx in range(num_games):
        swap = (game_idx % 2 == 1)
        game = BankGame(num_players=config.num_players, num_rounds=config.num_rounds)
        game.reset()

        while not game.game_over:
            if not game.waiting_for_decisions:
                break

            actions = {}
            for pid in game.players_in_round:
                is_model1 = (pid % 2 == 0) if not swap else (pid % 2 == 1)
                net = network1 if is_model1 else network2

                state_vec = game.get_state_vector(pid)
                with torch.no_grad():
                    action, _, _ = net.select_action(state_vec, deterministic=False)
                actions[pid] = "bank" if action == 0 else "continue"

            game.step(actions)

        scores = game.get_final_scores()
        winners = game.get_winners()

        m1_players = [pid for pid in range(config.num_players)
                      if ((pid % 2 == 0) if not swap else (pid % 2 == 1))]
        m2_players = [pid for pid in range(config.num_players)
                      if pid not in m1_players]

        m1_avg = np.mean([scores[pid] for pid in m1_players])
        m2_avg = np.mean([scores[pid] for pid in m2_players])

        model1_scores.append(float(m1_avg))
        model2_scores.append(float(m2_avg))

        m1_won = any(pid in winners for pid in m1_players)
        m2_won = any(pid in winners for pid in m2_players)

        if m1_won and m2_won:
            ties += 1
        elif m1_won:
            model1_wins += 1
        elif m2_won:
            model2_wins += 1
        else:
            ties += 1

    return {
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "ties": ties,
        "model1_win_rate": model1_wins / num_games,
        "model2_win_rate": model2_wins / num_games,
        "model1_avg_score": float(np.mean(model1_scores)),
        "model2_avg_score": float(np.mean(model2_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="AI vs AI match")
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--games", type=int, default=10000)
    args = parser.parse_args()

    config = Config()
    state_size = BankGame.state_size()

    network1 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    ckpt1 = torch.load(args.model1, weights_only=False)
    network1.load_state_dict(ckpt1["network_state_dict"])
    network1.eval()

    network2 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    ckpt2 = torch.load(args.model2, weights_only=False)
    network2.load_state_dict(ckpt2["network_state_dict"])
    network2.eval()

    print(f"Model 1: {args.model1} (batch {ckpt1.get('batch_num', '?')})")
    print(f"Model 2: {args.model2} (batch {ckpt2.get('batch_num', '?')})")
    print(f"Playing {args.games} games with {config.num_players} players...\n")

    results = run_match(network1, network2, config, num_games=args.games)

    print(f"Results:")
    print(f"  Model 1 wins: {results['model1_wins']} ({results['model1_win_rate']:.1%})")
    print(f"  Model 2 wins: {results['model2_wins']} ({results['model2_win_rate']:.1%})")
    print(f"  Ties:         {results['ties']}")
    print(f"  Model 1 avg score: {results['model1_avg_score']:.1f}")
    print(f"  Model 2 avg score: {results['model2_avg_score']:.1f}")


if __name__ == "__main__":
    main()

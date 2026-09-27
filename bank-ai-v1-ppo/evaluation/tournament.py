"""
Round-robin tournament: pit multiple strategies against each other.

Includes built-in baselines and supports loading checkpoints or external AIs.

Usage:
    python -m evaluation.tournament
    python -m evaluation.tournament --checkpoint models/checkpoint_1000.pt --games 1000
"""
import argparse
import sys
import os
import torch
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config
from game.environment import BankGame
from network.model import BankNetwork


# ============================================================
# Strategy Interface
# ============================================================

class Strategy:
    """Base class for tournament competitors. Subclass and implement decide()."""

    def __init__(self, name: str):
        self.name = name

    def decide(self, state: dict) -> str:
        """
        Given a game state dict, return 'bank' or 'continue'.

        State dict keys:
            current_pot, roll_number, round_number, my_banked_score,
            opponent_banked_scores, players_still_in, immunity_active
        """
        raise NotImplementedError


# ============================================================
# Built-in Baseline Strategies
# ============================================================

class AlwaysContinue(Strategy):
    """Never banks — pure recklessness."""

    def __init__(self):
        super().__init__("AlwaysContinue")

    def decide(self, state: dict) -> str:
        return "continue"


class ThresholdBot(Strategy):
    """Banks when the pot exceeds a given threshold."""

    def __init__(self, threshold: int):
        super().__init__(f"Threshold({threshold})")
        self.threshold = threshold

    def decide(self, state: dict) -> str:
        if state["current_pot"] > self.threshold:
            return "bank"
        return "continue"


class RandomBot(Strategy):
    """Banks with a fixed probability each decision."""

    def __init__(self, p: float):
        super().__init__(f"Random({p:.2f})")
        self.p = p

    def decide(self, state: dict) -> str:
        if np.random.random() < self.p:
            return "bank"
        return "continue"


class NetworkStrategy(Strategy):
    """Wraps a trained BankNetwork as a tournament competitor."""

    def __init__(self, network, name: str = "NeuralNet"):
        super().__init__(name)
        self.network = network
        self.network.eval()

    def decide(self, state: dict) -> str:
        # Build state vector matching environment.get_state_vector layout (22 inputs)
        s = state
        vec = [
            s["current_pot"] / 1000.0,
            s["roll_number"] / 10.0,
            s["round_number"] / 10.0,
            s["my_banked_score"] / 1000.0,
            s["players_still_in"] / 6.0,
            s.get("rounds_remaining", 10 - s["round_number"]) / 10.0,
            1.0 if s.get("is_final_round", s["round_number"] == 10) else 0.0,
            s.get("rolls_since_immunity", max(0, s["roll_number"] - 3)) / 10.0,
            s.get("my_rank", 1) / s.get("num_players", 4),
            s.get("gap_to_first", 0) / 1000.0,
            s.get("gap_to_last", 0) / 1000.0,
            1.0 if s.get("would_banking_win", False) else 0.0,
        ]
        opp = s["opponent_banked_scores"]
        for i in range(5):
            vec.append(opp[i] / 1000.0 if i < len(opp) else 0.0)
        opp_in = s.get("opponent_in_round", [])
        for i in range(5):
            vec.append(1.0 if (i < len(opp_in) and opp_in[i]) else 0.0)

        state_vec = np.array(vec, dtype=np.float32)
        with torch.no_grad():
            action, _, _ = self.network.select_action(state_vec, deterministic=False)
        return "bank" if action == 0 else "continue"


def load_checkpoint_strategy(checkpoint_path: str, config: Config, name: str = None) -> NetworkStrategy:
    """Load a checkpoint and wrap it as a Strategy."""
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    network.load_state_dict(ckpt["network_state_dict"])
    batch = ckpt.get("batch_num", "?")
    if name is None:
        name = f"NN(batch={batch})"
    return NetworkStrategy(network, name=name)


# ============================================================
# Tournament Runner
# ============================================================

def play_match(strategy_a: Strategy, strategy_b: Strategy, config: Config, num_games: int):
    """
    Play num_games between two strategies. In each game, players alternate
    being controlled by strategy_a or strategy_b.

    Returns (a_wins, b_wins, ties, a_avg_score, b_avg_score)
    """
    a_wins = 0
    b_wins = 0
    ties = 0
    a_scores = []
    b_scores = []

    for game_idx in range(num_games):
        swap = (game_idx % 2 == 1)
        game = BankGame(num_players=config.num_players, num_rounds=config.num_rounds)
        game.reset()

        while not game.game_over:
            if not game.waiting_for_decisions:
                break

            actions = {}
            for pid in game.players_in_round:
                is_a = (pid % 2 == 0) if not swap else (pid % 2 == 1)
                strategy = strategy_a if is_a else strategy_b
                state = game.get_state(pid)
                actions[pid] = strategy.decide(state)

            game.step(actions)

        scores = game.get_final_scores()
        winners = game.get_winners()

        a_players = [pid for pid in range(config.num_players)
                     if ((pid % 2 == 0) if not swap else (pid % 2 == 1))]
        b_players = [pid for pid in range(config.num_players) if pid not in a_players]

        a_scores.append(float(np.mean([scores[pid] for pid in a_players])))
        b_scores.append(float(np.mean([scores[pid] for pid in b_players])))

        a_won = any(pid in winners for pid in a_players)
        b_won = any(pid in winners for pid in b_players)

        if a_won and b_won:
            ties += 1
        elif a_won:
            a_wins += 1
        elif b_won:
            b_wins += 1
        else:
            ties += 1

    return a_wins, b_wins, ties, float(np.mean(a_scores)), float(np.mean(b_scores))


def run_tournament(strategies, config: Config, num_games: int = 1000):
    """
    Run a round-robin tournament. Every strategy plays against every other.

    Returns a leaderboard sorted by win rate.
    """
    n = len(strategies)
    # Track total wins and games for each strategy
    total_wins = {s.name: 0 for s in strategies}
    total_games = {s.name: 0 for s in strategies}
    total_scores = {s.name: [] for s in strategies}
    matchup_results = []

    print(f"Tournament: {n} competitors, {num_games} games per matchup")
    print(f"Total matchups: {n * (n - 1) // 2}\n")

    for i, j in combinations(range(n), 2):
        sa, sb = strategies[i], strategies[j]
        print(f"  {sa.name} vs {sb.name}...", end=" ", flush=True)

        a_wins, b_wins, ties, a_avg, b_avg = play_match(sa, sb, config, num_games)

        print(f"{sa.name}: {a_wins} wins ({a_wins/num_games:.1%}), "
              f"{sb.name}: {b_wins} wins ({b_wins/num_games:.1%}), "
              f"ties: {ties}")

        total_wins[sa.name] += a_wins
        total_wins[sb.name] += b_wins
        total_games[sa.name] += num_games
        total_games[sb.name] += num_games
        total_scores[sa.name].append(a_avg)
        total_scores[sb.name].append(b_avg)

        matchup_results.append({
            "a": sa.name, "b": sb.name,
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "a_avg_score": a_avg, "b_avg_score": b_avg,
        })

    # Build leaderboard
    leaderboard = []
    for s in strategies:
        games = total_games[s.name]
        wins = total_wins[s.name]
        avg_score = float(np.mean(total_scores[s.name])) if total_scores[s.name] else 0
        leaderboard.append({
            "name": s.name,
            "wins": wins,
            "games": games,
            "win_rate": wins / games if games > 0 else 0,
            "avg_score": avg_score,
        })

    leaderboard.sort(key=lambda x: x["win_rate"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  LEADERBOARD")
    print(f"{'='*60}")
    print(f"  {'Rank':<6}{'Name':<25}{'Win Rate':<12}{'Avg Score':<12}")
    print(f"  {'-'*55}")
    for rank, entry in enumerate(leaderboard, 1):
        print(f"  {rank:<6}{entry['name']:<25}{entry['win_rate']:.1%}{'':<4}{entry['avg_score']:.1f}")
    print()

    return leaderboard


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained checkpoint")
    parser.add_argument("--games", type=int, default=1000, help="Games per matchup")
    args = parser.parse_args()

    config = Config()

    strategies = [
        AlwaysContinue(),
        ThresholdBot(100),
        ThresholdBot(200),
        ThresholdBot(300),
        RandomBot(0.2),
        RandomBot(0.3),
    ]

    if args.checkpoint:
        nn_strategy = load_checkpoint_strategy(args.checkpoint, config)
        strategies.append(nn_strategy)

    run_tournament(strategies, config, num_games=args.games)


if __name__ == "__main__":
    main()

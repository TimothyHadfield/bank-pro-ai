"""
Evaluation module for Bank AI v2.

Provides:
  - play_match():           pit a network against a bot or another network in N games
  - evaluate_against_bots(): test current network vs a panel of threshold bots
  - evaluate_against_checkpoint(): current network vs a frozen older checkpoint
  - Evaluator class:        coordinates a per-player-count evaluation run

All evaluations break down win rates per player count (per the project goal of being
strong across 2-6 players).

No-peeking is preserved: each AI seat runs its own MCTS independently from the player's
own perspective. Bot seats use the public state dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from game import BankGame
from game.environment import BANK, CONTINUE
from mcts import MCTS
from network import BankNetwork
from training.opponent_pool import Bot, ThresholdBot
from training.checkpoint import load_checkpoint


# ---------------------------------------------------------------- types

@dataclass
class MatchResult:
    """Outcome of a single match (many games) between two competitors at one player count."""
    competitor_a: str
    competitor_b: str
    num_players: int
    games: int
    a_wins: float            # fractional (ties split)
    b_wins: float
    a_avg_score: float
    b_avg_score: float

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.games if self.games else 0.0

    @property
    def b_win_rate(self) -> float:
        return self.b_wins / self.games if self.games else 0.0


@dataclass
class EvaluationResult:
    """Aggregated evaluation result across multiple opponents and player counts."""
    iteration: int
    timestamp: float
    # opponent_name -> player_count -> win rate (0..1)
    win_rates: dict[str, dict[int, float]] = field(default_factory=dict)
    # opponent_name -> {games, wins, avg_score, opp_avg_score} per player count
    details: dict[str, dict[int, dict]] = field(default_factory=dict)

    def overall_win_rate(self) -> float:
        """Mean win rate across all (opponent, player_count) buckets."""
        all_rates = []
        for op_rates in self.win_rates.values():
            all_rates.extend(op_rates.values())
        return float(np.mean(all_rates)) if all_rates else 0.0

    def to_dict(self) -> dict:
        # JSON-friendly form for the dashboard
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "win_rates": {
                k: {str(pc): rate for pc, rate in v.items()}
                for k, v in self.win_rates.items()
            },
            "details": {
                k: {str(pc): info for pc, info in v.items()}
                for k, v in self.details.items()
            },
            "overall_win_rate": self.overall_win_rate(),
        }


# ---------------------------------------------------------------- competitor abstraction

class _AICompetitor:
    """An AI seat backed by a network + MCTS."""
    def __init__(self, network: BankNetwork, name: str, sims_per_move: int):
        self.network = network
        self.name = name
        self.sims_per_move = sims_per_move

    def choose(self, game: BankGame, pid: int, rng: np.random.Generator) -> str:
        mcts = MCTS(network=self.network, root_player=pid, rng=rng)
        visits = mcts.run(game, num_simulations=self.sims_per_move, add_dirichlet_noise=False)
        return max(visits, key=visits.get)


class _BotCompetitor:
    def __init__(self, bot: Bot):
        self.bot = bot
        self.name = bot.name

    def choose(self, game: BankGame, pid: int, rng: np.random.Generator) -> str:
        return self.bot.choose(game.get_state(pid))


def _make_competitor(thing) -> "_AICompetitor | _BotCompetitor":
    if isinstance(thing, Bot):
        return _BotCompetitor(thing)
    if isinstance(thing, _AICompetitor) or isinstance(thing, _BotCompetitor):
        return thing
    raise TypeError(f"can't wrap competitor: {thing!r}")


# ---------------------------------------------------------------- core match

def play_match(
    competitor_a,
    competitor_b,
    num_players: int,
    games: int,
    base_seed: int = 0,
) -> MatchResult:
    """
    Play `games` games at `num_players` between competitor_a and competitor_b.
    Seats are split as evenly as possible: half a_seats, half b_seats. Roles swap each game
    (a takes the "first half" seats in even games, "second half" in odd) to control for
    seat advantage.
    """
    a = _make_competitor(competitor_a)
    b = _make_competitor(competitor_b)

    a_wins = 0.0
    b_wins = 0.0
    a_score_total = 0
    b_score_total = 0
    a_seats_total = 0
    b_seats_total = 0

    for g in range(games):
        # Alternate which competitor sits in which seats for fairness
        a_seats = set(range(0, num_players, 2)) if g % 2 == 0 else set(range(1, num_players, 2))
        b_seats = set(range(num_players)) - a_seats

        game = BankGame(num_players=num_players, seed=base_seed + g)
        game.reset()
        rngs = {pid: np.random.default_rng(base_seed * 1000 + g * 100 + pid) for pid in range(num_players)}

        while not game.is_terminal():
            if not game.waiting_for_decisions:
                break
            actions = {}
            for pid in sorted(game.players_in_round):
                comp = a if pid in a_seats else b
                actions[pid] = comp.choose(game, pid, rngs[pid])
            game.step(actions)

        winners = set(game.winners())
        k = len(winners)
        for pid in winners:
            share = 1.0 / k
            if pid in a_seats:
                a_wins += share
            else:
                b_wins += share

        for pid in range(num_players):
            score = game.banked_scores[pid]
            if pid in a_seats:
                a_score_total += score
                a_seats_total += 1
            else:
                b_score_total += score
                b_seats_total += 1

    return MatchResult(
        competitor_a=getattr(a, "name", "A"),
        competitor_b=getattr(b, "name", "B"),
        num_players=num_players,
        games=games,
        a_wins=a_wins,
        b_wins=b_wins,
        a_avg_score=a_score_total / max(1, a_seats_total),
        b_avg_score=b_score_total / max(1, b_seats_total),
    )


# ---------------------------------------------------------------- panels

def evaluate_against_bots(
    network: BankNetwork,
    sims_per_move: int = config.SIMS_PER_MOVE_EVAL,
    games_per_player_count: int = config.EVAL_GAMES_PER_OPPONENT,
    player_counts: list[int] = config.EVAL_PLAYER_COUNTS,
    bot_thresholds: list[int] = config.THRESHOLD_BOT_THRESHOLDS,
    base_seed: int = 0,
    iteration: int = 0,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate `network` against a panel of threshold bots, broken down per player count."""
    import time
    ai = _AICompetitor(network, name="AI-current", sims_per_move=sims_per_move)
    result = EvaluationResult(iteration=iteration, timestamp=time.time())

    total_panels = len(bot_thresholds) * len(player_counts)
    panel_idx = 0
    if verbose:
        total_games = total_panels * games_per_player_count
        print(
            f"[eval] starting bots panel: {len(bot_thresholds)} bots × {len(player_counts)} player counts "
            f"× {games_per_player_count} games = {total_games} games at sims={sims_per_move}",
            flush=True,
        )
    eval_start = time.perf_counter()

    for thresh in bot_thresholds:
        bot = ThresholdBot(thresh)
        op_name = bot.name
        result.win_rates[op_name] = {}
        result.details[op_name] = {}
        for pc in player_counts:
            panel_idx += 1
            panel_start = time.perf_counter()
            mr = play_match(ai, _BotCompetitor(bot), num_players=pc, games=games_per_player_count, base_seed=base_seed + thresh + pc * 100)
            if verbose:
                panel_t = time.perf_counter() - panel_start
                print(
                    f"[eval] panel {panel_idx}/{total_panels} {op_name} vs {pc}p: "
                    f"win_rate={mr.a_win_rate:.3f} ({panel_t:.1f}s)",
                    flush=True,
                )
            result.win_rates[op_name][pc] = mr.a_win_rate
            result.details[op_name][pc] = {
                "games": mr.games,
                "wins": mr.a_wins,
                "avg_score": mr.a_avg_score,
                "opp_avg_score": mr.b_avg_score,
            }
    if verbose:
        print(f"[eval] bots panel done in {time.perf_counter() - eval_start:.1f}s", flush=True)
    return result


def evaluate_against_checkpoint(
    current_network: BankNetwork,
    other_checkpoint_path: str | Path,
    sims_per_move: int = config.SIMS_PER_MOVE_EVAL,
    games_per_player_count: int = 50,
    player_counts: list[int] = config.EVAL_PLAYER_COUNTS,
    base_seed: int = 0,
    iteration: int = 0,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate the current network against a frozen older checkpoint.

    Returns an EvaluationResult where the only opponent is the named checkpoint.
    """
    import time
    other_net = BankNetwork()
    info = load_checkpoint(other_checkpoint_path, other_net)
    op_name = f"checkpoint_iter_{info.get('iteration', '?')}"

    ai = _AICompetitor(current_network, name="AI-current", sims_per_move=sims_per_move)
    other = _AICompetitor(other_net, name=op_name, sims_per_move=sims_per_move)

    if verbose:
        total_games = len(player_counts) * games_per_player_count
        print(
            f"[eval] starting checkpoint match vs {op_name}: {len(player_counts)} player counts "
            f"× {games_per_player_count} games = {total_games} games",
            flush=True,
        )

    result = EvaluationResult(iteration=iteration, timestamp=time.time())
    result.win_rates[op_name] = {}
    result.details[op_name] = {}
    eval_start = time.perf_counter()
    for pc in player_counts:
        panel_start = time.perf_counter()
        mr = play_match(ai, other, num_players=pc, games=games_per_player_count, base_seed=base_seed + pc * 100)
        if verbose:
            panel_t = time.perf_counter() - panel_start
            print(
                f"[eval] vs {op_name} {pc}p: win_rate={mr.a_win_rate:.3f} ({panel_t:.1f}s)",
                flush=True,
            )
        result.win_rates[op_name][pc] = mr.a_win_rate
        result.details[op_name][pc] = {
            "games": mr.games,
            "wins": mr.a_wins,
            "avg_score": mr.a_avg_score,
            "opp_avg_score": mr.b_avg_score,
        }
    if verbose:
        print(f"[eval] checkpoint match done in {time.perf_counter() - eval_start:.1f}s", flush=True)
    return result


# ---------------------------------------------------------------- coordinator

class Evaluator:
    """
    Coordinates periodic evaluation during training.

    Tracks the most recent EvaluationResult and provides a method to merge multiple
    panels (e.g. vs bots + vs checkpoint) into a single dict for the dashboard.
    """

    def __init__(self):
        self.last_result: EvaluationResult | None = None

    def run(
        self,
        network: BankNetwork,
        iteration: int,
        sims_per_move: int = config.SIMS_PER_MOVE_EVAL,
        games_per_player_count: int = config.EVAL_GAMES_PER_OPPONENT,
        player_counts: list[int] | None = None,
        bot_thresholds: list[int] | None = None,
        old_checkpoint_path: str | None = None,
        base_seed: int = 0,
    ) -> EvaluationResult:
        pc = player_counts or config.EVAL_PLAYER_COUNTS
        bots = bot_thresholds or config.THRESHOLD_BOT_THRESHOLDS

        # Evaluate vs bots
        result = evaluate_against_bots(
            network=network,
            sims_per_move=sims_per_move,
            games_per_player_count=games_per_player_count,
            player_counts=pc,
            bot_thresholds=bots,
            base_seed=base_seed,
            iteration=iteration,
            verbose=True,
        )

        # Evaluate vs old checkpoint (if available)
        if old_checkpoint_path:
            try:
                ck_result = evaluate_against_checkpoint(
                    current_network=network,
                    other_checkpoint_path=old_checkpoint_path,
                    sims_per_move=sims_per_move,
                    games_per_player_count=max(2, games_per_player_count // 2),
                    player_counts=pc,
                    base_seed=base_seed + 9999,
                    iteration=iteration,
                    verbose=True,
                )
                result.win_rates.update(ck_result.win_rates)
                result.details.update(ck_result.details)
            except Exception as e:
                # Don't crash training if the old checkpoint can't be loaded
                print(f"[evaluator] WARN: could not evaluate vs {old_checkpoint_path}: {e}")

        self.last_result = result
        return result

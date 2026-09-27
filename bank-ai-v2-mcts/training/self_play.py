"""
Self-play game generation for Bank AI v2.

One call to `self_play_game()` plays a single complete game where every player uses either
(a) the network with their own independent MCTS, or (b) a bot from the opponent pool.
Returns labeled training examples for the AI seats only (bot decisions are not recorded).

Slice 2 changes:
  - Mixed opponent pool: configurable fraction of games include 1 or 2 bot seats
  - Bots come from `opponent_pool.build_default_bot_pool` (or any custom list)

CRITICAL no-peeking guarantee:
  At every decision point, we run each player's MCTS independently using a CLONE of the
  game (so one player's MCTS can't accidentally observe another's). All players' chosen
  actions are then collected into a single dict and applied to the engine in ONE step()
  call, which processes them simultaneously.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

import config
from game import BankGame
from game.environment import BANK, CONTINUE
from mcts import MCTS
from network import BankNetwork
from training.opponent_pool import Bot, build_default_bot_pool


ACTIONS = (BANK, CONTINUE)

# Sentinel marking an "AI seat" (uses the current network + records examples)
AI_SEAT = "AI"


@dataclass
class FrozenNet:
    """Marks a seat that uses a frozen older network. Examples are NOT recorded for these seats."""
    network: BankNetwork
    name: str
    sims_per_move: int = config.SIMS_PER_MOVE


@dataclass
class TrainingExample:
    """One labeled training data point recorded during self-play."""
    state_vector: np.ndarray              # shape (INPUT_SIZE,)
    mcts_policy: np.ndarray               # shape (NUM_ACTIONS,) - normalized visit distribution
    player_id: int                        # who was deciding
    num_players: int                      # game's player count
    round_number: int                     # game round at decision time
    # Filled at game end:
    value: float = 0.0


@dataclass
class GameResult:
    """Summary of one self-play game (used by callers for monitoring)."""
    num_players: int
    seat_assignments: list                # element is AI_SEAT or a Bot instance
    winners: list[int]
    final_scores: list[int]
    num_decisions: int                    # total decisions across all AI seats
    num_bot_seats: int                    # how many seats were bots


# ---------------------------------------------------------------- helpers

def _sample_action_from_visits(
    visits: dict[str, int], temperature: float, rng: random.Random
) -> str:
    if temperature == 0:
        max_n = max(visits.values())
        candidates = [a for a, n in visits.items() if n == max_n]
        return rng.choice(candidates)
    raw = [visits[a] ** (1.0 / temperature) for a in ACTIONS]
    total = sum(raw)
    if total <= 0:
        return rng.choice(list(ACTIONS))
    probs = [r / total for r in raw]
    r = rng.random()
    cum = 0.0
    for action, p in zip(ACTIONS, probs):
        cum += p
        if r <= cum:
            return action
    return ACTIONS[-1]


def _visits_to_policy_vector(visits: dict[str, int]) -> np.ndarray:
    total = sum(visits.values())
    if total == 0:
        return np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    return np.array([visits[a] / total for a in ACTIONS], dtype=np.float32)


def _temperature_for(round_number: int) -> float:
    rounds_remaining = config.NUM_ROUNDS - round_number
    if rounds_remaining < config.LATE_GAME_THRESHOLD_ROUNDS_REMAINING:
        return config.TEMPERATURE_LATE_GAME
    return config.TEMPERATURE_INITIAL


def _build_seat_assignments(
    num_players: int,
    bot_pool: list[Bot] | None,
    frozen_nets: list[FrozenNet] | None,
    rng: random.Random,
) -> list:
    """
    Decide which seats are AI, bots, or frozen older networks based on the configured mix.

    Returns a list of length num_players where each element is AI_SEAT, a Bot, or a FrozenNet.

    Per PLAN.md §5.2:
      - PURE_SELF_PLAY_FRACTION: all seats AI
      - ONE_BOT_FRACTION: one seat is a random bot
      - TWO_BOT_FRACTION: two seats are bots
      - OLD_CHECKPOINT_MIX_FRACTION: one or more seats use a frozen older network
    """
    pool_available = bot_pool is not None and len(bot_pool) > 0
    frozen_available = frozen_nets is not None and len(frozen_nets) > 0

    if (not pool_available and not frozen_available) or num_players < 3:
        return [AI_SEAT] * num_players

    r = rng.random()
    cum = 0.0
    cum += config.PURE_SELF_PLAY_FRACTION
    if r < cum:
        return [AI_SEAT] * num_players
    cum += config.ONE_BOT_FRACTION
    if r < cum and pool_available:
        seats = [AI_SEAT] * num_players
        seats[rng.randrange(num_players)] = rng.choice(bot_pool)
        return seats
    cum += config.TWO_BOT_FRACTION
    if r < cum and pool_available and num_players >= 4:
        seats = [AI_SEAT] * num_players
        for i in rng.sample(range(num_players), 2):
            seats[i] = rng.choice(bot_pool)
        return seats
    # OLD_CHECKPOINT_MIX_FRACTION
    if frozen_available:
        seats = [AI_SEAT] * num_players
        # Replace 1 or 2 seats with frozen networks
        n_frozen = 2 if (num_players >= 4 and rng.random() < 0.5) else 1
        for i in rng.sample(range(num_players), n_frozen):
            seats[i] = rng.choice(frozen_nets)
        return seats
    # Fallback: pure self-play
    return [AI_SEAT] * num_players


# ---------------------------------------------------------------- main entry

def self_play_game(
    network: BankNetwork,
    num_players: int | None = None,
    sims_per_move: int = config.SIMS_PER_MOVE,
    add_dirichlet_noise: bool = True,
    seed: int | None = None,
    bot_pool: list[Bot] | None = None,
    frozen_nets: list[FrozenNet] | None = None,
    force_pure_self_play: bool = False,
) -> tuple[list[TrainingExample], GameResult]:
    """
    Play one full self-play game and return (training examples, game result summary).

    Args:
        network: shared network used by all AI seats
        num_players: if None, sampled uniformly from NUM_PLAYERS_RANGE
        sims_per_move: MCTS simulations per AI decision
        add_dirichlet_noise: True for self-play (exploration), False for evaluation
        seed: optional seed for reproducibility
        bot_pool: optional list of bots; if None, uses build_default_bot_pool()
        frozen_nets: optional list of frozen older networks for OLD_CHECKPOINT_MIX_FRACTION
        force_pure_self_play: if True, ignores bot_pool/frozen_nets and uses AI seats only
    """
    rng = random.Random(seed)
    np_rng_seed = rng.randint(0, 2**31 - 1)

    if num_players is None:
        lo, hi = config.NUM_PLAYERS_RANGE
        num_players = rng.randint(lo, hi)

    if force_pure_self_play:
        seat_assignments = [AI_SEAT] * num_players
    else:
        if bot_pool is None:
            bot_pool = build_default_bot_pool(seed=rng.randint(0, 2**31 - 1))
        seat_assignments = _build_seat_assignments(num_players, bot_pool, frozen_nets, rng)

    game = BankGame(num_players=num_players, seed=rng.randint(0, 2**31 - 1))
    game.reset()

    examples: list[TrainingExample] = []
    player_rngs = {
        pid: np.random.default_rng(seed=np_rng_seed + pid * 1000 + 1)
        for pid in range(num_players)
    }

    while not game.is_terminal():
        if not game.waiting_for_decisions:
            raise RuntimeError("Self-play hit a non-decision, non-terminal state")

        active = sorted(game.players_in_round)
        actions: dict[int, str] = {}

        for pid in active:
            seat = seat_assignments[pid]
            if seat is AI_SEAT:
                mcts = MCTS(
                    network=network,
                    root_player=pid,
                    rng=player_rngs[pid],
                )
                visits = mcts.run(
                    game,
                    num_simulations=sims_per_move,
                    add_dirichlet_noise=add_dirichlet_noise,
                )
                temp = _temperature_for(game.round_number)
                action = _sample_action_from_visits(visits, temp, rng)
                actions[pid] = action

                examples.append(
                    TrainingExample(
                        state_vector=game.get_state_vector(pid),
                        mcts_policy=_visits_to_policy_vector(visits),
                        player_id=pid,
                        num_players=num_players,
                        round_number=game.round_number,
                    )
                )
            elif isinstance(seat, FrozenNet):
                # Frozen-network seat: run MCTS using the frozen network. NO examples recorded.
                mcts = MCTS(
                    network=seat.network,
                    root_player=pid,
                    rng=player_rngs[pid],
                )
                visits = mcts.run(
                    game,
                    num_simulations=seat.sims_per_move,
                    add_dirichlet_noise=False,
                )
                actions[pid] = max(visits, key=visits.get)
            else:
                # Bot seat: ask the bot for its choice using the public state dict
                bot: Bot = seat
                action = bot.choose(game.get_state(pid))
                if action not in (BANK, CONTINUE):
                    raise ValueError(
                        f"Bot {bot.name} returned invalid action {action!r}"
                    )
                actions[pid] = action

        game.step(actions)

    winners = game.winners()
    k = len(winners)
    win_value = 1.0 / k if k > 0 else 0.0
    for ex in examples:
        ex.value = win_value if ex.player_id in winners else 0.0

    result = GameResult(
        num_players=num_players,
        seat_assignments=seat_assignments,
        winners=winners,
        final_scores=list(game.banked_scores),
        num_decisions=len(examples),
        num_bot_seats=sum(1 for s in seat_assignments if s is not AI_SEAT),
    )
    return examples, result


def summarize_examples(examples: list[TrainingExample]) -> dict:
    if not examples:
        return {"count": 0}
    by_player = {}
    for ex in examples:
        by_player.setdefault(ex.player_id, 0)
        by_player[ex.player_id] += 1
    n_winners = sum(1 for ex in examples if ex.value > 0)
    avg_value = sum(ex.value for ex in examples) / len(examples)
    return {
        "count": len(examples),
        "num_players": examples[0].num_players,
        "decisions_per_player": by_player,
        "winning_examples": n_winners,
        "avg_value": avg_value,
    }

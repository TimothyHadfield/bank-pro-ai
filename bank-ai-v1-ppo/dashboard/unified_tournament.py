"""
Unified tournament runner for the AI vs AI tab.

Lets v1 networks, v2 networks, and Build AI custom strategies compete in the same game.
Uses v2's BankGame as the engine (it has clone() needed by v2 MCTS, and the rules are
identical to v1's). v1 networks see a 22-input state vector built from the public state.

CRITICAL no-peeking rule still holds: every seat decides independently from the public
state, and all actions are applied in one engine step().
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

_V1_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_V2_PROJECT = os.path.abspath(os.path.join(_V1_PROJECT, "..", "bank-ai-v2-mcts"))


def _load_pkg_as(name: str, init_path: str, search: str):
    """Load a package via importlib at the given name, with submodule search location."""
    spec = importlib.util.spec_from_file_location(
        name, init_path, submodule_search_locations=[search],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_v2_modules():
    """
    Load v2's modules in a way that makes v2's bare `from game import ...` work.

    Strategy: temporarily swap sys.modules['game'], sys.modules['network'],
    sys.modules['mcts'], sys.modules['config'] to point at v2's versions while
    v2's modules load. Save and restore v1's versions around it.
    """
    saved = {}
    for name in ("game", "game.environment", "network", "network.model",
                 "mcts", "mcts.search", "config"):
        if name in sys.modules:
            saved[name] = sys.modules[name]

    try:
        # Add v2 to path so v2's modules find each other via normal import machinery
        if _V2_PROJECT not in sys.path:
            sys.path.insert(0, _V2_PROJECT)

        # Remove v1's cached entries so v2's load fresh
        for name in list(saved.keys()):
            if name in sys.modules:
                del sys.modules[name]

        # Now do the v2 imports — they'll find v2's game/, network/, mcts/, config in V2_PROJECT
        import importlib
        v2_config = importlib.import_module("config")
        v2_game = importlib.import_module("game")
        v2_environment = importlib.import_module("game.environment")
        v2_network = importlib.import_module("network")
        v2_network_model = importlib.import_module("network.model")
        v2_mcts = importlib.import_module("mcts")
        v2_mcts_search = importlib.import_module("mcts.search")

        # Stash these under unique names for our use
        sys.modules["_v2_config"] = v2_config
        sys.modules["_v2_game"] = v2_game
        sys.modules["_v2_game.environment"] = v2_environment
        sys.modules["_v2_network"] = v2_network
        sys.modules["_v2_network.model"] = v2_network_model
        sys.modules["_v2_mcts"] = v2_mcts
        sys.modules["_v2_mcts.search"] = v2_mcts_search

        return v2_environment, v2_network_model, v2_mcts_search
    finally:
        # Restore v1's modules so the rest of the server keeps working
        for name in ("game", "game.environment", "network", "network.model",
                     "mcts", "mcts.search", "config"):
            if name in sys.modules:
                del sys.modules[name]
        for name, mod in saved.items():
            sys.modules[name] = mod
        # Remove v2 path so v1's bare imports work going forward
        if _V2_PROJECT in sys.path:
            sys.path.remove(_V2_PROJECT)


_v2_environment, _v2_network_model, _v2_mcts_search = _load_v2_modules()

V2BankGame = _v2_environment.BankGame
BANK = _v2_environment.BANK
CONTINUE = _v2_environment.CONTINUE
V2BankNetwork = _v2_network_model.BankNetwork
V2MCTS = _v2_mcts_search.MCTS

# ---------------------------------------------------------------- v1 imports
# v1's project dir is on sys.path (set by server.py). Use direct imports.
from network.model import BankNetwork as V1BankNetwork  # noqa: E402
from config import Config as V1Config                   # noqa: E402

from dashboard.build_ai_compiler import compile_build_ai, state_for_build_ai  # noqa: E402


# ============================================================================
# Competitor adapters
# ============================================================================

@dataclass
class Competitor:
    name: str
    kind: str          # "v1", "v2", "custom"
    decide: Callable   # (game, player_id, rng) -> action string


def _v1_state_vector_from_v2_game(game, player_id: int) -> np.ndarray:
    """
    Build v1's 22-input state vector from a v2 BankGame's public state.
    Matches the layout in bank-ai-v1-ppo/game/environment.py:get_state_vector().
    """
    my_score = int(game.banked_scores[player_id])
    all_scores = [int(s) for s in game.banked_scores]
    opp_scores = [int(s) for i, s in enumerate(all_scores) if i != player_id]
    opp_in_round = [1 if i in game.players_in_round else 0
                    for i in range(game.num_players) if i != player_id]

    my_rank = 1 + sum(1 for s in all_scores if s > my_score)
    highest = max(all_scores)
    lowest = min(all_scores)
    highest_opp = max((s for i, s in enumerate(all_scores) if i != player_id), default=0)
    rolls_since_immunity = max(0, game.roll_number - 3)
    rounds_remaining = 10 - game.round_number  # v1 uses 10 rounds; v2 uses NUM_ROUNDS=10

    vec = [
        game.pot / 1000.0,
        game.roll_number / 10.0,
        game.round_number / 10.0,
        my_score / 1000.0,
        len(game.players_in_round) / 6.0,
        rounds_remaining / 10.0,
        1.0 if game.round_number == 10 else 0.0,
        rolls_since_immunity / 10.0,
        my_rank / max(1, game.num_players),
        (my_score - highest) / 1000.0,
        (my_score - lowest) / 1000.0,
        1.0 if (my_score + game.pot) >= highest_opp else 0.0,
    ]
    for i in range(5):
        vec.append(opp_scores[i] / 1000.0 if i < len(opp_scores) else 0.0)
    for i in range(5):
        vec.append(float(opp_in_round[i]) if i < len(opp_in_round) else 0.0)
    return np.array(vec, dtype=np.float32)


def make_v1_competitor(checkpoint_path: str, name: str) -> Competitor:
    """Load a v1 checkpoint into a competitor. Greedy argmax inference, no MCTS."""
    cfg = V1Config()
    net = V1BankNetwork(state_size=22, hidden_size=cfg.hidden_size)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    net.load_state_dict(ckpt["network_state_dict"])
    net.eval()

    def decide(game, player_id, rng):
        sv = _v1_state_vector_from_v2_game(game, player_id)
        with torch.no_grad():
            x = torch.from_numpy(sv).unsqueeze(0)
            # v1 BankNetwork has separate policy and value heads
            logits = net.policy(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        # Greedy
        action_idx = int(np.argmax(probs))
        return [BANK, CONTINUE][action_idx]

    return Competitor(name=name, kind="v1", decide=decide)


def make_v2_competitor(checkpoint_path: str, name: str, sims_per_move: int = 100) -> Competitor:
    """Load a v2 checkpoint into a competitor. Uses server-side MCTS."""
    net = V2BankNetwork()
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt["network_state_dict"] if isinstance(ckpt, dict) and "network_state_dict" in ckpt else ckpt
    net.load_state_dict(sd)
    net.eval()

    def decide(game, player_id, rng):
        mcts = V2MCTS(network=net, root_player=player_id, rng=rng)
        visits = mcts.run(game, num_simulations=sims_per_move, add_dirichlet_noise=False)
        return max(visits, key=visits.get)

    return Competitor(name=name, kind="v2", decide=decide)


def make_custom_competitor(code: str, name: str) -> Competitor:
    """Compile a Build AI pseudo-Python strategy into a competitor."""
    fn = compile_build_ai(code)

    def decide(game, player_id, rng):
        s = state_for_build_ai(game, player_id)
        try:
            out = fn(s)
        except Exception as e:
            raise RuntimeError(f"{name}: decide() raised {type(e).__name__}: {e}")
        if out not in (BANK, CONTINUE):
            raise RuntimeError(f"{name}: decide() must return 'bank' or 'continue', got {out!r}")
        return out

    return Competitor(name=name, kind="custom", decide=decide)


# ============================================================================
# Tournament runner
# ============================================================================

@dataclass
class CompetitorResult:
    name: str
    kind: str
    games_played: int = 0
    wins: float = 0.0  # fractional (ties split)
    score_sum: int = 0
    seats_played: int = 0  # how many seats this competitor occupied across all games

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.seats_played if self.seats_played else 0.0


def run_tournament(
    competitors: list[Competitor],
    num_players: int = 4,
    games_per_matchup: int = 30,
    seed: int = 12345,
) -> dict:
    """
    Run a round-robin: every distinct combination of `num_players` competitors plays
    `games_per_matchup` games. Returns a leaderboard with win rates, average scores,
    and a head-to-head summary.

    To keep the player count small enough for round-robin to be tractable, callers
    should keep len(competitors) modest (e.g. <= 8).
    """
    if len(competitors) < num_players:
        raise ValueError(
            f"Need at least {num_players} competitors for {num_players}-player games"
        )

    from itertools import combinations

    results = {c.name: CompetitorResult(name=c.name, kind=c.kind) for c in competitors}
    head_to_head: dict[tuple[str, str], dict] = {}

    rng_master = np.random.default_rng(seed)

    for combo_idxs in combinations(range(len(competitors)), num_players):
        seats = [competitors[i] for i in combo_idxs]
        for g in range(games_per_matchup):
            game_seed = int(rng_master.integers(0, 2**31 - 1))
            game = V2BankGame(num_players=num_players, seed=game_seed)
            game.reset()
            seat_rngs = [np.random.default_rng(game_seed + 1 + i * 1000) for i in range(num_players)]

            while not game.is_terminal():
                if not game.waiting_for_decisions:
                    break
                actions = {}
                for pid in sorted(game.players_in_round):
                    actions[pid] = seats[pid].decide(game, pid, seat_rngs[pid])
                game.step(actions)

            winners = game.winners()
            k = len(winners)
            share = 1.0 / k if k > 0 else 0.0

            for seat_idx, comp in enumerate(seats):
                r = results[comp.name]
                r.games_played += 1
                r.seats_played += 1
                r.score_sum += int(game.banked_scores[seat_idx])
                if seat_idx in winners:
                    r.wins += share

            # Head-to-head wins
            for i in range(num_players):
                for j in range(num_players):
                    if i == j:
                        continue
                    a = seats[i].name
                    b = seats[j].name
                    h = head_to_head.setdefault((a, b), {"games": 0, "a_wins": 0.0})
                    h["games"] += 1
                    if i in winners:
                        h["a_wins"] += share

    leaderboard = sorted(
        [
            {
                "name": r.name,
                "kind": r.kind,
                "games": r.games_played,
                "wins": round(r.wins, 2),
                "win_rate": round(r.win_rate, 4),
                "avg_score": round(r.avg_score, 1),
            }
            for r in results.values()
        ],
        key=lambda r: -r["win_rate"],
    )

    h2h = [
        {
            "a": a, "b": b,
            "games": v["games"],
            "a_wins": round(v["a_wins"], 2),
            "a_win_rate": round(v["a_wins"] / v["games"], 4) if v["games"] else 0.0,
        }
        for (a, b), v in head_to_head.items()
    ]

    return {
        "leaderboard": leaderboard,
        "head_to_head": h2h,
        "num_players": num_players,
        "games_per_matchup": games_per_matchup,
    }

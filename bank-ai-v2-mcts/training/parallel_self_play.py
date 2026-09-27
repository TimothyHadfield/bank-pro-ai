"""
Parallel self-play game generation.

A `multiprocessing.Pool` of N worker processes plays games in parallel. Each worker:
  - Receives the network's serialized state_dict (sent once per call)
  - Reconstructs a BankNetwork in CPU memory
  - Plays K games using the same self_play_game() function as serial mode
  - Returns (training_examples, GameResult) lists

This is the biggest single performance win available for Bank AI v2 on a multi-core
laptop (typical 4-8 cores). We expect roughly N× speedup at the cost of one network
state_dict copy per call (cheap — ~100 KB for our 128-hidden network).

Notes:
  - We use the "spawn" multiprocessing context to be Windows-compatible.
  - Each worker calls torch.set_num_threads(1) to avoid worker-internal threading
    over-subscribing CPU cores (which would slow things down with N workers).
  - Workers do NOT share the network in memory; we send state_dict bytes per call.
    This is fine because forwards inside each game don't need to share state with the
    main trainer process.
"""

from __future__ import annotations

import atexit
import io
import multiprocessing
from typing import Optional

import numpy as np
import torch

import config
from network import BankNetwork
from training.opponent_pool import build_default_bot_pool
from training.self_play import GameResult, TrainingExample, self_play_game


# ---------------------------------------------------------------- persistent pool
# Spawning new worker processes every iteration was costing ~3-5 sec of overhead per
# call (Windows spawn = re-import torch in each worker). We keep a module-level Pool
# alive for the lifetime of the training run so workers boot once and stay warm.

_persistent_pool: multiprocessing.pool.Pool | None = None
_persistent_pool_workers: int = 0


def _get_or_create_pool(n_workers: int) -> multiprocessing.pool.Pool:
    """Return a persistent pool with n_workers. Creates one on first call;
    recreates if n_workers changes."""
    global _persistent_pool, _persistent_pool_workers
    if _persistent_pool is None or _persistent_pool_workers != n_workers:
        if _persistent_pool is not None:
            try:
                _persistent_pool.close()
                _persistent_pool.join()
            except Exception:
                pass
        ctx = multiprocessing.get_context("spawn")
        _persistent_pool = ctx.Pool(processes=n_workers)
        _persistent_pool_workers = n_workers
    return _persistent_pool


def shutdown_pool() -> None:
    """Tear down the persistent pool. Called on training shutdown / process exit."""
    global _persistent_pool, _persistent_pool_workers
    if _persistent_pool is not None:
        try:
            _persistent_pool.close()
            _persistent_pool.terminate()
            _persistent_pool.join()
        except Exception:
            pass
        _persistent_pool = None
        _persistent_pool_workers = 0


# Make sure the pool is cleaned up if the process exits abnormally.
atexit.register(shutdown_pool)


# ---------------------------------------------------------------- worker helpers
# Each worker process gets its own copy of these globals.

# Cache of frozen networks by name (filename). Lives for the worker's lifetime.
_FROZEN_NET_CACHE: dict[str, "BankNetwork"] = {}

# Whether torch.compile has been applied to a given network instance (id-keyed).
# We can't compile frozen networks once and reuse forever — actually we can, since
# their weights never change. Live network changes every iter so compiling each
# call would defeat the cache; we just do eager mode for the live network.
_COMPILED_FROZEN: set[int] = set()


def _get_or_load_frozen_in_worker(name: str, sd_bytes: bytes):
    """Return a cached frozen network or load it once and cache it."""
    if name not in _FROZEN_NET_CACHE:
        f_net = BankNetwork()
        f_buf = io.BytesIO(sd_bytes)
        f_net.load_state_dict(torch.load(f_buf, map_location="cpu", weights_only=True))
        f_net.eval()
        # If torch.compile is enabled, compile frozen nets once (their weights never change)
        f_net = _maybe_compile_worker_net(f_net, is_frozen=True)
        _FROZEN_NET_CACHE[name] = f_net
    return _FROZEN_NET_CACHE[name]


def _maybe_compile_worker_net(net, is_frozen: bool = False):
    """Optionally JIT-compile the network's forward via torch.compile.
    Enabled only when env var BANK_AI_TORCH_COMPILE=1. Falls back silently on failure."""
    import os as _os
    if _os.environ.get("BANK_AI_TORCH_COMPILE", "0") != "1":
        return net
    # For the live network, only worth compiling if we're going to do many forwards
    # before the weights change. Skip unless caller specifically asked for frozen.
    try:
        compiled = torch.compile(net, mode="reduce-overhead", dynamic=False)
        if is_frozen:
            _COMPILED_FROZEN.add(id(compiled))
        return compiled
    except Exception as e:
        print(f"[worker] torch.compile failed ({type(e).__name__}: {e}), using eager.", flush=True)
        return net


# ---------------------------------------------------------------- worker

def _worker_play_games(
    state_dict_bytes: bytes,
    n_games: int,
    base_seed: int,
    sims_per_move: int,
    add_dirichlet_noise: bool,
    use_bot_pool: bool,
    frozen_state_dicts: list[tuple[str, bytes]],
) -> tuple[list[TrainingExample], list[GameResult]]:
    """Run inside each worker process. Plays n_games and returns (examples, results)."""
    # Avoid over-subscription: each worker should use one BLAS thread.
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # Reconstruct the live network (must rebuild every call — weights change each iter)
    net = BankNetwork()
    buf = io.BytesIO(state_dict_bytes)
    net.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    net.eval()
    net = _maybe_compile_worker_net(net)

    # Frozen networks: cached by name across iterations so we don't reload them
    # from bytes every call. The cache lives in this worker process for its lifetime.
    from training.self_play import FrozenNet
    frozen_nets: list[FrozenNet] = []
    for name, sd_bytes in (frozen_state_dicts or []):
        f_net = _get_or_load_frozen_in_worker(name, sd_bytes)
        frozen_nets.append(FrozenNet(network=f_net, name=name, sims_per_move=sims_per_move))

    bot_pool = build_default_bot_pool(seed=base_seed) if use_bot_pool else None

    all_examples: list[TrainingExample] = []
    all_results: list[GameResult] = []
    for g in range(n_games):
        examples, result = self_play_game(
            network=net,
            num_players=None,
            sims_per_move=sims_per_move,
            add_dirichlet_noise=add_dirichlet_noise,
            seed=base_seed + g,
            bot_pool=bot_pool,
            frozen_nets=frozen_nets if frozen_nets else None,
        )
        all_examples.extend(examples)
        all_results.append(result)
    return all_examples, all_results


# ---------------------------------------------------------------- public API

def parallel_self_play(
    network: BankNetwork,
    n_games: int,
    n_workers: int,
    sims_per_move: int = config.SIMS_PER_MOVE,
    base_seed: int = 0,
    add_dirichlet_noise: bool = True,
    use_bot_pool: bool = True,
    frozen_checkpoint_paths: list[str] | None = None,
) -> tuple[list[TrainingExample], list[GameResult]]:
    """
    Play `n_games` self-play games distributed across `n_workers` processes.

    Returns (combined_examples, combined_game_results).

    `frozen_checkpoint_paths` are optional file paths to older network checkpoints; if
    provided, each worker will load them and use them as opponents in the
    OLD_CHECKPOINT_MIX_FRACTION of games.
    """
    # Materialize frozen state dicts once (loading from disk to bytes)
    frozen_state_dicts: list[tuple[str, bytes]] = []
    if frozen_checkpoint_paths:
        for p in frozen_checkpoint_paths:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                sd = ck.get("network_state_dict") if isinstance(ck, dict) else ck
                buf = io.BytesIO()
                torch.save(sd, buf)
                from pathlib import Path as _P
                frozen_state_dicts.append((_P(p).name, buf.getvalue()))
            except Exception as e:
                print(f"[parallel_self_play] WARN: could not load frozen checkpoint {p}: {e}")

    if n_workers <= 1 or n_games <= 1:
        return _serial_self_play(
            network=network,
            n_games=n_games,
            sims_per_move=sims_per_move,
            base_seed=base_seed,
            add_dirichlet_noise=add_dirichlet_noise,
            use_bot_pool=use_bot_pool,
            frozen_state_dicts=frozen_state_dicts,
        )

    # Serialize the live network state_dict once
    buf = io.BytesIO()
    torch.save(network.state_dict(), buf)
    state_bytes = buf.getvalue()

    games_per_worker = [n_games // n_workers] * n_workers
    for i in range(n_games % n_workers):
        games_per_worker[i] += 1

    work = []
    cursor = 0
    for w, gpw in enumerate(games_per_worker):
        if gpw == 0:
            continue
        worker_seed = base_seed + cursor * 1000 + w * 1_000_003
        work.append((
            state_bytes, gpw, worker_seed, sims_per_move,
            add_dirichlet_noise, use_bot_pool, frozen_state_dicts,
        ))
        cursor += gpw

    pool = _get_or_create_pool(n_workers)
    results = pool.starmap(_worker_play_games, work)

    all_examples: list[TrainingExample] = []
    all_results: list[GameResult] = []
    for examples, game_results in results:
        all_examples.extend(examples)
        all_results.extend(game_results)
    return all_examples, all_results


def _serial_self_play(
    network: BankNetwork,
    n_games: int,
    sims_per_move: int,
    base_seed: int,
    add_dirichlet_noise: bool,
    use_bot_pool: bool,
    frozen_state_dicts: list[tuple[str, bytes]] | None = None,
) -> tuple[list[TrainingExample], list[GameResult]]:
    from training.self_play import FrozenNet

    bot_pool = build_default_bot_pool(seed=base_seed) if use_bot_pool else None

    frozen_nets: list[FrozenNet] = []
    if frozen_state_dicts:
        for name, sd_bytes in frozen_state_dicts:
            f_net = BankNetwork()
            f_buf = io.BytesIO(sd_bytes)
            f_net.load_state_dict(torch.load(f_buf, map_location="cpu", weights_only=True))
            f_net.eval()
            frozen_nets.append(FrozenNet(network=f_net, name=name, sims_per_move=sims_per_move))

    all_examples: list[TrainingExample] = []
    all_results: list[GameResult] = []
    for g in range(n_games):
        examples, result = self_play_game(
            network=network,
            num_players=None,
            sims_per_move=sims_per_move,
            add_dirichlet_noise=add_dirichlet_noise,
            seed=base_seed + g,
            bot_pool=bot_pool,
            frozen_nets=frozen_nets if frozen_nets else None,
        )
        all_examples.extend(examples)
        all_results.append(result)
    return all_examples, all_results

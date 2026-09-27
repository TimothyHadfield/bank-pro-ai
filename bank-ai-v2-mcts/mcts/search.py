"""
Monte Carlo Tree Search for Bank AI v2.

Slice 1 implementation: minimal but correct.

Tree structure:
  - Root represents the current game state where the root_player is about to act.
  - At each "decision node" (root_player is in players_in_round), children are 'bank' / 'continue'.
  - Opponents act simultaneously with us; their actions are SAMPLED from their policy
    (never optimized). This enforces the no-peeking rule per PLAN.md §4.6.
  - Dice rolls happen inside game.step() automatically. We don't model them as separate nodes;
    the resulting state is just whatever step() produced.
  - Children are deduplicated across simulations by a state-key hash.

Backup uses standard MCTS: each simulation propagates a value back to the root,
incrementing N and W along the path. Action selection uses the PUCT formula.

Slice 1 uses unbatched per-call network inference. Batching is a Slice 4 optimization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import config
from game.environment import BANK, CONTINUE, BankGame

if TYPE_CHECKING:
    from network.model import BankNetwork


# ---------------------------------------------------------------- helpers

ACTIONS = (BANK, CONTINUE)


def state_key(game: BankGame) -> tuple:
    """Hashable identifier for a game state. Used to dedupe child nodes across simulations."""
    return (
        game.round_number,
        game.roll_number,
        game.pot,
        tuple(game.banked_scores),
        tuple(sorted(game.players_in_round)),
        game.last_dice,
        game.game_over,
    )


def outcome_value(game: BankGame, root_player: int) -> float:
    """
    Reward from the root player's perspective at a terminal state.
      - 1.0 if root won outright (sole winner)
      - 1/k if tied for first with k winners
      - 0.0 otherwise
    """
    winners = game.winners()
    if root_player in winners:
        return 1.0 / len(winners)
    return 0.0


# ---------------------------------------------------------------- node

class Node:
    """
    A single state in the MCTS tree.

    Stats are accumulated across simulations that pass through this node.

    For decision nodes (root_player is currently waiting to act), we keep:
      - action_priors:  P(a) from the policy network
      - action_N[a]:    visit count for taking action a from this node
      - action_W[a]:    total backed-up value when we took action a
    For non-decision nodes (root_player is out of the round but game continues),
    we just step forward (sampling opponents) and accumulate aggregate N/W on this node.
    """

    __slots__ = (
        "game",
        "root_player",
        "is_decision_node",
        "is_terminal",
        "is_expanded",
        "N",
        "W",
        "action_priors",
        "action_N",
        "action_W",
        "children_by_key",
    )

    def __init__(self, game: BankGame, root_player: int):
        self.game = game
        self.root_player = root_player
        self.is_terminal = game.is_terminal()
        # Decision node iff root is currently in the round AND game is awaiting decisions
        self.is_decision_node = (
            (not self.is_terminal)
            and game.waiting_for_decisions
            and root_player in game.players_in_round
        )
        self.is_expanded = False

        self.N = 0
        self.W = 0.0

        # Decision-node bookkeeping (filled lazily on expansion)
        self.action_priors: dict[str, float] = {}
        self.action_N: dict[str, int] = {a: 0 for a in ACTIONS}
        self.action_W: dict[str, float] = {a: 0.0 for a in ACTIONS}

        # Cache of resulting child nodes keyed by state_key(next_game)
        self.children_by_key: dict[tuple, "Node"] = {}

    def Q(self, action: str) -> float:
        n = self.action_N[action]
        return self.action_W[action] / n if n > 0 else 0.0


# ---------------------------------------------------------------- mcts

class MCTS:
    """
    Run MCTS for one decision from a single player's perspective.

    Usage:
        mcts = MCTS(network, root_player=0, c_puct=1.4)
        visit_dist = mcts.run(game, num_simulations=100, add_dirichlet_noise=True)
        # visit_dist is dict[str, int] giving visit counts per action
    """

    def __init__(
        self,
        network: "BankNetwork",
        root_player: int,
        c_puct: float = config.C_PUCT,
        rng: np.random.Generator | None = None,
    ):
        self.network = network
        self.root_player = root_player
        self.c_puct = c_puct
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------ public API

    def run(
        self,
        game: BankGame,
        num_simulations: int,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = config.DIRICHLET_ALPHA,
        dirichlet_epsilon: float = config.DIRICHLET_EPSILON,
    ) -> dict[str, int]:
        """
        Build a tree from a clone of `game` and return action visit counts at the root.

        The root player must be currently in the round and waiting to decide; otherwise
        this is a programming error (we don't run MCTS for a player who can't act).
        """
        if game.is_terminal():
            raise ValueError("Cannot MCTS from a terminal state")
        if not game.waiting_for_decisions:
            raise ValueError("Cannot MCTS unless engine is waiting for decisions")
        if self.root_player not in game.players_in_round:
            raise ValueError(
                f"Root player {self.root_player} is not in the round; cannot run MCTS for them"
            )

        # Build root from a clone (do NOT mutate caller's game)
        root = Node(game.clone(), self.root_player)

        # Expand root once with the network so we have priors before the loop
        self._expand(root)

        # Optional Dirichlet noise mixed into the root prior (self-play exploration)
        if add_dirichlet_noise:
            self._apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)

        return dict(root.action_N)

    # ------------------------------------------------------------ core simulation

    def _simulate(self, node: Node) -> float:
        """
        Recursively simulate from `node`, returning the value for the root player.

        Updates N and W along the path on the way back up.
        """
        # Terminal: return real outcome
        if node.is_terminal:
            v = outcome_value(node.game, self.root_player)
            node.N += 1
            node.W += v
            return v

        # Decision-for-root node
        if node.is_decision_node:
            if not node.is_expanded:
                # First visit: expand with network (returns the value estimate)
                v = self._expand(node)
                node.N += 1
                node.W += v
                return v

            # Already expanded: PUCT-select an action and recurse
            action = self._puct_select(node)
            opp_actions = self._sample_opponent_actions(node.game)
            all_actions = {self.root_player: action, **opp_actions}

            next_game = node.game.clone()
            next_game.step(all_actions)
            child = self._get_or_create_child(node, next_game)
            v = self._simulate(child)

            # Backup
            node.N += 1
            node.W += v
            node.action_N[action] += 1
            node.action_W[action] += v
            return v

        # Non-decision node (root is out of this round but game continues).
        # Just step forward by sampling all active players' actions.
        opp_actions = self._sample_opponent_actions(node.game, exclude=None)
        # exclude=None means sample for ALL players in round (root isn't in)
        next_game = node.game.clone()
        next_game.step(opp_actions)
        child = self._get_or_create_child(node, next_game)
        v = self._simulate(child)

        node.N += 1
        node.W += v
        return v

    # ------------------------------------------------------------ helpers

    def _expand(self, node: Node) -> float:
        """
        Evaluate the node with the network. Fills priors and returns value estimate.
        Should only be called on a decision_node.
        """
        assert node.is_decision_node
        sv = node.game.get_state_vector(self.root_player)
        policy_probs, value = self.network.predict(sv)
        node.action_priors = {
            BANK: float(policy_probs[0]),
            CONTINUE: float(policy_probs[1]),
        }
        node.is_expanded = True
        return value

    def _puct_select(self, node: Node) -> str:
        """Pick the action with highest UCB. PUCT formula per PLAN.md §4.2."""
        sqrt_parent_N = math.sqrt(max(1, node.N))
        best_action = None
        best_score = -float("inf")
        for action in ACTIONS:
            q = node.Q(action)
            p = node.action_priors[action]
            n = node.action_N[action]
            u = self.c_puct * p * sqrt_parent_N / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        assert best_action is not None
        return best_action

    def _sample_opponent_actions(
        self, game: BankGame, exclude: int | None = "ROOT"  # type: ignore[assignment]
    ) -> dict[int, str]:
        """
        For each player in `game.players_in_round` (excluding root unless overridden),
        sample an action from THEIR policy distribution.

        This is where the "no peeking" rule lives in MCTS: each opponent's action is
        sampled independently from their own state vector's policy. They don't see
        each other's samples, and we don't see them at the real game level either.

        Args:
            exclude:
              - "ROOT" (default sentinel) : exclude root_player (we already have their action)
              - None                      : sample actions for ALL players in round
              - int                       : exclude that specific player id
        """
        if exclude == "ROOT":
            exclude_set = {self.root_player}
        elif exclude is None:
            exclude_set = set()
        else:
            exclude_set = {exclude}

        # Collect every opponent we need to sample for, then do ONE batched forward
        # instead of one forward per opponent. Identical outputs, ~5× fewer Python+
        # tensor calls when there are 5 opponents at the decision point.
        pids_to_sample: list[int] = []
        for pid in game.players_in_round:
            if pid in exclude_set:
                continue
            pids_to_sample.append(pid)

        if not pids_to_sample:
            return {}

        state_vectors = np.stack([game.get_state_vector(pid) for pid in pids_to_sample])
        policy_probs_batch, _ = self.network.predict_batch(state_vectors)

        actions: dict[int, str] = {}
        for i, pid in enumerate(pids_to_sample):
            idx = self.rng.choice(len(ACTIONS), p=policy_probs_batch[i])
            actions[pid] = ACTIONS[idx]
        return actions

    def _get_or_create_child(self, node: Node, next_game: BankGame) -> Node:
        key = state_key(next_game)
        if key not in node.children_by_key:
            node.children_by_key[key] = Node(next_game, self.root_player)
        return node.children_by_key[key]

    def _apply_dirichlet_noise(
        self, root: Node, alpha: float, epsilon: float
    ) -> None:
        """Mix Dirichlet(alpha) noise into the root prior for exploration during self-play."""
        noise = self.rng.dirichlet([alpha] * len(ACTIONS))
        for i, action in enumerate(ACTIONS):
            root.action_priors[action] = (
                (1 - epsilon) * root.action_priors[action] + epsilon * noise[i]
            )

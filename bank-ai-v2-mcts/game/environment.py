"""
BankGame engine for v2.

Implements the rules in GAME_RULES.md exactly. Designed to be:
  - deterministic given a seed
  - clonable (deep copy with independent RNG) for MCTS rollouts
  - simultaneous-action: step() requires actions for ALL active players at once
  - variable player count (2-6 players)
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Iterable

import numpy as np

import config

IMMUNITY = "immunity"
POST_IMMUNITY = "post_immunity"

BANK = "bank"
CONTINUE = "continue"


class BankGame:
    """
    Stateful Bank simulator. The external interface is:
      reset()                                        -> initial state
      get_state(player_id)                           -> dict (public state from that player's view)
      get_state_vector(player_id)                    -> np.ndarray (fixed size INPUT_SIZE)
      step(actions: dict[int, str])                  -> advances the game
      clone()                                        -> independent deep copy
      legal_actions(player_id)                       -> list of legal action strings
      is_terminal()                                  -> bool
      winners()                                      -> list[int]  (handles ties)

    External flow:
      game.reset()
      while not game.is_terminal():
          if game.waiting_for_decisions:
              actions = {pid: choose(...) for pid in game.players_in_round}
              game.step(actions)
          else:
              # The engine auto-handles immunity rolls; this branch should not be hit
              # by callers — reset() advances through immunity automatically.
              ...
    """

    # ---------------------------------------------------------------- init / reset

    def __init__(self, num_players: int, seed: int | None = None):
        if not (2 <= num_players <= 6):
            raise ValueError(f"num_players must be 2..6 (got {num_players})")
        self.num_players = num_players
        self._initial_seed = config.DEFAULT_SEED if seed is None else seed
        # _seed_counter is used so cloned games get distinct but deterministic seeds
        self._seed_counter = 0
        self.rng = random.Random(self._initial_seed)

        self.banked_scores: list[int] = [0] * num_players
        self.round_number = 0
        self.roll_number = 0
        self.pot = 0
        self.players_in_round: set[int] = set()
        self.last_dice: tuple[int, int] | None = None
        self.immunity_history: list[tuple[int, int]] = []
        self.game_over = False
        self.round_over = False
        self.waiting_for_decisions = False
        # Tracks per-round how many players banked (lets state vector see "previous bankers" without leaking same-turn)
        self.bankers_history_this_round: list[int] = []  # ordered list of (post-bank) banker counts
        # Tracks rolls since immunity ended within the current round
        self.rolls_since_immunity_ended = 0

    def reset(self) -> None:
        """Start a new game from scratch."""
        self.banked_scores = [0] * self.num_players
        self.round_number = 0
        self.roll_number = 0
        self.pot = 0
        self.players_in_round = set()
        self.last_dice = None
        self.immunity_history = []
        self.game_over = False
        self.round_over = False
        self.waiting_for_decisions = False
        self.bankers_history_this_round = []
        self.rolls_since_immunity_ended = 0
        self._start_new_round()

    # ---------------------------------------------------------------- internal: round mgmt

    def _start_new_round(self) -> None:
        """Begin a new round. Advances through immunity automatically."""
        if self.round_number >= config.NUM_ROUNDS:
            self.game_over = True
            self.waiting_for_decisions = False
            return

        self.round_number += 1
        self.roll_number = 0
        self.pot = 0
        self.players_in_round = set(range(self.num_players))
        self.immunity_history = []
        self.round_over = False
        self.bankers_history_this_round = []
        self.rolls_since_immunity_ended = 0

        # Auto-roll the immunity rolls (1-3). Players cannot bank during immunity.
        for _ in range(config.IMMUNITY_ROLLS):
            self._roll_and_apply()
            if self.round_over:  # Shouldn't happen during immunity (7 doesn't bust), but defensive
                break

        if not self.round_over:
            self.waiting_for_decisions = True

    def _roll_dice(self) -> tuple[int, int]:
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        return d1, d2

    def _roll_and_apply(self) -> None:
        """Advance one roll, applying immunity vs post-immunity rules."""
        self.roll_number += 1
        d1, d2 = self._roll_dice()
        self.last_dice = (d1, d2)
        total = d1 + d2

        if self.roll_number <= config.IMMUNITY_ROLLS:
            # Immunity: 7 adds 70, doubles just sum, others sum
            if total == 7:
                self.pot += 70
            else:
                self.pot += total
            self.immunity_history.append((d1, d2))
        else:
            # Post-immunity
            if total == 7:
                # Bust: pot wiped, round ends. Players still in get nothing.
                self.pot = 0
                self.round_over = True
                self.waiting_for_decisions = False
                # Players still in lose their chance this round. End round.
                self._end_round()
                return
            elif d1 == d2:
                # Doubles: pot multiplies by 2
                self.pot *= 2
            else:
                self.pot += total
            self.rolls_since_immunity_ended += 1

    def _end_round(self) -> None:
        """Wrap up a round and start the next (or end the game)."""
        self.round_over = True
        self.waiting_for_decisions = False
        self.players_in_round = set()
        if self.round_number >= config.NUM_ROUNDS:
            self.game_over = True
        else:
            self._start_new_round()

    # ---------------------------------------------------------------- step (the simultaneous-action gate)

    def step(self, actions: dict[int, str]) -> None:
        """
        Advance the game by one decision point.

        `actions` must contain EXACTLY one entry for each player currently in the round.
        All actions are applied simultaneously.

        After processing:
          - bankers receive the current pot value
          - bankers are removed from players_in_round
          - if any players remain, the next dice are rolled
          - if a 7 is rolled, the round ends
          - if all players banked, the round ends
          - waiting_for_decisions is updated for the next loop
        """
        if self.game_over:
            raise RuntimeError("Cannot step a terminal game")
        if not self.waiting_for_decisions:
            raise RuntimeError("Engine is not waiting for decisions; call reset() if needed")

        # --- enforce simultaneous-action invariant ---
        active = self.players_in_round
        if set(actions.keys()) != active:
            raise ValueError(
                f"step() requires exactly one action per active player. "
                f"Active={sorted(active)}, got={sorted(actions.keys())}"
            )
        for pid, a in actions.items():
            if a not in (BANK, CONTINUE):
                raise ValueError(f"Invalid action {a!r} for player {pid}")

        # --- gather bankers, then apply all together ---
        bankers = [pid for pid, a in actions.items() if a == BANK]
        continuers = [pid for pid, a in actions.items() if a == CONTINUE]

        pot_at_decision = self.pot  # all bankers receive THIS value
        for pid in bankers:
            self.banked_scores[pid] += pot_at_decision

        # Remove bankers from active set
        self.players_in_round = set(continuers)
        self.bankers_history_this_round.append(len(bankers))

        # If everyone banked, round ends; pot stays at its current value but is unused
        if not self.players_in_round:
            self._end_round()
            return

        # Otherwise, roll the next die
        self._roll_and_apply()

        # If the roll busted (post-immunity 7), _roll_and_apply already ended the round.
        if self.game_over:
            return
        if self.round_over:
            return

        # Still players in, no bust: waiting for next decision
        self.waiting_for_decisions = True

    # ---------------------------------------------------------------- introspection

    def is_terminal(self) -> bool:
        return self.game_over

    def legal_actions(self, player_id: int) -> list[str]:
        if player_id not in self.players_in_round:
            return []
        return [BANK, CONTINUE]

    def winners(self) -> list[int]:
        if not self.game_over:
            return []
        top = max(self.banked_scores)
        return [pid for pid, s in enumerate(self.banked_scores) if s == top]

    def get_state(self, player_id: int) -> dict:
        """Public state from this player's perspective. NEVER reveals same-turn actions."""
        opp_scores = [s for i, s in enumerate(self.banked_scores) if i != player_id]
        opp_in_round = [int(i in self.players_in_round) for i in range(self.num_players) if i != player_id]
        return {
            "player_id": player_id,
            "num_players": self.num_players,
            "round_number": self.round_number,
            "roll_number": self.roll_number,
            "pot": self.pot,
            "my_banked_score": self.banked_scores[player_id],
            "opponent_banked_scores": opp_scores,
            "opponent_in_round": opp_in_round,
            "players_still_in": len(self.players_in_round),
            "immunity_active": self.roll_number <= config.IMMUNITY_ROLLS,
            "rolls_since_immunity_ended": self.rolls_since_immunity_ended,
            "last_dice": self.last_dice,
            "round_over": self.round_over,
            "game_over": self.game_over,
            "waiting_for_decisions": self.waiting_for_decisions,
            "i_am_in_round": player_id in self.players_in_round,
        }

    # ---------------------------------------------------------------- state vector for the network

    def get_state_vector(self, player_id: int) -> np.ndarray:
        """
        Return a fixed-size INPUT_SIZE float vector. Layout matches PLAN.md §2.3.

        Padded opponent slots use sentinel values so the network can detect padding via
        input #25 (num_real_opponents).
        """
        v = np.zeros(config.INPUT_SIZE, dtype=np.float32)

        my_score = self.banked_scores[player_id]
        opp_pairs = [
            (s, int(i in self.players_in_round))
            for i, s in enumerate(self.banked_scores)
            if i != player_id
        ]
        # Sort opponents by score descending for stable ordering
        opp_pairs.sort(key=lambda x: -x[0])

        all_scores = list(self.banked_scores)
        sorted_scores_desc = sorted(all_scores, reverse=True)
        my_rank = sorted_scores_desc.index(my_score) + 1  # 1 = leading
        # If ties, rank() picks the first occurrence — fine for our use

        gap_to_first = sorted_scores_desc[0] - my_score
        gap_to_last = my_score - sorted_scores_desc[-1]

        # Would banking now make me first?
        hypothetical = my_score + self.pot
        max_other = max((s for i, s in enumerate(self.banked_scores) if i != player_id), default=0)
        would_make_first = int(hypothetical > max_other)

        rounds_remaining = config.NUM_ROUNDS - self.round_number
        is_final_round = int(self.round_number == config.NUM_ROUNDS)
        immunity_active = int(self.roll_number <= config.IMMUNITY_ROLLS)
        num_real_opponents = self.num_players - 1

        # ---- fill the vector ----
        v[0] = self.pot / 1000.0
        v[1] = self.roll_number / 10.0
        v[2] = self.round_number / 10.0
        v[3] = my_score / 1000.0
        v[4] = self.num_players / 6.0
        v[5] = len(self.players_in_round) / 6.0
        v[6] = rounds_remaining / 10.0
        v[7] = float(is_final_round)
        v[8] = self.rolls_since_immunity_ended / 10.0
        v[9] = my_rank / max(1, self.num_players)
        v[10] = gap_to_first / 1000.0
        v[11] = gap_to_last / 1000.0
        v[12] = float(would_make_first)
        v[13] = float(immunity_active)

        # 14..18 (5 slots): opponent scores normalized; padding -1.0/1000.0 = -0.001
        # 19..23 (5 slots): opponent in-round flags; padding 0
        for i in range(config.MAX_OPPONENTS):
            if i < len(opp_pairs):
                v[14 + i] = opp_pairs[i][0] / 1000.0
                v[19 + i] = float(opp_pairs[i][1])
            else:
                v[14 + i] = -1.0 / 1000.0  # sentinel
                v[19 + i] = 0.0

        v[24] = num_real_opponents / config.MAX_OPPONENTS

        # 25..27: one-hot (leading, middle, trailing)
        if my_rank == 1:
            v[25] = 1.0  # leading
        elif my_rank == self.num_players:
            v[27] = 1.0  # trailing
        else:
            v[26] = 1.0  # middle

        return v

    # ---------------------------------------------------------------- cloning for MCTS

    def clone(self) -> "BankGame":
        """
        Deep copy of the game with an independent RNG.

        The cloned RNG is seeded deterministically from the parent's seed counter so
        clones produce reproducible (but distinct) random streams.
        """
        # Get a fresh deterministic seed from the parent
        self._seed_counter += 1
        derived_seed = (self._initial_seed * 1_000_003 + self._seed_counter) & 0x7FFFFFFF

        new = BankGame.__new__(BankGame)
        new.num_players = self.num_players
        new._initial_seed = derived_seed
        new._seed_counter = 0
        new.rng = random.Random(derived_seed)
        new.banked_scores = list(self.banked_scores)
        new.round_number = self.round_number
        new.roll_number = self.roll_number
        new.pot = self.pot
        new.players_in_round = set(self.players_in_round)
        new.last_dice = self.last_dice
        new.immunity_history = list(self.immunity_history)
        new.game_over = self.game_over
        new.round_over = self.round_over
        new.waiting_for_decisions = self.waiting_for_decisions
        new.bankers_history_this_round = list(self.bankers_history_this_round)
        new.rolls_since_immunity_ended = self.rolls_since_immunity_ended
        return new

    # ---------------------------------------------------------------- debug helpers

    def __repr__(self) -> str:
        return (
            f"BankGame(N={self.num_players}, R{self.round_number}/{config.NUM_ROUNDS} "
            f"roll={self.roll_number} pot={self.pot} "
            f"in={sorted(self.players_in_round)} scores={self.banked_scores} "
            f"{'TERMINAL' if self.game_over else ('WAIT' if self.waiting_for_decisions else 'BETWEEN')})"
        )

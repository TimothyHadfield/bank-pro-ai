import numpy as np
from typing import Dict, List, Tuple, Optional


class BankGame:
    """
    Simulates the dice game 'Bank' for 2-6 players over 10 rounds.

    The pot is shared among all players. Each round, the first 3 rolls
    (immunity) happen automatically — no banking allowed. Starting from
    roll 4, players decide to bank or continue each roll.
    """

    def __init__(self, num_players: int = 4, num_rounds: int = 10, seed: Optional[int] = None):
        if not 2 <= num_players <= 6:
            raise ValueError("num_players must be between 2 and 6")
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> Dict[int, dict]:
        """Reset the game to its initial state. Returns initial states for all players."""
        self.round_number = 1
        self.roll_number = 0
        self.pot = 0
        self.banked_scores = np.zeros(self.num_players, dtype=np.int64)
        self.players_in_round = set(range(self.num_players))
        self.game_over = False
        self.round_over = False
        self.waiting_for_decisions = False
        self.last_dice = None
        self.immunity_history = []

        self._roll_immunity()

        return self._get_all_states()

    def _roll_dice(self):
        """Roll two dice and apply the result to the pot. May trigger round end."""
        self.roll_number += 1
        die1 = self.rng.randint(1, 7)
        die2 = self.rng.randint(1, 7)
        self.last_dice = (die1, die2)
        dice_sum = die1 + die2
        is_doubles = (die1 == die2)
        is_seven = (dice_sum == 7)
        immunity_active = (self.roll_number <= 3)

        if immunity_active:
            if is_seven:
                self.pot += 70
            else:
                self.pot += dice_sum
            self.round_over = False
            self.waiting_for_decisions = False  # No decisions during immunity
        else:
            if is_seven:
                self.pot = 0
                self.round_over = True
                self.waiting_for_decisions = False
            elif is_doubles:
                self.pot *= 2
                self.round_over = False
                self.waiting_for_decisions = True
            else:
                self.pot += dice_sum
                self.round_over = False
                self.waiting_for_decisions = True

    def _roll_immunity(self):
        """Roll the 3 immunity rolls. After this, players get their first decision."""
        self.immunity_history = []
        for _ in range(3):
            self._roll_dice()
            self.immunity_history.append({
                "dice": self.last_dice,
                "pot": int(self.pot),
                "roll_number": self.roll_number,
            })
        # After immunity, players decide bank/continue BEFORE the next roll
        self.waiting_for_decisions = True

    def _start_new_round(self):
        """Begin a new round."""
        self.round_number += 1
        if self.round_number > self.num_rounds:
            self.game_over = True
            return
        self.roll_number = 0
        self.pot = 0
        self.players_in_round = set(range(self.num_players))
        self.round_over = False
        self._roll_immunity()

    def get_state(self, player_id: int) -> dict:
        """Return the game state from a specific player's perspective."""
        my_score = int(self.banked_scores[player_id])
        all_scores = [int(self.banked_scores[i]) for i in range(self.num_players)]

        # Per-opponent info in fixed player order (excluding self)
        opponent_scores = []
        opponent_in_round = []
        for i in range(self.num_players):
            if i == player_id:
                continue
            opponent_scores.append(int(self.banked_scores[i]))
            opponent_in_round.append(i in self.players_in_round)

        # Rank: 1 = first place (fewest players with higher score + 1)
        my_rank = 1 + sum(1 for s in all_scores if s > my_score)

        # Score gaps
        highest_score = max(all_scores)
        lowest_score = min(all_scores)
        highest_opponent = max(s for i, s in enumerate(all_scores) if i != player_id) if self.num_players > 1 else 0

        return {
            "current_pot": int(self.pot),
            "roll_number": self.roll_number,
            "round_number": self.round_number,
            "my_banked_score": my_score,
            "opponent_banked_scores": opponent_scores,
            "opponent_in_round": opponent_in_round,
            "players_still_in": len(self.players_in_round),
            "rounds_remaining": self.num_rounds - self.round_number,
            "is_final_round": self.round_number == self.num_rounds,
            "rolls_since_immunity": max(0, self.roll_number - 3),
            "my_rank": my_rank,
            "gap_to_first": my_score - highest_score,
            "gap_to_last": my_score - lowest_score,
            "would_banking_win": (my_score + self.pot) >= highest_opponent,
            "num_players": self.num_players,
        }

    def get_state_vector(self, player_id: int) -> np.ndarray:
        """Return the game state as a flat numpy array for neural network input.

        22 inputs:
          0: pot/1000
          1: roll_number/10
          2: round_number/10
          3: my_score/1000
          4: players_still_in/6
          5: rounds_remaining/10
          6: is_final_round (0 or 1)
          7: rolls_since_immunity/10
          8: my_rank/num_players
          9: gap_to_first/1000
         10: gap_to_last/1000
         11: would_banking_win (0 or 1)
         12-16: opponent_scores[0..4]/1000
         17-21: opponent_in_round[0..4] (0 or 1)
        """
        s = self.get_state(player_id)
        vec = [
            s["current_pot"] / 1000.0,
            s["roll_number"] / 10.0,
            s["round_number"] / 10.0,
            s["my_banked_score"] / 1000.0,
            s["players_still_in"] / 6.0,
            s["rounds_remaining"] / 10.0,
            1.0 if s["is_final_round"] else 0.0,
            s["rolls_since_immunity"] / 10.0,
            s["my_rank"] / s["num_players"],
            s["gap_to_first"] / 1000.0,
            s["gap_to_last"] / 1000.0,
            1.0 if s["would_banking_win"] else 0.0,
        ]
        # Per-opponent scores (padded to 5)
        for i in range(5):
            vec.append(s["opponent_banked_scores"][i] / 1000.0 if i < len(s["opponent_banked_scores"]) else 0.0)
        # Per-opponent in-round status (padded to 5)
        for i in range(5):
            vec.append(1.0 if (i < len(s["opponent_in_round"]) and s["opponent_in_round"][i]) else 0.0)
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def state_size() -> int:
        """Return the size of the state vector."""
        return 22  # 12 base features + 5 opponent scores + 5 opponent in-round

    def _get_all_states(self) -> Dict[int, dict]:
        """Return states for all players currently in the round."""
        return {pid: self.get_state(pid) for pid in self.players_in_round}

    def step(self, actions: Dict[int, str]) -> Tuple[Dict[int, dict], bool, bool, dict]:
        """
        Process player actions and advance the game.

        Args:
            actions: dict mapping player_id -> "bank" or "continue" for each
                     player still in the round.

        Returns:
            states: dict of player states for players still in the round
            round_done: whether the current round just ended
            game_done: whether the entire game just ended
            info: additional information about what happened
        """
        if self.game_over:
            raise RuntimeError("Game is already over. Call reset().")
        if not self.waiting_for_decisions:
            raise RuntimeError("Not waiting for decisions. The round may have ended.")

        # Validate actions
        for pid in self.players_in_round:
            if pid not in actions:
                raise ValueError(f"Missing action for player {pid}")
            if actions[pid] not in ("bank", "continue"):
                raise ValueError(f"Invalid action '{actions[pid]}' for player {pid}. Must be 'bank' or 'continue'.")

        info = {
            "dice": self.last_dice,
            "pot_before_actions": int(self.pot),
            "banked_players": [],
            "round_ended_by_seven": False,
        }

        # Process banking decisions
        bankers = []
        for pid in list(self.players_in_round):
            if actions[pid] == "bank":
                self.banked_scores[pid] += self.pot
                bankers.append(pid)

        info["banked_players"] = bankers

        # Remove bankers from the round
        for pid in bankers:
            self.players_in_round.discard(pid)

        # Check if all players have banked
        if len(self.players_in_round) == 0:
            self.round_over = True
            self.waiting_for_decisions = False
            self._start_new_round()
            return (
                self._get_all_states() if not self.game_over else {},
                True,
                self.game_over,
                info,
            )

        # Roll the next dice (always post-immunity since decisions only happen post-immunity)
        self._roll_dice()

        if self.round_over:
            # A 7 was rolled — pot wiped, round ends
            info["round_ended_by_seven"] = True
            self._start_new_round()
            return (
                self._get_all_states() if not self.game_over else {},
                True,
                self.game_over,
                info,
            )

        # Round continues — return states for remaining players
        return (
            self._get_all_states(),
            False,
            False,
            info,
        )

    def get_final_scores(self) -> np.ndarray:
        """Return the final banked scores for all players."""
        return self.banked_scores.copy()

    def get_winners(self) -> List[int]:
        """Return list of player IDs with the highest score (handles ties)."""
        max_score = self.banked_scores.max()
        return [i for i in range(self.num_players) if self.banked_scores[i] == max_score]

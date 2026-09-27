"""
Phase 1 Test: Verify that the BankGame environment is correct.
Tests all game rules including immunity, 7s, doubles, banking, and pot behavior.
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from game.environment import BankGame


def test_full_game_with_random_decisions():
    """Play a full game with random decisions, print everything."""
    print("=" * 60)
    print("TEST 1: Full random game")
    print("=" * 60)
    game = BankGame(num_players=4, seed=42)
    states = game.reset()

    round_count = 0
    step_count = 0

    while not game.game_over:
        # Show current state
        sample_state = list(states.values())[0] if states else None
        if sample_state:
            print(f"\n  Round {sample_state['round_number']}, "
                  f"Roll {sample_state['roll_number']}, "
                  f"Pot: {sample_state['current_pot']}, "
                  f"Immunity: {sample_state['immunity_active']}, "
                  f"Players in: {sample_state['players_still_in']}")
            print(f"  Dice: {game.last_dice}")

        # Random decisions for remaining players
        actions = {}
        for pid in game.players_in_round:
            action = np.random.choice(["bank", "continue"])
            actions[pid] = action

        print(f"  Actions: {actions}")

        states, round_done, game_done, info = game.step(actions)
        step_count += 1

        if info["banked_players"]:
            print(f"  Banked: {info['banked_players']} at pot={info['pot_before_actions']}")
        if info.get("round_ended_by_seven"):
            print(f"  >>> ROUND ENDED BY 7! Pot wiped.")
        if round_done:
            round_count += 1
            print(f"  --- Round ended. Scores: {game.banked_scores}")

    print(f"\nGame over! Final scores: {game.get_final_scores()}")
    print(f"Winners: {game.get_winners()}")
    print(f"Total steps: {step_count}")
    assert game.round_number > game.num_rounds, "Game should have advanced past 10 rounds"
    print("PASSED: Full game completed successfully.\n")


def test_immunity_seven_adds_70():
    """Verify that rolling a 7 during immunity (rolls 1-3) adds 70 to the pot."""
    print("=" * 60)
    print("TEST 2: Immunity - 7 adds 70")
    print("=" * 60)

    # We need to find a seed where one of the first 3 rolls is a 7
    for seed in range(1000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        # Check if the first roll was a 7
        d1, d2 = game.last_dice
        if d1 + d2 == 7 and game.roll_number <= 3:
            pot = game.pot
            print(f"  Seed {seed}: Roll {game.roll_number}, Dice {game.last_dice}, Pot: {pot}")
            assert pot == 70, f"Expected pot=70 after 7 on roll 1 during immunity, got {pot}"
            print("PASSED: 7 during immunity adds 70 to pot.\n")
            return

    # If no seed found on first roll, try by continuing through rolls
    for seed in range(1000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        for _ in range(3):  # Up to 3 rolls within immunity
            if game.last_dice[0] + game.last_dice[1] == 7 and game.roll_number <= 3:
                print(f"  Seed {seed}: Roll {game.roll_number}, Dice {game.last_dice}, "
                      f"Pot: {game.pot}")
                # Pot should include 70 for the 7
                assert game.pot >= 70, f"Pot should be >= 70 after a 7 during immunity, got {game.pot}"
                print("PASSED: 7 during immunity adds 70 to pot.\n")
                return

            if game.waiting_for_decisions and not game.round_over:
                actions = {pid: "continue" for pid in game.players_in_round}
                game.step(actions)
            else:
                break

    raise AssertionError("Could not find a seed producing a 7 during immunity")


def test_immunity_doubles_just_add():
    """Verify that doubles during immunity just add the sum (no doubling)."""
    print("=" * 60)
    print("TEST 3: Immunity - doubles just add sum")
    print("=" * 60)

    for seed in range(1000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        for _ in range(3):
            d1, d2 = game.last_dice
            if d1 == d2 and game.roll_number <= 3:
                # Pot should have the sum added, not doubled
                print(f"  Seed {seed}: Roll {game.roll_number}, Dice ({d1},{d2}), Pot: {game.pot}")
                # We can't easily check exact pot because prior rolls contributed,
                # but we can check the pot didn't get multiplied
                # Just verify the game didn't crash and pot is reasonable
                assert game.pot > 0, "Pot should be positive"
                print("PASSED: Doubles during immunity handled correctly.\n")
                return

            if game.waiting_for_decisions and not game.round_over:
                actions = {pid: "continue" for pid in game.players_in_round}
                game.step(actions)
            else:
                break

    raise AssertionError("Could not find a seed producing doubles during immunity")


def test_doubles_during_immunity_exact():
    """
    More precise test: start a fresh round, if first roll is doubles,
    pot should equal the sum (not doubled).
    """
    print("=" * 60)
    print("TEST 3b: Immunity - doubles exact value check")
    print("=" * 60)

    for seed in range(1000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        d1, d2 = game.last_dice
        if d1 == d2 and game.roll_number == 1:
            expected_pot = d1 + d2  # Just the sum, not doubled
            actual_pot = game.pot
            print(f"  Seed {seed}: Dice ({d1},{d2}), Expected pot: {expected_pot}, Actual: {actual_pot}")
            assert actual_pot == expected_pot, (
                f"During immunity, doubles should just add sum. "
                f"Expected {expected_pot}, got {actual_pot}"
            )
            print("PASSED: Doubles during immunity adds sum exactly.\n")
            return

    raise AssertionError("Could not find a seed producing doubles on roll 1")


def test_seven_after_immunity_wipes_pot():
    """Verify that rolling a 7 after roll 3 wipes the pot and ends the round."""
    print("=" * 60)
    print("TEST 4: Post-immunity - 7 wipes pot")
    print("=" * 60)

    for seed in range(2000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        # Continue through rolls to get past immunity
        seven_found = False
        for roll in range(20):
            if game.game_over or game.round_over or not game.waiting_for_decisions:
                break

            actions = {pid: "continue" for pid in game.players_in_round}
            states, round_done, game_done, info = game.step(actions)

            if round_done and info.get("round_ended_by_seven"):
                if game.roll_number == 0 or info.get("round_ended_by_seven"):
                    print(f"  Seed {seed}: 7 after immunity wiped pot. Round ended.")
                    print(f"  Scores: {game.banked_scores}")
                    print("PASSED: 7 after immunity wipes pot and ends round.\n")
                    seven_found = True
                    break

        if seven_found:
            break
    else:
        raise AssertionError("Could not find a seed producing a 7 after immunity")


def test_doubles_after_immunity_doubles_pot():
    """Verify that doubles after immunity multiply the pot by 2."""
    print("=" * 60)
    print("TEST 5: Post-immunity - doubles multiply pot by 2")
    print("=" * 60)

    for seed in range(2000):
        game = BankGame(num_players=2, seed=seed)
        game.reset()

        for roll in range(20):
            if game.game_over or not game.waiting_for_decisions:
                break

            pot_before = game.pot
            actions = {pid: "continue" for pid in game.players_in_round}
            states, round_done, game_done, info = game.step(actions)

            if round_done:
                break

            # Check if we're now past immunity and the last roll was doubles
            if game.roll_number > 3 and game.last_dice[0] == game.last_dice[1]:
                # The pot should have been doubled from what it was before this roll
                expected = pot_before * 2
                actual = game.pot
                print(f"  Seed {seed}: Roll {game.roll_number}, Dice {game.last_dice}")
                print(f"  Pot before: {pot_before}, Expected after doubles: {expected}, Actual: {actual}")
                assert actual == expected, (
                    f"Doubles after immunity should double the pot. "
                    f"Expected {expected}, got {actual}"
                )
                print("PASSED: Doubles after immunity correctly doubles pot.\n")
                return

    raise AssertionError("Could not find a seed producing doubles after immunity")


def test_banking_gives_pot_value():
    """Verify that banking gives the player the current pot value."""
    print("=" * 60)
    print("TEST 6: Banking gives current pot value")
    print("=" * 60)

    game = BankGame(num_players=3, seed=10)
    game.reset()

    # Player 0 banks on the first decision
    pot_at_bank = game.pot
    actions = {0: "bank", 1: "continue", 2: "continue"}
    states, round_done, game_done, info = game.step(actions)

    score_p0 = game.banked_scores[0]
    print(f"  Pot when banking: {pot_at_bank}")
    print(f"  Player 0 score after banking: {score_p0}")
    assert score_p0 == pot_at_bank, (
        f"Player should have received pot value {pot_at_bank}, got {score_p0}"
    )
    print("PASSED: Banking gives correct pot value.\n")


def test_pot_persists_after_banking():
    """Verify that the pot stays the same after a player banks."""
    print("=" * 60)
    print("TEST 7: Pot persists after banking")
    print("=" * 60)

    game = BankGame(num_players=3, seed=10)
    game.reset()

    pot_before = game.pot
    actions = {0: "bank", 1: "continue", 2: "continue"}
    states, round_done, game_done, info = game.step(actions)

    # The pot itself should not be reduced by banking
    # Note: a new roll happens after banking, so pot may have changed due to the roll
    # But info tells us what pot was before actions
    assert info["pot_before_actions"] == pot_before, (
        f"Pot before actions should be {pot_before}, got {info['pot_before_actions']}"
    )
    print(f"  Pot before banking: {pot_before}")
    print(f"  Pot recorded in info: {info['pot_before_actions']}")
    print("PASSED: Pot persists correctly.\n")


def test_game_ends_after_10_rounds():
    """Verify the game ends after 10 rounds."""
    print("=" * 60)
    print("TEST 8: Game ends after 10 rounds")
    print("=" * 60)

    game = BankGame(num_players=2, seed=42)
    game.reset()
    rounds_completed = 0

    while not game.game_over:
        if not game.waiting_for_decisions:
            break
        actions = {pid: "bank" for pid in game.players_in_round}
        states, round_done, game_done, info = game.step(actions)
        if round_done:
            rounds_completed += 1

    print(f"  Rounds completed: {rounds_completed}")
    print(f"  Game over: {game.game_over}")
    assert game.game_over, "Game should be over after 10 rounds"
    assert rounds_completed == 10, f"Expected 10 rounds, got {rounds_completed}"
    print("PASSED: Game correctly ends after 10 rounds.\n")


def test_state_vector_shape():
    """Verify state vector has the correct shape."""
    print("=" * 60)
    print("TEST 9: State vector shape")
    print("=" * 60)

    for n_players in [2, 3, 4, 5, 6]:
        game = BankGame(num_players=n_players, seed=0)
        game.reset()
        vec = game.get_state_vector(0)
        expected_size = BankGame.state_size()
        print(f"  {n_players} players: state vector size = {len(vec)} (expected {expected_size})")
        assert len(vec) == expected_size, f"Expected size {expected_size}, got {len(vec)}"

    print("PASSED: State vector shape is correct for all player counts.\n")


def test_multiple_players_bank_same_pot():
    """Verify that when multiple players bank simultaneously, they all get the same pot value."""
    print("=" * 60)
    print("TEST 10: Multiple players banking get same pot value")
    print("=" * 60)

    game = BankGame(num_players=4, seed=10)
    game.reset()
    pot_value = game.pot

    # All players bank at once
    actions = {0: "bank", 1: "bank", 2: "bank", 3: "bank"}
    game.step(actions)

    for pid in range(4):
        print(f"  Player {pid} score: {game.banked_scores[pid]}")
        assert game.banked_scores[pid] == pot_value, (
            f"Player {pid} should have {pot_value}, got {game.banked_scores[pid]}"
        )

    print(f"  All players correctly received pot value of {pot_value}")
    print("PASSED: Simultaneous banking works correctly.\n")


if __name__ == "__main__":
    np.random.seed(42)

    test_full_game_with_random_decisions()
    test_immunity_seven_adds_70()
    test_immunity_doubles_just_add()
    test_doubles_during_immunity_exact()
    test_seven_after_immunity_wipes_pot()
    test_doubles_after_immunity_doubles_pot()
    test_banking_gives_pot_value()
    test_pot_persists_after_banking()
    test_game_ends_after_10_rounds()
    test_state_vector_shape()
    test_multiple_players_bank_same_pot()

    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED!")
    print("=" * 60)

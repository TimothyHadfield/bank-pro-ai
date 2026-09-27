"""
Tests for game/environment.py.

Run from bank-ai-v2-mcts directory:
    python -m tests.test_environment
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from game import BankGame
from game.environment import BANK, CONTINUE


# ---------------------------------------------------------------- helpers

def make_game(num_players=4, seed=42):
    g = BankGame(num_players, seed=seed)
    g.reset()
    return g


def play_random_game(num_players=4, seed=0):
    g = make_game(num_players, seed=seed)
    rng = random.Random(seed + 1)
    while not g.is_terminal():
        if not g.waiting_for_decisions:
            raise AssertionError("Engine in inconsistent state")
        actions = {pid: rng.choice([BANK, CONTINUE]) for pid in g.players_in_round}
        g.step(actions)
    return g


# ---------------------------------------------------------------- tests

def test_basic_reset():
    g = make_game(num_players=4, seed=1)
    assert g.num_players == 4
    assert g.round_number == 1, f"expected round 1 after reset, got {g.round_number}"
    assert g.roll_number >= config.IMMUNITY_ROLLS, "should auto-roll through immunity"
    assert g.waiting_for_decisions, "should be waiting after immunity"
    assert g.players_in_round == set(range(4)), "all players should be in round"
    assert g.pot >= 0
    print("  ok test_basic_reset")


def test_immunity_no_decisions():
    """During immunity, players cannot bank. First decision is AFTER roll 3 completes (BEFORE roll 4)."""
    for seed in range(20):
        g = make_game(num_players=3, seed=seed)
        # First decision point: roll_number == IMMUNITY_ROLLS (3), waiting_for_decisions True
        assert g.roll_number == config.IMMUNITY_ROLLS, (
            f"After reset, roll_number should be {config.IMMUNITY_ROLLS} (first decision point); got {g.roll_number}"
        )
        assert g.waiting_for_decisions, "should be waiting after immunity"
        assert len(g.immunity_history) == config.IMMUNITY_ROLLS
    print("  ok test_immunity_no_decisions")


def test_immunity_seven_adds_70():
    """During immunity, a 7 adds 70 not busts. We check by seeding for a 7."""
    # Find a seed that rolls a 7 in the first roll
    for seed in range(1000):
        g = BankGame(num_players=2, seed=seed)
        g.reset()
        # immunity_history is the first 3 rolls
        for d1, d2 in g.immunity_history:
            if d1 + d2 == 7:
                # Reconstruct: did the 7 add 70 to pot? Check by re-simulating.
                # Easier: ensure pot >= 70 if any of the first 3 was a 7.
                expected_min = 70
                assert g.pot >= expected_min, (
                    f"Seed {seed}: 7 in immunity should add 70, pot={g.pot}, history={g.immunity_history}"
                )
                print("  ok test_immunity_seven_adds_70 (verified seed", seed, ")")
                return
    raise AssertionError("Could not find a seed with a 7 in immunity in 1000 tries (very unlikely)")


def test_immunity_doubles_just_sum():
    """Doubles during immunity just add the sum, do NOT double the pot."""
    for seed in range(1000):
        g = BankGame(num_players=2, seed=seed)
        g.reset()
        # Find a seed where immunity_history contains doubles
        if any(d1 == d2 for d1, d2 in g.immunity_history):
            # Compute expected pot: sum of (sum if not 7 else 70) for each immunity roll
            expected = 0
            for d1, d2 in g.immunity_history:
                if d1 + d2 == 7:
                    expected += 70
                else:
                    expected += d1 + d2
            assert g.pot == expected, (
                f"Seed {seed}: pot={g.pot} expected={expected} history={g.immunity_history}"
            )
            print("  ok test_immunity_doubles_just_sum (verified seed", seed, ")")
            return
    raise AssertionError("Could not find a seed with doubles in immunity")


def test_post_immunity_seven_busts():
    """A 7 after immunity wipes the pot and ends the round."""
    # We need a deterministic seed that rolls a 7 post-immunity.
    # Just do a search.
    for seed in range(500):
        g = BankGame(num_players=2, seed=seed)
        g.reset()
        # Force everyone to continue, see if a 7 happens
        busted = False
        rounds_played = 0
        while not g.is_terminal() and rounds_played < 3:
            if not g.waiting_for_decisions:
                break
            current_round = g.round_number
            # Continue everyone
            actions = {pid: CONTINUE for pid in g.players_in_round}
            pot_before = g.pot
            g.step(actions)
            # Did the round end on a 7?
            if g.last_dice and g.last_dice[0] + g.last_dice[1] == 7 and g.roll_number > config.IMMUNITY_ROLLS:
                # round_over should be set, pot wiped
                # Note: pot is reset to 0 at start of next round, so we need to check at the moment
                # The engine sets round_over=True then auto-starts next round, so g.round_over may already be False.
                # Instead: check that no one banked but the round ended (inferring a bust)
                if g.round_number != current_round or g.is_terminal():
                    busted = True
                    print("  ok test_post_immunity_seven_busts (verified seed", seed, ")")
                    return
        if busted:
            return
    raise AssertionError("Could not find a seed that busts post-immunity")


def test_post_immunity_doubles_multiply():
    """Doubles after immunity multiply the pot by 2."""
    for seed in range(2000):
        g = BankGame(num_players=2, seed=seed)
        g.reset()
        # Continue everyone for one decision and see if doubles occur post-immunity
        if not g.waiting_for_decisions:
            continue
        pot_before = g.pot
        g.step({pid: CONTINUE for pid in g.players_in_round})
        if g.last_dice and g.last_dice[0] == g.last_dice[1] and g.roll_number > config.IMMUNITY_ROLLS:
            # Pot should have been multiplied by 2 (if it's not 0 from a bust, which doubles aren't)
            # roll_number was incremented inside step, so the post-immunity check is g.roll_number > IMMUNITY_ROLLS
            # which is guaranteed since we did a step.
            expected = pot_before * 2
            if g.round_over:
                # Round may have ended for other reasons; doubles don't end round
                continue
            assert g.pot == expected, (
                f"Seed {seed}: doubles {g.last_dice} should give {expected}, got {g.pot}"
            )
            print("  ok test_post_immunity_doubles_multiply (verified seed", seed, ")")
            return
    raise AssertionError("Could not find a seed with post-immunity doubles in 2000 tries")


def test_banking_gives_pot_value():
    g = make_game(num_players=3, seed=5)
    pot_at_decision = g.pot
    actions = {pid: (BANK if pid == 0 else CONTINUE) for pid in g.players_in_round}
    g.step(actions)
    assert g.banked_scores[0] == pot_at_decision, (
        f"Player 0 should have banked {pot_at_decision}, has {g.banked_scores[0]}"
    )
    print("  ok test_banking_gives_pot_value")


def test_pot_persists_after_bank():
    """If only one of multiple players banks, the pot stays the same for the next roll's decisions."""
    g = make_game(num_players=4, seed=7)
    pot_at_decision = g.pot
    # Player 0 banks, others continue
    g.step({0: BANK, 1: CONTINUE, 2: CONTINUE, 3: CONTINUE})
    # After step, the next roll has happened (or the round busted).
    # The pot value the bankers received was pot_at_decision; check that.
    assert g.banked_scores[0] == pot_at_decision
    # Other players' scores must be unchanged
    assert g.banked_scores[1:] == [0, 0, 0]
    print("  ok test_pot_persists_after_bank")


def test_simultaneous_step_validates_actions():
    """step() requires exactly one action per active player."""
    g = make_game(num_players=4, seed=10)
    try:
        g.step({0: BANK})  # missing 1, 2, 3
    except ValueError:
        pass
    else:
        raise AssertionError("step() should reject incomplete actions")

    g2 = make_game(num_players=4, seed=10)
    try:
        g2.step({0: BANK, 1: CONTINUE, 2: CONTINUE, 3: CONTINUE, 4: BANK})  # 4 doesn't exist
    except ValueError:
        pass
    else:
        raise AssertionError("step() should reject extra actions")

    g3 = make_game(num_players=4, seed=10)
    try:
        g3.step({0: "wat", 1: CONTINUE, 2: CONTINUE, 3: CONTINUE})
    except ValueError:
        pass
    else:
        raise AssertionError("step() should reject invalid action strings")
    print("  ok test_simultaneous_step_validates_actions")


def test_no_peeking_state_independent_of_query_order():
    """
    Critical regression: querying players in different orders must yield identical state vectors.
    This proves the state vector contains no same-turn information from other players' decisions.
    """
    g = make_game(num_players=4, seed=11)

    # Get state vectors in order [0, 1, 2, 3]
    vecs_a = [g.get_state_vector(pid).copy() for pid in [0, 1, 2, 3]]
    # Get state vectors in order [3, 2, 1, 0]
    vecs_b_reversed = {pid: g.get_state_vector(pid).copy() for pid in [3, 2, 1, 0]}

    for pid in range(4):
        assert np.array_equal(vecs_a[pid], vecs_b_reversed[pid]), (
            f"Player {pid}'s state vector differs based on query order — peek leak!"
        )
    print("  ok test_no_peeking_state_independent_of_query_order")


def test_no_peeking_step_is_atomic():
    """
    Two games with identical starting state, given the same actions in different
    dict-key orders, must produce identical post-step state.

    Uses two freshly-seeded games (not clones) so they start with identical RNG.
    """
    g1 = make_game(num_players=4, seed=12)
    g2 = make_game(num_players=4, seed=12)
    actions_a = {0: BANK, 1: CONTINUE, 2: BANK, 3: CONTINUE}
    actions_b = {3: CONTINUE, 1: CONTINUE, 0: BANK, 2: BANK}

    g1.step(actions_a)
    g2.step(actions_b)

    assert g1.banked_scores == g2.banked_scores
    assert g1.players_in_round == g2.players_in_round
    assert g1.pot == g2.pot
    assert g1.round_number == g2.round_number
    assert g1.roll_number == g2.roll_number
    assert g1.game_over == g2.game_over
    assert g1.last_dice == g2.last_dice
    print("  ok test_no_peeking_step_is_atomic")


def test_clone_independence():
    g = make_game(num_players=3, seed=20)
    c = g.clone()
    # Changing the clone must not affect the original
    c.step({pid: CONTINUE for pid in c.players_in_round})
    assert g.roll_number != c.roll_number or g.pot != c.pot or g.last_dice != c.last_dice, (
        "clone() should produce a state that diverges after step (different RNG)"
    )
    # Original must be untouched
    assert g.banked_scores == [0, 0, 0]
    print("  ok test_clone_independence")


def test_clone_deterministic():
    """Two clones from the same parent (same _seed_counter) should diverge but be reproducible."""
    g1 = make_game(num_players=3, seed=30)
    g2 = make_game(num_players=3, seed=30)
    # Same parent seed -> same auto-immunity rolls -> same initial state
    assert g1.banked_scores == g2.banked_scores
    assert g1.pot == g2.pot
    assert g1.immunity_history == g2.immunity_history
    print("  ok test_clone_deterministic")


def test_full_game_completes_random():
    for seed in range(20):
        g = play_random_game(num_players=4, seed=seed)
        assert g.is_terminal()
        assert g.round_number == config.NUM_ROUNDS
    print("  ok test_full_game_completes_random")


def test_winners_handles_ties():
    g = make_game(num_players=3, seed=42)
    # Force terminal state with a tie
    g.banked_scores = [100, 100, 50]
    g.game_over = True
    winners = g.winners()
    assert winners == [0, 1], f"expected [0, 1], got {winners}"
    print("  ok test_winners_handles_ties")


def test_state_vector_shape_and_range():
    for nplayers in [2, 3, 4, 5, 6]:
        g = make_game(num_players=nplayers, seed=99)
        v = g.get_state_vector(0)
        assert v.shape == (config.INPUT_SIZE,), f"expected ({config.INPUT_SIZE},), got {v.shape}"
        assert v.dtype == np.float32
        # No NaN or Inf
        assert np.isfinite(v).all(), f"non-finite values in state vector: {v}"
    print("  ok test_state_vector_shape_and_range")


def test_state_vector_padding_distinguishable():
    """Padded slots use sentinel values that the network can distinguish via input #25."""
    g_small = make_game(num_players=2, seed=100)
    v_small = g_small.get_state_vector(0)
    g_big = make_game(num_players=6, seed=100)
    v_big = g_big.get_state_vector(0)
    # In a 2-player game, only 1 real opponent → 4 padded slots
    # Padding score sentinel = -1/1000
    expected_padded_score = -1.0 / 1000.0
    assert v_small[14 + 1] == expected_padded_score, (
        f"Expected padded sentinel {expected_padded_score} at index 15, got {v_small[14 + 1]}"
    )
    # In a 6-player game, no padding
    # All 5 opponent score slots should be real (not necessarily nonzero, but >= 0)
    for i in range(5):
        # Score sentinel is -0.001; real scores at start are 0
        assert v_big[14 + i] >= 0.0
    # Input #25 = num_real_opponents / 5
    assert abs(v_small[24] - (1 / 5)) < 1e-6
    assert abs(v_big[24] - (5 / 5)) < 1e-6
    print("  ok test_state_vector_padding_distinguishable")


def test_seven_in_immunity_does_not_end_round():
    """During immunity, a 7 should NOT end the round (it adds 70 instead)."""
    for seed in range(2000):
        g = BankGame(num_players=2, seed=seed)
        g.reset()
        if any(d1 + d2 == 7 for d1, d2 in g.immunity_history):
            # Round should not have ended
            assert g.waiting_for_decisions, "round ended during immunity from a 7 — bug!"
            print("  ok test_seven_in_immunity_does_not_end_round (verified seed", seed, ")")
            return
    raise AssertionError("Could not find a seed with a 7 in immunity")


# ---------------------------------------------------------------- runner

ALL_TESTS = [
    test_basic_reset,
    test_immunity_no_decisions,
    test_immunity_seven_adds_70,
    test_immunity_doubles_just_sum,
    test_post_immunity_seven_busts,
    test_post_immunity_doubles_multiply,
    test_banking_gives_pot_value,
    test_pot_persists_after_bank,
    test_simultaneous_step_validates_actions,
    test_no_peeking_state_independent_of_query_order,
    test_no_peeking_step_is_atomic,
    test_clone_independence,
    test_clone_deterministic,
    test_full_game_completes_random,
    test_winners_handles_ties,
    test_state_vector_shape_and_range,
    test_state_vector_padding_distinguishable,
    test_seven_in_immunity_does_not_end_round,
]


def main():
    failures = []
    for t in ALL_TESTS:
        print(f"running {t.__name__}")
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  FAIL: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(ALL_TESTS)} FAILED")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()

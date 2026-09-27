"""
Phase 8 Test: Verify ai_vs_ai and tournament work correctly.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from config import Config
from game.environment import BankGame
from network.model import BankNetwork
from evaluation.ai_vs_ai import run_match
from evaluation.tournament import (
    AlwaysContinue, ThresholdBot, RandomBot, NetworkStrategy,
    run_tournament, play_match,
)


def test_ai_vs_ai():
    print("=" * 60)
    print("TEST 1: AI vs AI match")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()

    torch.manual_seed(42)
    net1 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    net2 = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    # Make net2 different
    with torch.no_grad():
        for p in net2.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    net1.eval()
    net2.eval()

    results = run_match(net1, net2, config, num_games=100)
    print(f"  Model 1 wins: {results['model1_wins']} ({results['model1_win_rate']:.1%})")
    print(f"  Model 2 wins: {results['model2_wins']} ({results['model2_win_rate']:.1%})")
    print(f"  Ties: {results['ties']}")
    print(f"  Model 1 avg score: {results['model1_avg_score']:.1f}")
    print(f"  Model 2 avg score: {results['model2_avg_score']:.1f}")

    total = results['model1_wins'] + results['model2_wins'] + results['ties']
    assert total == 100, f"Total should be 100, got {total}"
    print("PASSED: AI vs AI match works.\n")


def test_tournament_baselines():
    print("=" * 60)
    print("TEST 2: Tournament with baselines (50 games/matchup)")
    print("=" * 60)

    config = Config()
    strategies = [
        AlwaysContinue(),
        ThresholdBot(100),
        ThresholdBot(200),
        RandomBot(0.3),
    ]

    leaderboard = run_tournament(strategies, config, num_games=50)

    assert len(leaderboard) == 4, "Leaderboard should have 4 entries"
    # AlwaysContinue should generally lose (it never banks, so a 7 wipes everything)
    # ThresholdBot strategies should do reasonably well
    print("PASSED: Tournament runs correctly.\n")


def test_strategy_interface():
    print("=" * 60)
    print("TEST 3: Strategy interface")
    print("=" * 60)

    game = BankGame(num_players=4, seed=42)
    game.reset()
    state = game.get_state(0)

    ac = AlwaysContinue()
    assert ac.decide(state) == "continue"
    print(f"  AlwaysContinue: {ac.decide(state)}")

    tb = ThresholdBot(5)
    result = tb.decide(state)
    expected = "bank" if state["current_pot"] > 5 else "continue"
    assert result == expected, f"ThresholdBot(5) should return {expected}, got {result}"
    print(f"  ThresholdBot(5) with pot={state['current_pot']}: {result}")

    rb = RandomBot(1.0)
    assert rb.decide(state) == "bank"
    print(f"  RandomBot(1.0): {rb.decide(state)}")

    rb0 = RandomBot(0.0)
    assert rb0.decide(state) == "continue"
    print(f"  RandomBot(0.0): {rb0.decide(state)}")

    print("PASSED: Strategy interface works.\n")


def test_network_strategy():
    print("=" * 60)
    print("TEST 4: NetworkStrategy wrapper")
    print("=" * 60)

    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)
    ns = NetworkStrategy(network, name="TestNet")

    game = BankGame(num_players=4, seed=42)
    game.reset()
    state = game.get_state(0)

    decision = ns.decide(state)
    assert decision in ("bank", "continue")
    print(f"  NetworkStrategy decision: {decision}")
    print("PASSED: NetworkStrategy wrapper works.\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    test_ai_vs_ai()
    test_tournament_baselines()
    test_strategy_interface()
    test_network_strategy()

    print("=" * 60)
    print("ALL PHASE 8 TESTS PASSED!")
    print("=" * 60)

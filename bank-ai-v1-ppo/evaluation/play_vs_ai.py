"""
Interactive terminal game: Human vs AI opponents.

Usage:
    python -m evaluation.play_vs_ai --checkpoint models/checkpoint_1000.pt --num_ai 3
"""
import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config
from game.environment import BankGame
from network.model import BankNetwork


def display_state(game, round_num, roll_num, num_ai):
    total_players = 1 + num_ai
    immunity = "YES" if roll_num <= 3 else "NO"

    print()
    print("-" * 50)
    print(f"  Round: {round_num}/10    Roll: {roll_num}    Immunity: {immunity}")
    print(f"  Dice: {game.last_dice[0]} + {game.last_dice[1]} = {sum(game.last_dice)}")
    d1, d2 = game.last_dice
    if d1 == d2:
        print(f"  ** DOUBLES! **")
    if d1 + d2 == 7:
        print(f"  ** SEVEN! **")
    print(f"  Pot: {game.pot}")
    print()
    print("  Scores:")
    print(f"    You (Player 0): {game.banked_scores[0]}")
    for i in range(1, total_players):
        status = "in" if i in game.players_in_round else "banked"
        print(f"    AI {i}: {game.banked_scores[i]}  [{status}]")
    human_status = "in" if 0 in game.players_in_round else "banked"
    print(f"  Your status: {human_status}")
    print(f"  Players still in: {len(game.players_in_round)}")
    print("-" * 50)


def get_human_action():
    while True:
        choice = input("\n  Your decision — (b)ank or (c)ontinue? ").strip().lower()
        if choice in ("b", "bank"):
            return "bank"
        elif choice in ("c", "continue"):
            return "continue"
        else:
            print("  Please type 'b' (bank) or 'c' (continue).")


def main():
    parser = argparse.ArgumentParser(description="Play Bank against AI opponents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_ai", type=int, default=3, help="Number of AI opponents (1-5)")
    args = parser.parse_args()

    if not 1 <= args.num_ai <= 5:
        print("Error: num_ai must be between 1 and 5")
        sys.exit(1)

    config = Config()
    num_players = 1 + args.num_ai

    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()

    batch_num = checkpoint.get("batch_num", "?")
    print(f"\nLoaded checkpoint: {args.checkpoint} (batch {batch_num})")
    print(f"Playing against {args.num_ai} AI opponent(s)")
    print(f"You are Player 0. AI players are 1-{args.num_ai}.")
    print()

    game = BankGame(num_players=num_players, num_rounds=config.num_rounds)
    game.reset()
    current_round = 1

    while not game.game_over:
        if not game.waiting_for_decisions:
            break

        state = game.get_state(0)
        round_num = state["round_number"]

        if round_num != current_round:
            print(f"\n{'='*50}")
            print(f"  ROUND {round_num}")
            print(f"{'='*50}")
            current_round = round_num

        display_state(game, round_num, state["roll_number"], args.num_ai)

        actions = {}
        if 0 in game.players_in_round:
            actions[0] = get_human_action()

        for pid in game.players_in_round:
            if pid == 0:
                continue
            state_vec = game.get_state_vector(pid)
            with torch.no_grad():
                action, _, _ = network.select_action(state_vec, deterministic=False)
            actions[pid] = "bank" if action == 0 else "continue"

        print("\n  Decisions:")
        for pid in sorted(actions.keys()):
            label = "You" if pid == 0 else f"AI {pid}"
            print(f"    {label}: {actions[pid].upper()}")

        states, round_done, game_done, info = game.step(actions)

        if info["banked_players"]:
            for pid in info["banked_players"]:
                label = "You" if pid == 0 else f"AI {pid}"
                print(f"  >> {label} banked {info['pot_before_actions']} points!")

        if info.get("round_ended_by_seven"):
            print(f"\n  !!! A 7 was rolled! Pot wiped! Round over! !!!")
            remaining = [pid for pid in range(num_players) if pid not in info["banked_players"]]
            for pid in remaining:
                label = "You" if pid == 0 else f"AI {pid}"
                print(f"  >> {label} gets nothing this round.")

        if round_done:
            print(f"\n  Round ended. Current scores:")
            for pid in range(num_players):
                label = "You" if pid == 0 else f"AI {pid}"
                print(f"    {label}: {game.banked_scores[pid]}")

    print(f"\n{'='*50}")
    print(f"  GAME OVER!")
    print(f"{'='*50}")
    print(f"\n  Final Scores:")
    for pid in range(num_players):
        label = "You" if pid == 0 else f"AI {pid}"
        print(f"    {label}: {game.banked_scores[pid]}")

    winners = game.get_winners()
    if 0 in winners and len(winners) == 1:
        print(f"\n  YOU WIN!")
    elif 0 in winners:
        winner_names = ["You" if w == 0 else f"AI {w}" for w in winners]
        print(f"\n  TIE! Winners: {', '.join(winner_names)}")
    else:
        winner_names = [f"AI {w}" for w in winners]
        print(f"\n  You lost. Winner(s): {', '.join(winner_names)}")
    print()


if __name__ == "__main__":
    main()

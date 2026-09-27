"""
Export a trained checkpoint's policy network weights to JSON for the browser-based player.

Usage:
    python export_weights.py models/checkpoint_1000.pt
    python export_weights.py   (uses latest checkpoint or creates random weights)

Outputs: play/weights.json
"""
import os
import sys
import json
import glob
import torch

sys.path.insert(0, ".")
from config import Config
from game.environment import BankGame
from network.model import BankNetwork


def export(checkpoint_path=None):
    config = Config()
    state_size = BankGame.state_size()
    network = BankNetwork(state_size=state_size, hidden_size=config.hidden_size)

    batch_num = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        network.load_state_dict(ckpt["network_state_dict"])
        batch_num = ckpt.get("batch_num", 0)
        print(f"Loaded checkpoint: {checkpoint_path} (batch {batch_num})")
    else:
        print("No checkpoint specified — exporting random weights.")

    # Export only the policy network (that's all the browser needs)
    weights = {}
    for name, param in network.policy.named_parameters():
        weights[name] = param.detach().numpy().tolist()

    data = {
        "batch_num": batch_num,
        "hidden_size": config.hidden_size,
        "state_size": state_size,
        "weights": weights,
    }

    out_path = os.path.join("play", "weights.json")
    with open(out_path, "w") as f:
        json.dump(data, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        export(sys.argv[1])
    else:
        pts = sorted(glob.glob("models/*.pt"))
        if pts:
            export(pts[-1])
        else:
            export(None)

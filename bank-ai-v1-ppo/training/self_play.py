"""
Training module: AI (Player 0) plays against bot opponents.
Includes a frozen-self bot that snapshots the AI periodically.
"""
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from game.environment import BankGame
from network.model import BankNetwork, PolicyNetwork
from config import Config
import copy


# ==================== BOT OPPONENTS ====================

class BotOpponent:
    def decide(self, game, pid):
        raise NotImplementedError


class ThresholdBot(BotOpponent):
    def __init__(self, threshold):
        self.threshold = threshold
    def decide(self, game, pid):
        return "bank" if game.pot > self.threshold else "continue"


class RoundAwareBot(BotOpponent):
    def __init__(self, base_threshold=400):
        self.base = base_threshold
    def decide(self, game, pid):
        rounds_left = game.num_rounds - game.round_number
        threshold = self.base * (0.3 + 0.7 * rounds_left / game.num_rounds)
        return "bank" if game.pot > threshold else "continue"


class AggressiveBot(BotOpponent):
    def decide(self, game, pid):
        return "bank" if game.pot > 800 else "continue"


class RandomBot(BotOpponent):
    def __init__(self, p=0.25):
        self.p = p
    def decide(self, game, pid):
        return "bank" if np.random.random() < self.p else "continue"


class PositionalBot(BotOpponent):
    def decide(self, game, pid):
        my_score = game.banked_scores[pid]
        max_opp = max(game.banked_scores[i] for i in range(game.num_players) if i != pid)
        if my_score > max_opp:
            return "bank" if game.pot > 150 else "continue"
        else:
            gap = max_opp - my_score
            threshold = max(200, gap * 0.5)
            return "bank" if game.pot > threshold else "continue"


class FrozenSelfBot(BotOpponent):
    """Uses a frozen snapshot of the AI's policy network."""
    def __init__(self, policy_state_dict, state_size, hidden_size):
        self.policy = PolicyNetwork(state_size, hidden_size)
        self.policy.load_state_dict(policy_state_dict)
        self.policy.eval()

    def decide(self, game, pid):
        sv = game.get_state_vector(pid)
        with torch.no_grad():
            logits = self.policy(torch.FloatTensor(sv).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            # Sample (not deterministic) so frozen copies play diversely
            action = torch.multinomial(probs, 1).item()
        return "bank" if action == 0 else "continue"


# Challenging bot pool — no trivially easy bots
BOT_POOL = [
    ThresholdBot(200),
    ThresholdBot(300),
    ThresholdBot(400),
    ThresholdBot(500),
    RoundAwareBot(300),
    RoundAwareBot(400),
    RoundAwareBot(500),
    AggressiveBot(),
    RandomBot(0.3),
    PositionalBot(),
]

# Frozen self bots get added dynamically during training
_frozen_bots = []
_last_snapshot_batch = 0
SNAPSHOT_INTERVAL = 200  # Snapshot the AI every 200 batches


def update_frozen_bots(network, batch_num, config):
    """Periodically snapshot the current AI as a frozen opponent."""
    global _last_snapshot_batch
    if batch_num - _last_snapshot_batch >= SNAPSHOT_INTERVAL:
        state_dict = copy.deepcopy(network.policy.state_dict())
        bot = FrozenSelfBot(state_dict, BankGame.state_size(), config.hidden_size)
        _frozen_bots.append(bot)
        # Keep at most 5 frozen snapshots
        if len(_frozen_bots) > 5:
            _frozen_bots.pop(0)
        _last_snapshot_batch = batch_num


def _get_opponent_pool():
    """Return the full pool including any frozen self-play bots."""
    return BOT_POOL + _frozen_bots


def _compute_reward(final_scores, player_id, num_players):
    """Win = +1.0, everything else = -1.0. Ties for 1st share +1.0."""
    max_score = max(final_scores[i] for i in range(num_players))
    if final_scores[player_id] == max_score:
        return 1.0
    return -1.0


def run_self_play(network, config: Config, num_games: int = None):
    """
    Run a batch of games: AI (Player 0) vs randomly selected bot opponents.
    Only collects experience for Player 0 (the AI).
    """
    if num_games is None:
        num_games = config.batch_size

    network.eval()
    pool = _get_opponent_pool()

    game_states = [[] for _ in range(num_games)]
    game_actions = [[] for _ in range(num_games)]
    game_log_probs = [[] for _ in range(num_games)]
    game_values = [[] for _ in range(num_games)]

    action_counts = {0: 0, 1: 0}

    games = []
    game_bots = []
    for _ in range(num_games):
        game = BankGame(num_players=config.num_players, num_rounds=config.num_rounds)
        game.reset()
        games.append(game)
        bots = [pool[np.random.randint(len(pool))] for _ in range(config.num_players - 1)]
        game_bots.append(bots)

    active = [True] * num_games

    with torch.no_grad():
        while any(active):
            ai_indices = []
            ai_states = []

            for gi in range(num_games):
                if not active[gi]:
                    continue
                game = games[gi]
                if not game.waiting_for_decisions:
                    active[gi] = False
                    continue
                if 0 in game.players_in_round:
                    state_vec = game.get_state_vector(0)
                    ai_indices.append(gi)
                    ai_states.append(state_vec)

            ai_actions_map = {}
            if ai_states:
                states_tensor = torch.FloatTensor(np.array(ai_states))
                policy_logits = network.policy(states_tensor)
                values = network.value(states_tensor).squeeze(-1)

                probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(probs)
                sampled = dist.sample()
                log_probs = dist.log_prob(sampled)

                for idx, gi in enumerate(ai_indices):
                    action = sampled[idx].item()
                    lp = log_probs[idx].item()
                    val = values[idx].item()

                    game_states[gi].append(ai_states[idx])
                    game_actions[gi].append(action)
                    game_log_probs[gi].append(lp)
                    game_values[gi].append(val)
                    action_counts[action] += 1

                    ai_actions_map[gi] = "bank" if action == 0 else "continue"

            for gi in range(num_games):
                if not active[gi]:
                    continue
                game = games[gi]
                if game.game_over:
                    active[gi] = False
                    continue
                if not game.waiting_for_decisions:
                    active[gi] = False
                    continue

                actions = {}
                for pid in game.players_in_round:
                    if pid == 0:
                        actions[pid] = ai_actions_map.get(gi, "continue")
                    else:
                        bot = game_bots[gi][pid - 1]
                        actions[pid] = bot.decide(game, pid)

                _, _, game_done, _ = game.step(actions)
                if game_done:
                    active[gi] = False

    all_states = []
    all_actions = []
    all_log_probs = []
    all_returns = []
    all_advantages = []
    total_scores = []
    total_rewards = []
    wins = 0

    for gi in range(num_games):
        game = games[gi]
        final_scores = game.get_final_scores()

        reward = _compute_reward(final_scores, 0, config.num_players)

        total_scores.append(float(final_scores[0]))
        total_rewards.append(reward)
        if 0 in game.get_winners():
            wins += 1

        n_steps = len(game_states[gi])
        if n_steps == 0:
            continue

        values = game_values[gi]
        returns = []
        advantages = []

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n_steps)):
            step_reward = reward if t == n_steps - 1 else 0.0
            delta = step_reward + config.gamma * next_value - values[t]
            gae = delta + config.gamma * config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]

        all_states.extend(game_states[gi])
        all_actions.extend(game_actions[gi])
        all_log_probs.extend(game_log_probs[gi])
        all_returns.extend(returns)
        all_advantages.extend(advantages)

    states_tensor = torch.FloatTensor(np.array(all_states))
    actions_tensor = torch.LongTensor(all_actions)
    log_probs_tensor = torch.FloatTensor(all_log_probs)
    returns_tensor = torch.FloatTensor(all_returns)
    advantages_tensor = torch.FloatTensor(all_advantages)

    if len(all_advantages) > 1:
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        if adv_std > 1e-8:
            advantages_tensor = (advantages_tensor - adv_mean) / adv_std
            advantages_tensor = torch.clamp(advantages_tensor, -3.0, 3.0)

    avg_score = float(np.mean(total_scores)) if total_scores else 0.0
    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    win_rate = wins / num_games if num_games > 0 else 0.0

    return {
        "states": states_tensor,
        "actions": actions_tensor,
        "log_probs": log_probs_tensor,
        "returns": returns_tensor,
        "advantages": advantages_tensor,
        "avg_score": avg_score,
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "action_distribution": action_counts,
    }

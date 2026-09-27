# Bank AI v2 — Detailed Implementation Plan

This is the master technical plan for building the AlphaZero-style Bank AI.

**Read first:** `NOTES.md` (article summary), `GAME_RULES.md` (game definition), `ANALYSIS.md` (known risks).

**Goal:** A neural-network agent that plays Bank as close to theoretically optimal as possible, **against any number of opponents from 2 to 6 players**, with the only ceiling being dice luck.

---

## 0. Top-level rules these decisions follow

1. **Variable player count is core, not deferred.** State vector, network, MCTS, self-play, and evaluation all support 2–6 players from day one.
2. **Local-only first.** The AI runs on this laptop. The dashboard runs on this laptop. Multi-device web hosting is out of scope.
3. **Wins are everything.** 1st place = win, 2nd–6th = loss. No partial credit. No score-based reward shaping. Ties for first share the win fractionally.
4. **No player ever sees another's same-turn decision** — enforced in the engine, in MCTS, in self-play, and in the play UI.
5. **Build thin vertical slices, not strict isolated phases.** Get one game running end-to-end with the simplest possible MCTS first; then improve each piece.

---

## 1. High-Level Strategy

We are building three components that work together:

1. **A two-headed neural network** that predicts (a) what action to take and (b) how good a game state is for the player whose turn it is.
2. **Monte Carlo Tree Search (MCTS)** that uses the network as a guide to look ahead at future possibilities.
3. **A self-play training loop** where the AI plays against itself, uses MCTS to improve its decisions, and trains the network to mimic those improved decisions.

**The key insight:** MCTS is *better* than the raw network alone (because it actually thinks ahead). When we record what MCTS does and train the network to copy it, the network becomes stronger. Then MCTS guided by the stronger network is even better. This loop is what makes AlphaZero work.

---

## 2. Game Engine (Module: `game/`)

### 2.1 Core class: `BankGame`

The engine must be **deterministic given a seed**, **clonable** (so MCTS can fork the state to explore branches), and **support 2–6 players** without code changes.

```
class BankGame:
    state:
        num_players (2-6)
        round_number (1-10)
        roll_number (1, 2, 3, ...)
        pot
        banked_scores (list, length num_players)
        players_in_round (set of player IDs still active this round)
        last_dice (d1, d2)
        immunity_history (list of dice rolls during immunity, for state inspection)
        game_over (bool)
        round_over (bool)
        waiting_for_decisions (bool)
        rng (seeded random number generator)

    methods:
        reset() -> initial state
        get_state(player_id) -> dict of public state from that player's perspective
        get_state_vector(player_id) -> numpy array for network input (FIXED SIZE regardless of num_players)
        step(actions: dict[player_id -> "bank"|"continue"]) -> new state
        clone() -> independent deep copy of the game (for MCTS rollouts)
        legal_actions(player_id) -> ["bank", "continue"]
        is_terminal() -> bool
        winners() -> list of player IDs (handles ties)
```

### 2.2 Critical rule: simultaneous decisions

The `step` function MUST collect all actions before processing any of them:

1. Validate that every player still in the round provided an action.
2. Process all bankers simultaneously (everyone gets the same pot value).
3. Remove bankers from `players_in_round`.
4. Roll the next dice (post-immunity rules apply).
5. Return the new state.

**No player's state vector ever includes another player's same-turn action.** Only previous banking is visible.

### 2.3 State vector design (FIXED SIZE for variable players)

The network input must be a **fixed-size vector regardless of how many players are in the game**. This lets one network handle 2–6 players.

We pad the opponent slots to a maximum of 5 (since the calling player is one, and max total is 6).

**Inputs (28 floats total):**

| # | Input | Normalization |
|---|---|---|
| 1 | pot | / 1000 |
| 2 | roll_number | / 10 |
| 3 | round_number | / 10 |
| 4 | my_banked_score | / 1000 |
| 5 | num_players_in_game | / 6 (so 2-player = 0.33, 6-player = 1.0) |
| 6 | players_still_in_round | / 6 |
| 7 | rounds_remaining | / 10 |
| 8 | is_final_round | 0 / 1 |
| 9 | rolls_since_immunity_ended | / 10 |
| 10 | my_rank (1 = leading) | / num_players |
| 11 | gap_to_first | / 1000 |
| 12 | gap_to_last | / 1000 |
| 13 | would_banking_now_make_me_first | 0 / 1 |
| 14 | immunity_active | 0 / 1 |
| 15-19 | opponent_scores (5 slots, padded with -1) | / 1000 |
| 20-24 | opponent_in_round (5 slots, padded with 0) | 0 / 1 |
| 25 | num_real_opponents | / 5 (lets the network distinguish padding from real values) |
| 26-28 | one-hot [I'm leading, I'm in middle, I'm trailing] | 0 / 1 |

**Total: 28 inputs.**

The opponent slots are sorted by score (descending) before being placed in the vector — this gives the network a stable ordering. Padded slots get sentinel values (-1 score, 0 in-round).

### 2.4 Cloning for MCTS

`clone()` must be a deep copy with its own independent RNG. MCTS will fork hundreds of game states per decision — they must not share state.

To make rollouts reproducible inside the tree but vary across rollouts, each cloned game gets a fresh RNG seeded from a global counter (stored on the parent so child seeds are deterministic given the original seed).

### 2.5 Game-rule reference

All rule details live in `GAME_RULES.md`. The engine implements that exactly. Any disagreement = engine bug.

---

## 3. Neural Network (Module: `network/`)

### 3.1 Architecture

A standard AlphaZero-style two-headed network, sized for the small state space:

```
Input (28 floats)
  -> Linear(28 -> 128) -> ReLU
  -> Linear(128 -> 128) -> ReLU
  -> Shared trunk output (128 features)

Two heads from the shared trunk:
  Policy head:
    -> Linear(128 -> 64) -> ReLU
    -> Linear(64 -> 2)
    -> Softmax (gives probability for [bank, continue])

  Value head:
    -> Linear(128 -> 64) -> ReLU
    -> Linear(64 -> 1)
    -> Sigmoid (gives probability that THIS player wins from this state)
```

### 3.2 Why two heads instead of separate networks

The policy head guides MCTS (which branches to explore first). The value head evaluates leaf nodes (whether the position looks good). Sharing a trunk means features useful for prediction also help evaluation. AlphaZero showed this is significantly more parameter-efficient than two separate networks.

### 3.3 Why we don't need an embedding layer

Dominion needed embeddings because they had 500+ unique cards. Bank has zero diverse entities — the state is just numbers. The 28-input flat vector captures everything that matters, including player count.

### 3.4 Forward pass interface

```
class BankNetwork(nn.Module):
    forward(state_vector: Tensor[28]) -> (policy_probs: Tensor[2], value: Tensor[1])
    forward_batch(state_vectors: Tensor[N, 28]) -> (policy_probs: Tensor[N, 2], values: Tensor[N, 1])
```

`forward_batch` is critical — MCTS will call this for many leaf nodes at once.

---

## 4. Monte Carlo Tree Search (Module: `mcts/`)

This is the core of the new approach. It's a tree where each node represents a game state, and we use the network to decide which branches to explore.

### 4.1 Node types

The Bank game tree has THREE node types:

1. **Decision nodes** — *our* player must choose an action. Children: 2 (bank, continue).
2. **Chance nodes** — the dice will roll. Children: dice outcomes (we sample, not enumerate — see 4.7).
3. **Opponent decision nodes** — opponents are deciding simultaneously with us. Children: combinations of opponent actions, sampled from their policies (NOT optimized — see 4.6).

### 4.2 The PUCT formula (selection)

When choosing which child to explore from an OUR-decision node:

```
UCB(child) = Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))

where:
  Q(child) = average value seen from this child
  P(child) = prior probability from policy network
  N(child) = number of visits to this child
  N(parent) = visits to parent
  c_puct = exploration constant (~1.4 to start)
```

Pick the child with highest UCB.

### 4.3 The MCTS loop (one simulation)

```
function simulate(node):
    if node.is_terminal:
        return outcome (1 if our player won, 0.5/k if tied for first with k winners, 0 if lost)

    if node is a chance node:
        sample dice outcome from true probability
        child = node.children[dice_outcome]  # create if missing
        value = simulate(child)

    elif node is an opponent decision node:
        # Sample opponents' actions from THEIR policies — never optimized
        for each opponent still in round:
            policy = network.forward(opponent_state).policy
            opp_action = sample_from_distribution(policy)
        # Apply opponents' actions together with our committed action
        next_game = clone_and_step(node.game, all_actions)
        child = create_or_get_child(node, action_combo)
        value = simulate(child)

    elif node is OUR decision node:
        if node is unexpanded (first visit):
            policy, value_estimate = network.forward(node.state)
            for each action in [bank, continue]:
                node.children[action] = new_node(prior=policy[action])
            return value_estimate  # Use network estimate, don't recurse further

        else:
            best_action = argmax UCB over children
            value = simulate(node.children[best_action])

    # Update statistics on the way back up
    node.N += 1
    node.W += value
    return value
```

### 4.4 Number of simulations per move

Start with **100 simulations per decision**. AlphaZero used 800 for Go, but our action space (2 actions) and state space are far smaller. Tunable in Phase 9.

### 4.5 Action selection from MCTS

After all simulations:

```
visits = [N(child) for child in node.children]

if temperature == 0:
    action = argmax(visits)  # greedy, used during evaluation
else:
    probs = visits ** (1/temperature) / sum
    action = sample_from(probs)  # explore more during early training
```

Why visits and not Q-values? Visits naturally weight by both quality AND confidence. A child with one lucky simulation has Q=1 but only 1 visit; we don't want to pick it.

### 4.6 The simultaneous decision problem (RESOLVED)

**Each player runs their own independent MCTS.** When player A's tree needs to model what other players will do, it samples from their network's policy distribution. It does NOT see what they actually chose.

When the actual game `step()` happens:
1. Player A runs MCTS, gets action A_action
2. Player B runs MCTS, gets action B_action (independently, can't see A_action)
3. (...continue for all players still in the round)
4. Engine processes all actions simultaneously

Inside player A's MCTS tree:
- When expanding a node where opponents act, sample their actions from their policies
- This sampling represents A's belief about what they'll do — it's an approximation, not a peek
- A different sampling each simulation gives MCTS a distribution of possible opponent behaviors

This satisfies the "no peeking" rule because A never sees B's actual decision in either the real game or the tree. B's decisions in the tree are A's *prediction* of B, not B's actual move.

### 4.7 Chance node handling

When the game tree hits a dice roll (after `continue`), we have a chance node. We use **probability-weighted sampling**:

- Sample one (d1, d2) outcome per visit using true 1/36 probabilities
- Store all outcomes seen so far as children
- When backing up values, weight each child's contribution by its true probability rather than visit fraction (reduces noise from undersampled outcomes)

This is more accurate than naive sampling and cheaper than full 36-way expansion.

### 4.8 Tree reuse across decisions

After we make a move in the real game, we **build a fresh tree for the next decision.** Reasoning:

- Other players also made decisions (we couldn't see which subtree they'd commit to)
- Dice rolls happened (random outcome we couldn't predict)
- The new state is one specific node deep in a tree of possibilities — most often it doesn't exist as a node we can find

Subtree reuse with stochastic + simultaneous play is a pile of edge cases; not worth the complexity. Revisit only if profiling shows tree build is a major cost.

### 4.9 Dirichlet noise at root

To encourage exploration during self-play, add Dirichlet noise to the root prior:

```
adjusted_prior = (1 - epsilon) * raw_prior + epsilon * Dirichlet(alpha)
```

Per the AlphaZero paper, alpha should scale inversely with action space. Bank has 2 actions, so:
- `DIRICHLET_ALPHA = 5.0` (much larger than the chess/Go default of 0.3)
- `DIRICHLET_EPSILON = 0.15` (noise mix, lowered because our tree is shallow)

Apply ONLY during self-play training, NOT during evaluation or play vs human.

---

## 5. Self-Play (Module: `training/self_play.py`)

### 5.1 One game of self-play

```
def self_play_game(network, opponent_pool=None):
    # Variable player count: randomly choose 2-6 each game
    num_players = random.choice([2, 3, 4, 5, 6])

    # Build the seat assignments
    seats = build_seats(network, num_players, opponent_pool)
    # seats[i] is either:
    #   - "self_mcts" (uses current network with MCTS, gets data recorded for training)
    #   - a bot strategy (no recording, just plays the game)

    game = BankGame(num_players)
    game.reset()
    history = []  # (state, mcts_policy, player_id) tuples

    while not game.is_terminal():
        actions = {}
        for pid in game.players_in_round:
            seat = seats[pid]
            if seat == "self_mcts":
                mcts = MCTS(network, root_game=game.clone(), root_player=pid)
                visit_distribution = mcts.run(num_simulations=SIMS_PER_MOVE)
                action = sample_action(visit_distribution, temperature=current_temp)
                actions[pid] = action
                # Record only for our seats
                state_vec = game.get_state_vector(pid)
                history.append((state_vec, visit_distribution, pid))
            else:
                actions[pid] = seat.choose(game.get_state(pid))

        game.step(actions)  # All actions applied simultaneously

    # Game over — assign rewards
    winners = game.winners()
    training_data = []
    for state_vec, policy, pid in history:
        if pid in winners:
            value = 1.0 / len(winners)  # Fractional credit for ties
        else:
            value = 0.0
        training_data.append((state_vec, policy, value))

    return training_data
```

### 5.2 Mixed opponent pool

By default, ~80% of self-play games use `"self_mcts"` for ALL seats (pure self-play).

The remaining ~20% use a mix:
- 10%: 1 of N seats is a bot (other seats are self_mcts)
- 5%: 2 of N seats are bots
- 5%: tournament mix (different older checkpoints in different seats)

Bot pool includes:
- `ThresholdBot(threshold)` for thresholds {100, 150, 200, 250, 300}
- `RandomBot(p_bank)` for p ∈ {0.2, 0.5, 0.8}
- `AlwaysContinue` (rare, mostly an adversarial training case)

This keeps the network calibrated against playstyles other than its own. See `ANALYSIS.md §1` for the rationale.

### 5.3 Variable player count distribution

Per-game player count is sampled uniformly from {2, 3, 4, 5, 6}. We log win rate by player count to make sure no count is silently underperforming.

### 5.4 Temperature schedule

- **First N self-play games** (where N is small, ~500): temperature = 1.0 throughout (full exploration)
- **Steady state:** temperature = 1.0 in early game, drops to 0.0 in last 3 rounds (exploit when stakes are clearest)
- **Evaluation games:** temperature = 0.0 throughout (always greedy)

### 5.5 Replay buffer

Keep the last N self-play decisions in a buffer (start: `REPLAY_BUFFER_SIZE = 50000`). Each training step samples a minibatch.

To handle staleness (see `ANALYSIS.md §7`):
- Tag each entry with the iteration number it was generated in
- Weight sampling probability by recency: newer entries get sampled more often
- Discard entries older than `MAX_BUFFER_AGE = 30 iterations`

---

## 6. Training (Module: `training/trainer.py`)

### 6.1 Loss function

```
loss = policy_loss + value_loss + l2_regularization

policy_loss = -sum(mcts_policy * log(network_policy))   # cross-entropy
value_loss = (network_value - actual_outcome) ** 2      # MSE
l2 = sum(weights ** 2) * 1e-4                           # weight decay
```

The key insight: we train the policy head to match what MCTS did (the visit distribution). MCTS is smarter than the raw policy, so this drives improvement.

### 6.2 Optimizer

Adam with learning rate 1e-3, decreased to 1e-4 after 1000 iterations.

### 6.3 Training loop structure

```
while True:
    # Generate self-play games (in parallel workers if available)
    for _ in range(GAMES_PER_ITERATION):
        data = self_play_game(network, opponent_pool=opponent_pool)
        replay_buffer.extend(data, iteration=current_iter)

    # Train network on buffer samples
    for _ in range(TRAINING_STEPS_PER_ITERATION):
        batch = replay_buffer.sample_weighted_by_recency(batch_size=512)
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()

    # Periodic evaluation
    if iteration % EVAL_INTERVAL == 0:
        results = evaluate_against_old_checkpoints(player_counts=[2,3,4,5,6])
        save_checkpoint()
        log_metrics(results)
```

### 6.4 Iteration sizes (initial guesses)

- Games per iteration: 50
- Training steps per iteration: 100
- Batch size: 512
- One iteration on CPU: best guess **30 seconds to 5 minutes** depending on MCTS speed (we will measure in Phase 1).

---

## 7. Evaluation (Module: `evaluation/`)

### 7.1 What to measure (per evaluation)

For each tested player count (2, 3, 4, 5, 6) separately:
- **Win rate vs threshold bots** at thresholds {100, 150, 200, 250, 300}
- **Win rate vs frozen older checkpoint** (saved every 10 iterations)
- **Win rate vs higher-sim version of itself** (200 sims vs 100 sims — measures whether the policy is yet a perfect proxy for MCTS)
- **MCTS-vs-raw-network agreement** — how often does MCTS pick the same action the raw policy would? Higher means the network is well-trained
- **Average game length** — sanity that the AI isn't degenerating

### 7.2 Pit two AIs against each other

```
def play_match(ai1, ai2, num_games=1000, num_players_dist=[2,3,4,5,6]):
    # Distribute seats fairly: ai1 takes half, ai2 takes half, swap each game
    # Each AI runs MCTS independently with its own network
    # Results bucketed by player count
```

### 7.3 Use higher MCTS sims for evaluation

Self-play uses 100 sims (for speed). Evaluation uses 200–400 (for quality).

### 7.4 What we don't compare against

Per user direction: **NOT** comparing v2 against v1's PPO model. v1 was never converged to a stable state and isn't a meaningful benchmark.

---

## 8. Dashboard (Module: `dashboard/`) — REBUILT FRESH

We are **not** copying v1's dashboard files. We will build this fresh, taking lessons from v1 (what worked, what didn't) but writing all dashboard code clean for v2.

### 8.1 Architecture (same shape as v1, but rewritten)

- Python HTTP server (`dashboard/server.py`) serves the static UI and runs the training subprocess
- `metrics.json` written by the training process, polled by the browser
- API endpoints for start/stop/reset/save/load/play

### 8.2 Tabs (in build order)

Build in this order — each tab works before we start the next:

1. **Training tab** — start/stop/reset training, charts of loss + win rate by player count + visit entropy + games/min, current checkpoint status
2. **Play vs AI tab** — local-only, server-side MCTS (see 8.4)
3. **Game Stats tab** — pure dice probability calculator and visualizer (no AI involved)
4. **Tournament tab** — pit checkpoints against bots and against each other, broken down by player count
5. **Build AI tab** — Python-style code editor for hand-coded strategies (carries the v1 design forward, but rewritten clean)

### 8.3 Metrics tracked (per training iteration)

- Total loss, policy loss, value loss
- Win rate vs each threshold bot, **separately for each player count**
- Win rate vs frozen old checkpoint (e.g. 10 iterations ago), per player count
- MCTS visit entropy (high = exploring, low = converged)
- Games per minute (throughput)
- Replay buffer size + average age
- MCTS-vs-raw-network agreement %

### 8.4 Play vs AI (local only)

The browser sends the game state to the local Python server. The server runs MCTS using the loaded checkpoint and returns the AI's decision.

- Quality mode: 200 sims/move (~1–2 sec latency per AI move)
- Fast mode: 50 sims/move (~0.5 sec)

Multiple humans can play together by being prompted in turn — their decisions are collected privately, then all actions sent to the engine simultaneously. This naturally enforces the no-peeking rule. Human seats and AI seats are interchangeable in the engine.

### 8.5 Web/multi-device deferral

Per user direction: **multi-device hosting is OUT OF SCOPE for v2.** Local-only dashboard. After the AI is good, we revisit hosting.

---

## 9. File Structure

```
bank-ai-v2-mcts/
├── NOTES.md                  # Article notes
├── PLAN.md                   # This file
├── ANALYSIS.md               # Risk analysis
├── GAME_RULES.md             # Authoritative game rules
├── BankAI-v2.bat             # Launcher (eventually)
│
├── config.py                 # All hyperparameters
│
├── game/
│   ├── __init__.py
│   └── environment.py        # BankGame class
│
├── network/
│   ├── __init__.py
│   └── model.py              # Two-headed BankNetwork
│
├── mcts/
│   ├── __init__.py
│   ├── tree.py               # Node class
│   └── search.py             # MCTS algorithm
│
├── training/
│   ├── __init__.py
│   ├── self_play.py          # Game generation (with opponent pool)
│   ├── trainer.py            # Network training
│   ├── replay_buffer.py      # Sample storage with recency weighting
│   ├── opponent_pool.py      # Bot definitions for mixed pool
│   └── watchdog.py           # Halt detection
│
├── evaluation/
│   ├── __init__.py
│   ├── play_vs_ai.py
│   ├── ai_vs_ai.py
│   └── tournament.py
│
├── monitoring/
│   ├── __init__.py
│   └── logger.py             # Writes metrics.json
│
├── dashboard/                # Rebuilt fresh
│   ├── server.py
│   ├── index.html
│   ├── train.js
│   ├── play.js
│   ├── stats.js
│   ├── tournament.js
│   ├── build.js
│   └── style.css
│
├── tests/
│   ├── test_environment.py
│   ├── test_no_peeking.py    # Specific regression for the no-peeking rule
│   ├── test_clone.py
│   ├── test_network.py
│   ├── test_mcts.py
│   └── test_self_play.py
│
├── train.py                  # Training entry point
│
├── models/                   # Saved checkpoints
├── runs/                     # Training run logs
├── saves/                    # Saved training sessions
└── requirements.txt
```

---

## 10. Build Approach: Thin Vertical Slices

Instead of strict-sequential phases, we build **end-to-end thin slices** and grow each piece. Each slice produces a working system; later slices replace placeholders with real implementations.

### Slice 1: Skeleton end-to-end (target: working but dumb)
1. `BankGame` engine with `clone()`, `get_state_vector()`, simultaneous `step()`
2. `BankNetwork` with random weights
3. Trivial MCTS (10 sims, no Dirichlet noise, sampling chance nodes naively)
4. One self-play game runs end-to-end and produces (state, policy, value) data
5. Training step runs on a tiny buffer and updates the network

**Goal:** prove the integration works. The AI plays badly. That's fine.

### Slice 2: Make it correct (target: technically right)
6. Probability-weighted chance node backup
7. Proper Dirichlet noise at root
8. Variable player count: random 2–6 per game
9. Mixed opponent pool (bots in 20% of games)
10. Replay buffer with recency weighting
11. Watchdog and logger
12. Save/load checkpoints

**Goal:** the algorithm matches the plan exactly.

### Slice 3: Make it visible (target: usable dashboard)
13. Dashboard server with start/stop/reset training
14. Training tab with charts
15. Play vs AI tab (server-side MCTS)
16. Tournament tab with per-player-count breakdown
17. Game Stats tab
18. Build AI tab

**Goal:** you can interact with training and the AI through the browser.

### Slice 4: Make it good (target: strong play)
19. Profile and speed up bottlenecks (likely: parallelize self-play games, batch network forwards)
20. Tune c_puct, sims per move, learning rate, replay buffer size
21. Long training run (overnight, then multiple)
22. Sanity-check evaluation: win rate vs bots should rise over iterations

**Goal:** the AI demonstrably gets better over time.

---

## 11. Hyperparameters (Initial Values)

```
# Game
NUM_PLAYERS_RANGE = (2, 6)       # Sampled per game
NUM_ROUNDS = 10

# Network
HIDDEN_SIZE = 128
INPUT_SIZE = 28

# MCTS
SIMS_PER_MOVE = 100              # During self-play
SIMS_PER_MOVE_EVAL = 300         # During evaluation
C_PUCT = 1.4
DIRICHLET_ALPHA = 5.0            # Adjusted for 2-action space
DIRICHLET_EPSILON = 0.15         # Lowered for shallow tree

# Training
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_AT = 1000    # iterations
LEARNING_RATE_DECAYED = 1e-4
L2_WEIGHT = 1e-4
GAMES_PER_ITERATION = 50
TRAINING_STEPS_PER_ITERATION = 100
REPLAY_BUFFER_SIZE = 50000
MAX_BUFFER_AGE_ITERATIONS = 30

# Self-play opponent mix
PURE_SELF_PLAY_FRACTION = 0.80
ONE_BOT_FRACTION = 0.10
TWO_BOT_FRACTION = 0.05
OLD_CHECKPOINT_MIX_FRACTION = 0.05

# Temperature
TEMPERATURE_INITIAL = 1.0
TEMPERATURE_LATE_GAME = 0.0      # In last 3 rounds, go greedy

# Eval
EVAL_INTERVAL_ITERATIONS = 10
CHECKPOINT_INTERVAL_ITERATIONS = 10
EVAL_GAMES_PER_OPPONENT = 200    # per player count

# Bots in opponent pool
THRESHOLD_BOT_THRESHOLDS = [100, 150, 200, 250, 300]
RANDOM_BOT_PROBABILITIES = [0.2, 0.5, 0.8]
```

---

## 12. Hard Rules (Reminders)

1. **No player ever sees another player's same-turn decision.** Engine collects all actions before processing. State vectors only include past actions. Regression test required.
2. **Opponents in MCTS trees are sampled from policy, never optimized.** They model belief, not collusion.
3. **One MCTS tree per player per decision.** Don't share trees across players or across turns.
4. **Reward is binary win/loss only.** No score-based rewards. Tied for first = fractional reward (1/k for k winners).
5. **Game engine must be deterministic given a seed.** Required for testing and for MCTS rollout reproducibility.
6. **Variable 2–6 players supported in every component.** Engine, network, MCTS, self-play, evaluation, dashboard.
7. **Local-only.** No web hosting in v2 scope.

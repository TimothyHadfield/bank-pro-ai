# Bank AI v2 — MCTS + Neural Network Approach

Notes from studying Temple Gates Games' Dominion AI article and planning how to replicate their approach for the Bank dice game.

Source: https://www.templegatesgames.com/dominion-ai/

---

## Part 1: How Temple Gates Built Their Dominion AI

### The big picture
Their AI is "the first commercial implementation of the techniques behind AlphaZero." It combines three ideas:

1. **Monte Carlo Tree Search (MCTS)** — explores the tree of possible future game states by simulating random playouts
2. **Neural Network** — evaluates how good a game state is (predicts win probability)
3. **Self-play reinforcement learning** — the AI trains by playing against itself, with no human-labeled data

The neural network *guides* the tree search (telling it which branches are worth exploring deeper) and *evaluates* leaf nodes (predicting who wins from there). The tree search produces stronger play than the network alone, and the resulting game outcomes train the network to be even better. This loop is the AlphaZero recipe.

### Network architecture
- **Input layer:** game state — player count, scores, cards in hand, cards on board, etc. Values normalized to 0–1.
- **Embedding layer:** captures the "DNA" of each card (cost, victory points, draw, actions, buys). Compresses thousands of card-specific inputs down to ~10 embedding nodes. *Cards with similar effects end up near each other in embedding space — King's Court ends up close to Festival even though their text is different.*
- **Hidden layer:** standard, size determined by training
- **Output layer:** one probability per player (chance that player wins)

### Decision-making at game time
For each turn:
1. Build a tree of possible actions
2. Use MCTS to focus exploration on promising branches (random walks, pick the highest-value branch, expand from there)
3. Pass leaf states through the network to score them
4. Choose the action that leads to the best predicted outcome
5. Their AI looks "1000 steps into the future" — much deeper than the older Race for the Galaxy AI which was only 1-turn lookahead

### Training
- **Self-play** — AI plays against copies of itself
- **Temporal difference learning** — mid-game predictions get used as feedback signals before the game even ends
- **Terminal feedback** — once the game ends, win/loss adjusts the weights based on how wrong the prediction was

### Why this beats older approaches
- Hand-coded rules don't scale — every new card or expansion adds complexity that becomes "bramblier to debug"
- The Race for the Galaxy AI needed **24 separate networks** for different expansion combinations. Dominion uses **one network** that generalizes thanks to the embedding layer.
- The network learns features beyond what humans specify — it discovers which cards play similar functional roles even when the text is totally different.

### Key results
- The Race for the Galaxy AI hit a 97% win rate against humans
- Dominion AI still being trained at the time of the article

---

## Part 2: Comparing Dominion to Bank

| Aspect | Dominion | Bank |
|---|---|---|
| Game type | Deck-building card game | Dice game |
| State complexity | Huge — 500+ cards, hands, decks, supplies | Tiny — pot, scores, round, who's in |
| Action space | 100s of choices per turn (which card to play, buy, etc.) | **2 choices: bank or continue** |
| Randomness | Card draws | Pure dice rolls |
| Turn structure | Sequential (one player at a time) | **Simultaneous** (all remaining players choose at once) |
| Game length | ~20-30 turns | 10 rounds, ~5-15 decisions per round per player |
| Why embedding layer matters | Critical — too many cards to enumerate | **Useless** — no cards, no diverse entities to embed |
| Search depth meaningful? | Very — long causal chains from card purchases | Limited — only matters for the rest of this round + maybe a couple rounds ahead |
| Hidden information | Yes (opponent's deck) | **No** (everything is public) |

### The critical differences for our purposes

1. **Bank has only 2 actions.** This makes MCTS *easier*, not harder — but also means the branching is mostly driven by **dice randomness**, not opponent choices.

2. **Simultaneous decisions** are tricky for traditional MCTS. AlphaZero assumes turn-based games. We need to handle the fact that all players bank/continue at the same instant. Two options:
   - Treat each player's decision independently, modeling opponents as a fixed strategy or a previous version of the AI
   - Use a "joint action" where the tree branches on all combinations of all players' choices simultaneously (gets exponential)

3. **No embedding layer needed.** The state space is simple enough to feed into the network as a flat vector (similar to v1).

4. **Dice randomness dominates.** Each "step" in the tree isn't a player decision — it's a dice roll outcome (chance node), then players decide. This is a *stochastic game tree* with both decision nodes and chance nodes.

5. **Win signal is noisy.** Even with perfect play, you can lose to bad dice. So the network needs lots of games to filter the signal from the luck.

---

## Part 3: Plan for Bank AI v2 (AlphaZero-style)

### Architecture

**Neural network (much simpler than Dominion's):**
- **Input layer:** Same flat state vector as v1 — pot, round, roll, my score, opponent scores, players in round, etc. (no embedding needed)
- **Hidden layers:** 2 layers of 128 neurons (ReLU)
- **Two output heads:**
  - **Policy head:** probability of `bank` vs `continue` (2 outputs, softmax)
  - **Value head:** predicted win probability for this player from this state (1 output, sigmoid)

This is the standard AlphaZero "two-headed" network. The policy head says *what to do*, the value head says *how good the position is*.

### MCTS (the new part)

The tree alternates between two node types:

- **Decision nodes** — "I'm here, what should I do?" Children are `bank` and `continue`.
- **Chance nodes** — after `continue`, the dice roll. Children are the 36 possible (d1, d2) outcomes (or 11 sums if we approximate).

Each node tracks:
- `N` — visits
- `W` — total value from this node
- `Q = W / N` — average value
- `P` — prior probability from the policy network

Selection uses the standard **PUCT formula:**
```
UCB(child) = Q(child) + c * P(child) * sqrt(N(parent)) / (1 + N(child))
```
This balances exploitation (high Q) with exploration (high P, low N).

**For chance nodes:** instead of selection, we sample the dice outcome from its true probability (since dice are random). This is sometimes called "Expectimax MCTS."

**Handling opponents:** we model their decisions using either:
- Their own neural network (assume they use the same network as us — fair assumption in self-play)
- A frozen older version of our network (provides diverse opposition during training)

### Self-play training loop

```
For each game:
  Reset game
  while game not over:
    For each player still in the round:
      Run MCTS for K simulations from current state
      Get visit counts for each action — these are the "improved policy"
      Sample an action proportional to visit counts (with temperature for exploration)
      Record: (state, improved_policy, player_id)
    Step game with chosen actions
  When game ends:
    For each recorded (state, policy, player_id), set value = 1 if that player won, else 0
    Add to replay buffer

For each training step:
  Sample batch from replay buffer
  Loss = policy_loss (cross-entropy with MCTS visits) + value_loss (MSE with actual outcome)
  Update network
```

The key insight: the MCTS-derived policy is **better** than the raw network's policy (because the tree search corrects mistakes). Training the network to mimic MCTS policy is what drives improvement.

### Specific design decisions to make

1. **MCTS simulations per move:** Start with 50–100. AlphaZero used 800. We can probably get away with less because Bank is simpler.

2. **Tree search depth:** Should we limit how far ahead we look? The whole game is short, so we could simulate to game end every time. But that's expensive. Truncate after ~5 future decisions and use the value network to evaluate the leaf.

3. **Chance node handling:** Either expand all 36 dice outcomes (expensive) or sample one outcome per visit (cheap, classical Expectimax). I'll start with sampling.

4. **Opponent modeling:** Use the same network for opponents during training. During evaluation, use a frozen previous version.

5. **Reward function:** Binary win/lose, like AlphaZero. No score-based reward, no rank-based reward — just "did you finish first." This forces the AI to actually try to win, not maximize score.

6. **Network input:** Same 22-input state vector from v1 should work. No embedding layer needed.

7. **Temperature schedule:** Early in a game, sample actions with high temperature (more exploration). Late in the game, use temperature → 0 (greedy). This is standard AlphaZero.

### What this gets us that v1 (PPO) didn't

- **Lookahead.** PPO is reflexive — it learns "in state X, do Y." MCTS actually thinks about consequences. For a game with bust risk (7 wipes the pot), this should make a huge difference.
- **Higher quality training signal.** PPO's policy loss was tiny because the reward signal was weak. AlphaZero's training target is the MCTS visit distribution, which is denser and more informative.
- **Better play out of the box.** Even an untrained network plus MCTS can play decently because the search itself does most of the work.

### Hard constraints (non-negotiable design rules)

**NO PLAYER MAY EVER SEE ANOTHER PLAYER'S DECISION ON THE SAME TURN.** This rule applies everywhere:

- **In the game engine:** All players must commit their bank/continue choice before any of them are processed. The engine collects all actions first, then applies them simultaneously. A player's state vector must NOT include "did opponent X bank this turn" — only "did opponent X bank in a previous turn."
- **In MCTS during play:** When the AI runs its tree search, opponents' decisions inside the tree must be modeled probabilistically (sampled from their policy distribution), not chosen optimally with knowledge of our decision. Otherwise the opponents would have unfair lookahead.
- **In self-play training:** Each player records its own decision based on its own state vector. The opponents' actions for the same turn are never inputs to that player's state — only past actions are.
- **In opponent modeling within the tree:** When we expand a chance/decision node where multiple players act, we must sample all their actions independently from their respective policies BEFORE evaluating the resulting state. We never let one player condition on another's just-made choice.

Why this matters: in Bank, knowing whether someone else banked is a huge informational advantage. It tells you (1) the pot might still be there for you to grab, (2) you might be in a better/worse rank position than expected, (3) your opponent thought continuing was bad. A "last-decider" who sees others' choices will always win these games. This isn't just a fairness issue — training would converge on degenerate strategies that exploit this peek, and the resulting AI would play terribly in real simultaneous games.

### Open questions to resolve before building

1. How do we handle the simultaneous-decision problem cleanly? Joint actions blow up exponentially. Best bet: treat each player's decision as if independent (assume opponents play their network's policy distribution, not their max-prob action) — this respects the "no peeking" rule because we're sampling, not coordinating.

2. What's the minimum number of MCTS simulations to get good play? Need to benchmark.

3. How long does each game take to play with MCTS? If it's 100x slower than v1, training will be much slower per batch but each batch is more informative.

4. Do we need a "policy" head at all if we have a value head and MCTS? The policy head guides the tree search (cheaper than evaluating every child with the value head). It's a major speedup.

### Implementation phases (when we start building)

1. **Phase 1: Game engine** — clean, deterministic Bank simulator. Should support cloning a state for tree exploration.
2. **Phase 2: Network model** — two-headed network in PyTorch.
3. **Phase 3: MCTS implementation** — tree search with PUCT selection, chance node sampling, value network evaluation at leaves.
4. **Phase 4: Self-play loop** — generate games using MCTS + current network.
5. **Phase 5: Training loop** — sample batches from replay buffer, update network.
6. **Phase 6: Dashboard / monitoring** — reuse most of v1's dashboard. Different metrics: visit count entropy, value loss, MCTS-vs-raw-network agreement, win rate vs old checkpoints.
7. **Phase 7: Evaluation** — pit against v1's PPO model and the threshold bots.

### Likely pitfalls based on Dominion experience

- **Training is slow.** AlphaZero needs lots of games. Bank is fast per game but MCTS adds overhead. Plan for overnight runs.
- **Hyperparameters matter a lot.** PUCT exploration constant, MCTS sims per move, temperature schedule, learning rate — all need tuning.
- **Overfitting to specific opponents.** During training, vary the opponent pool (mix in older checkpoints, threshold bots, random play) so the AI generalizes.
- **The simultaneous-decision problem might not be solvable cleanly.** Be prepared to make approximations and document them clearly.

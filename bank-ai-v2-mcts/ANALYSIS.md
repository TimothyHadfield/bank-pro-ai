# Bank AI v2 — Plan Analysis

A self-critical re-read of `PLAN.md`. The goal is to find problems *before* we build them in. Each issue below names what could go wrong, why, and how to mitigate it.

This document is updated as we resolve open questions and discover new ones.

---

## 1. The simultaneous-decision approximation is the biggest theoretical risk

### What the plan says
Each player runs its own MCTS. Inside that tree, opponents' moves are sampled from their policy network. We never let one player condition on another's same-turn choice.

### Why this is fragile
This is not a true game-theoretic solution to simultaneous play. We are training each player against a model of the opponent that is the opponent's own policy network — which is the same network during pure self-play. Two real risks:

1. **Convergence to bad equilibria.** In simultaneous games, naive "best response to opponent's current policy" training can oscillate or settle on dominated strategies.
2. **Variance from sampling.** With 100 MCTS sims and 3 opponents, each opponent's policy is sampled ~100 times. Variance creates noise in Q-values that can mislead PUCT.

### Mitigations baked into the plan
- **Mixed opponent pool (~20% bot games)** — keeps the network's beliefs about opponents grounded in non-self behavior. Reduces oscillation.
- **Use enough simulations** (≥100) so opponent action distributions are well-sampled.
- **Track equilibrium oscillation** in the dashboard: log policy entropy per state. Oscillation across iterations = stability problem.
- **Accept this is an approximation.** Bank's same-turn interaction is weak (opponent banking doesn't change the dice; only affects rank). Approximation should be tolerable.

### Article note
The Temple Gates article does NOT discuss any of this. They describe self-play as a clean success. The above risks come from broader AlphaZero literature, not from them.

---

## 2. MCTS will be slow on CPU — possibly painfully so

### Math
- 100 sims per decision
- ~10 decisions per player per game (varies by player count and strategy)
- Up to 6 players running independent MCTS
- = 100 × 10 × 6 = **6,000 network forward passes per 6-player game** at minimum
- 50 games per iteration with average 4 players = ~200,000 forwards per iteration

This is 50–200× more compute per game than v1's PPO did. **All on CPU.**

### What could go wrong
- Iterations take minutes instead of seconds. Total training time could be a week or more.
- Dashboard polling might lag because the training process is CPU-bound.
- Play vs AI tab will have 1–2 second delays per AI move (acceptable but visible).

### Mitigations
- **Batch network forward passes.** Coordinate MCTS across the 4–6 players' trees so leaf evaluations can be batched. Non-trivial — requires asynchronous tree search or virtual loss. Not in slice 1, added in slice 4.
- **Start with fewer sims (50) and only increase if play quality is weak.**
- **Parallelize self-play games across CPU workers** using `multiprocessing`. Each worker has its own copy of the network. Major speedup on multi-core laptops.
- **Profile early.** Slice 1 should include a wall-clock measurement per game before we commit further.

### Hardware reminder
Training is on a CPU-only laptop. No GPU acceleration. Plan for **3–10 overnight runs** to reach competent play, possibly more for "near-optimal."

---

## 3. Chance node sampling will be noisy with 100 sims (mitigated)

### Original concern
After `continue`, sample one dice outcome per visit. With 100 sims, the sample variance is high. A 7 (1/6 chance) might be drawn 10 times in one search and 25 times in another.

### Mitigation in the plan (Section 4.7)
**Probability-weighted backups:** when computing Q-values, weight each child's contribution by its true 1/36 probability rather than visit fraction. This dramatically reduces sample noise.

This should be sufficient for Bank because the dice space is small (36 outcomes, 11 unique sums) and the consequences are stark (7 wipes pot vs anything else). If it's still too noisy, fall back to deterministic expansion of high-probability outcomes (7 and doubles especially).

---

## 4. Building a fresh tree every turn wastes work but is correct

### What the plan says
Build a fresh tree each turn. (Section 4.8)

### Why this is wasteful but probably right
After our action, the tree we built had useful sub-information. But the opponents' actual actions (which we couldn't see) split the tree, and the dice roll we couldn't predict moves us to a specific subtree. Most of the tree we built represented branches that didn't happen.

### Decision (kept from prior analysis)
Stick with fresh trees. Subtree reuse with stochastic + simultaneous play is a pile of edge cases. Revisit only if profiling shows tree build is a major cost.

---

## 5. The reward signal is sparse but the policy loss is dense

### Why this is concerning
Binary win/loss arrives once per game. With 4 players, baseline win rate is 25%. Each game gives us ~10 decisions per player × 4 players = 40 (state, policy, value) tuples, all tagged with the same value.

### Mitigation
- **Trust the policy loss to do most of the work.** It's cross-entropy on MCTS visit distributions, which is *much* denser than the value loss. AlphaZero relies on this.
- **Consider TD-style value bootstrapping** as a slice 4 enhancement if value loss stays high. Article mentioned Dominion uses temporal difference learning ("mid-game predictions get used as feedback signals before the game even ends").
- **Don't expect win rate to climb fast.** Plan for thousands of games before measurable improvement.

---

## 6. Variable player count (2–6) introduces real complications (NEW)

### What the plan says
Support 2–6 players from day one. State vector pads opponents to 5 slots. Self-play randomizes player count per game. Evaluation tested separately per count.

### Why this is harder than fixed-player count
- **Wider distribution to learn.** A 2-player game has very different dynamics than a 6-player game (in 2-player, banking gives you a bigger lead per bank; in 6-player, the pot grows faster and a 7 is more common per round). The network has to learn both.
- **Sample efficiency suffers.** With per-game player count randomization, only 1/5 of games are at any given count. Each count effectively has 1/5 the training data.
- **Padding leakage.** If padded opponent slots aren't handled carefully, the network might learn spurious patterns from the sentinel values.
- **Optimal strategy depends on player count.** "Bank at pot > 200" might be optimal at 4 players but suboptimal at 2 or 6. The network must condition on `num_players_in_game` (input #5).

### Mitigations baked into plan
- Input #5 (`num_players_in_game`) and #25 (`num_real_opponents`) explicitly tell the network how many opponents are real.
- Padded slots use sentinel values (-1 score, 0 in-round) that are distinguishable from real values.
- Opponents sorted by score before placement gives stable feature ordering.
- Per-player-count win rate tracked separately in the dashboard so we catch counts that lag.

### Risk
We may discover that one network struggling to handle all 5 player counts forces compromises. **If after slice 4 a single player count consistently underperforms, escalate to:**
- Using a one-hot encoding of player count (5 inputs instead of 1) for clearer signal
- Or training a separate fine-tuned head per player count
- Or curriculum learning: start at fixed 4 players, then anneal in 2/3/5/6

We won't preempt this — first try the unified approach.

---

## 7. Hyperparameters are educated guesses

### What's specified
SIMS_PER_MOVE=100, C_PUCT=1.4, BATCH_SIZE=512, LEARNING_RATE=1e-3, GAMES_PER_ITERATION=50, REPLAY_BUFFER_SIZE=50000, DIRICHLET_ALPHA=5.0, DIRICHLET_EPSILON=0.15.

### Reality
- DIRICHLET_ALPHA was raised from the AlphaZero default (0.3) to 5.0 to reflect Bank's tiny action space (paper rule: alpha ≈ 10/num_actions).
- DIRICHLET_EPSILON was lowered from 0.25 to 0.15 because our tree is shallow.
- The rest are loose defaults; **slice 4 includes a tuning pass.**

---

## 8. Replay buffer staleness mitigated, not eliminated

### Original concern
Old samples reflect old policies. Training on them pulls us back toward old behavior.

### Mitigation in the plan
- Each entry tagged with iteration number
- Sampling probability weighted by recency (newer entries sampled more often)
- Discard entries older than 30 iterations

### Residual risk
30-iteration window might be too long if learning is fast, too short if learning is slow. Make this configurable from the dashboard so we can adjust without restarting training.

---

## 9. The "no peeking" rule is enforced in the engine but easy to violate elsewhere

### Risk
- **Self-play loop:** Iterating `for pid in players_in_round` and running each MCTS sequentially is correct as long as no actions are *applied* until all are gathered. The plan's pseudocode is correct; just need careful code review.
- **Inside the tree:** When sampling opponent actions, each must be sampled from their own policy independent of others'. No "best response chain."
- **Logging/dashboard:** Even how we display action history matters — never imply one player decided before another in the same turn.

### Mitigations
- **Engine invariant:** assert at the start of each `step()` that `actions` contains exactly one entry per active player. Process all in a single batch.
- **Regression test:** `tests/test_no_peeking.py` — replay a sequence with players queried in different orders, verify decisions are byte-identical.
- **Code review at every slice boundary:** explicitly check this rule before declaring a slice complete.

---

## 10. The value head needs diverse training opponents

### Concern
With pure self-play, the value head learns "win probability against the current self-policy." Against different strategies (humans, threshold bots), it's miscalibrated.

### Mitigation in plan
- Mixed opponent pool (~20% of games include bots or older checkpoints)
- Periodic value calibration check: compare value head predictions vs actual win rates by bucket

### When to escalate
If value loss plateaus but win rates against humans look weird, mix in more diverse opponents (raise mix to 30–40%).

---

## 11. Dashboard performance with MCTS in the loop (NEW)

### Concern
v1's dashboard polled metrics.json every second. v2's training process is CPU-bound running MCTS — even reading/writing metrics.json could starve.

### Mitigations
- **Run training in a separate process from the dashboard server** (already planned, was true for v1)
- **Write metrics.json atomically** (write to .tmp, then rename) so the browser never reads partial JSON
- **Use a longer polling interval** (e.g. 3–5 sec) since MCTS is slow — there's nothing to update faster than that anyway
- **Run MCTS for Play vs AI in a worker thread** so the main HTTP server stays responsive

---

## 12. Play vs AI latency budget

### Concern
100-sim MCTS at ~10ms per sim = 1 second per AI move. With 5 AI opponents in a 6-player game, that's 5 seconds of waiting per round.

### Acceptable, but mitigations
- **Cache the loaded model** between calls — don't reload weights on each move
- **Provide "fast" mode (50 sims) and "quality" mode (300 sims)**
- **Display "AI N is thinking..." status** so the user knows the wait is normal
- Long-term: parallelize the MCTS calls across players (they're independent)

---

## 13. Knowing the AI is actually learning

### Why win rate vs bots is the wrong metric for MCTS
Even an untrained network paired with MCTS plays decently because the search itself is competent. So early win rate against threshold bots will be high regardless of training. We need different signals.

### Metrics that actually show learning
- **Policy loss decreasing over iterations**
- **MCTS-vs-raw-network agreement increasing** (network is learning to predict what MCTS does)
- **Win rate vs frozen older checkpoints** (gold standard — does AI improve over time?)
- **Win rate vs higher-sim version of itself** (200-sim MCTS should beat 100-sim — if the gap *narrows* over training, the policy is becoming a good MCTS proxy)

All four are in the dashboard from slice 3.

---

## 14. Scope and complexity

### Concern
4 slices, 22+ items, dashboard rebuild from scratch, MCTS implementation from scratch, training loop, evaluation suite, opponent pool. Each piece has unknowns.

### Mitigations
- **Slice 1 is intentionally minimal.** Get the integration working before optimizing any piece.
- **Code review and testing at every slice boundary.**
- **Don't perfect early slices.** Move on as soon as the slice produces correct output.

---

## Summary: Top concerns ranked by severity

1. **Performance** — MCTS will be slow on CPU. Plan for parallelism (slice 4) and patience (multiple overnight runs).
2. **Variable player count** — adds real complexity to network learning. Mitigated, but watch per-count win rates.
3. **Hyperparameter tuning** — many defaults are educated guesses. Slice 4 sweep is non-optional.
4. **Simultaneous decision approximation** — theoretically imperfect, probably tolerable for Bank.
5. **Reward sparsity** — same issue v1 had; trust the policy loss.
6. **Right metrics to watch** — win rate vs bots is the wrong sole metric for MCTS; need self-play-vs-history.
7. **Replay buffer staleness** — mitigated but residual risk.
8. **Latency in Play vs AI** — acceptable but visible.

These are problems to anticipate, not reasons to abandon the plan. AlphaZero is a well-established approach. The mitigations are real and most are well-understood. We revisit this analysis at the end of each slice.

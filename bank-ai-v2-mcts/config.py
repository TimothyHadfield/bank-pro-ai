"""All hyperparameters for Bank AI v2 (MCTS + Neural Network)."""

# ==================== Game ====================
NUM_PLAYERS_RANGE = (2, 6)         # Sampled per game in self-play
NUM_ROUNDS = 10
IMMUNITY_ROLLS = 3                 # Rolls 1-3 are protected

# ==================== State Vector ====================
INPUT_SIZE = 28
MAX_OPPONENTS = 5                  # Padded to this in state vector

# ==================== Network ====================
HIDDEN_SIZE = 128
HEAD_SIZE = 64
NUM_ACTIONS = 2                    # bank, continue

# ==================== MCTS ====================
# Defaults tuned for "Balanced" preset — fast enough for daily training,
# strong enough to give meaningful per-iteration improvement.
SIMS_PER_MOVE = 50                 # During self-play
SIMS_PER_MOVE_EVAL = 100           # During evaluation
SIMS_PER_MOVE_PLAY_FAST = 50       # Play vs AI fast mode
SIMS_PER_MOVE_PLAY_QUALITY = 200   # Play vs AI quality mode

C_PUCT = 1.4
DIRICHLET_ALPHA = 5.0              # Adjusted for 2-action space (alpha ~= 10/num_actions)
DIRICHLET_EPSILON = 0.15           # Lowered from 0.25 because tree is shallow

# ==================== Training ====================
BATCH_SIZE = 256                   # Smaller than AlphaZero's 4096 — fine for tiny network
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_AT_ITER = 1000
LEARNING_RATE_DECAYED = 1e-4
L2_WEIGHT = 1e-4
GAMES_PER_ITERATION = 20           # Was 50; smaller iterations = quicker feedback
TRAINING_STEPS_PER_ITERATION = 50  # Was 100; matches reduced data per iter
REPLAY_BUFFER_SIZE = 50000
MAX_BUFFER_AGE_ITERATIONS = 30
DEFAULT_WORKERS = 8                # Parallel self-play default (use most of CPU cores)

# ==================== Self-Play Opponent Mix ====================
PURE_SELF_PLAY_FRACTION = 0.80
ONE_BOT_FRACTION = 0.10
TWO_BOT_FRACTION = 0.05
OLD_CHECKPOINT_MIX_FRACTION = 0.05

# ==================== Temperature Schedule ====================
TEMPERATURE_INITIAL = 1.0
TEMPERATURE_LATE_GAME = 0.0        # Last 3 rounds go greedy
LATE_GAME_THRESHOLD_ROUNDS_REMAINING = 3

# ==================== Evaluation ====================
EVAL_INTERVAL_ITERATIONS = 10
CHECKPOINT_INTERVAL_ITERATIONS = 10
# train.py rotates one player count per eval, so this is "games per (bot × player count)"
EVAL_GAMES_PER_OPPONENT = 3
EVAL_PLAYER_COUNTS = [2, 3, 4, 5, 6]

# ==================== Bots ====================
THRESHOLD_BOT_THRESHOLDS = [100, 150, 200, 250, 300]
RANDOM_BOT_PROBABILITIES = [0.2, 0.5, 0.8]

# ==================== Reproducibility ====================
DEFAULT_SEED = 42

# Bank Pro AI

A PyTorch AI that learns the dice game Bank through self-play, plus a local dashboard to train it, watch charts, play against it, run AI vs AI tournaments and write your own strategy. Local Windows app; no website, no accounts, no cloud. Repo: TimothyHadfield/bank-pro-ai (public). Read `progress.md` before starting.

## Layout
- `bank-ai-v1-ppo/`: v1, PPO policy-gradient AI. Also hosts the **unified dashboard** (all tabs, v1 and v2).
  - `dashboard/server.py`: `HTTPServer` on `127.0.0.1:5001` (single-threaded; imports torch at startup). Serves `index.html` + `dashboard.js`, `play.js`, `play_unified.js`, `stats.js`, `build.js`, `train_v2.js`, `aivai_combined.js`, `style.css`.
  - `dashboard/unified_tournament.py`: competitors (v1 net, v2 net+MCTS, custom Build AI code) on the v2 engine; used by Play vs AI (unified mode) and AI vs AI.
  - `dashboard/build_ai_compiler.py`: server-side Build AI compiler. Must match the browser compiler in `dashboard/build.js` (`BuildBankGame.getVars`).
  - `game/environment.py`, `network/`, `training/`, `train.py`, `config.py` (`model_dir = "models"`).
- `bank-ai-v2-mcts/`: v2, AlphaZero-style network + MCTS.
  - `dashboard/server.py`: `ThreadingHTTPServer` on `127.0.0.1:5002`; `main()` opens a browser window. Front end: `index.html`, `common.js` (globals `jget`, `jpost`, `drawLineChart`), `train.js`, `play.js`, `stats.js`, `tournament.js`, `build.js`.
  - `GAME_RULES.md` (rules), `PLAN.md` (design), `NOTES.md`, `ANALYSIS.md`.
- `BankAI.bat` starts the v1 unified dashboard and opens http://localhost:5001. `BankAI-v2.bat` starts the v2-only dashboard.
- Shipped models: `bank-ai-v1-ppo/models/checkpoint_4900.pt`, `bank-ai-v2-mcts/models/latest.pt`. Other `*.pt`, `models/*`, `saves/` are git-ignored.

## Run and test
- Needs Python 3 + torch (CPU is fine), numpy, tensorboard, matplotlib, flask (`bank-ai-v1-ppo/requirements.txt`).
- Tests are plain scripts with their own `__main__` runners (not a pytest suite):
  - v2: `cd bank-ai-v2-mcts` then `python tests/<file>.py` for each file in `tests/`.
  - v1: `cd bank-ai-v1-ppo` then `python test_<name>.py`.
- `test_dashboard_scripts.py` (v2) needs Node; it loads the dashboard scripts in one shared scope to catch duplicate-global SyntaxErrors.
- `test_dashboard_server.py` (v1) runs the real handler on a free port with a temp PROJECT_DIR. It never touches real saves.
- Background testing only: don't run `BankAI*.bat` or the v2 `main()` (they open browser windows). Start the v1 server directly (`python dashboard/server.py --port 8115`) and use phone-view.

## Traps
- Front-end files are classic scripts sharing one global scope. A top-level `const { jget } = ...` in a tab script throws "duplicate variable" and kills the whole file. Use the globals from `common.js` directly.
- The engines start the next round inside `step()` and roll its first dice, so after a 7 `last_dice` belongs to the new round. The unified move API returns `round_ended` / `busted` for this.
- Build AI `p1..p5_score` = ALL players' scores sorted high to low (p1 may be you); `p1..p5_banked` follow that rank order. Keep `build.js` and `build_ai_compiler.py` in sync.
- Build AI code runs through `exec` with limited builtins (not a sandbox). That is why the servers bind to 127.0.0.1 only.
- Save/checkpoint names from the browser go through `_is_plain_name` in v1 `server.py` before any path join or delete.
- `bank-ai-v1-ppo/test_web.py` expects `models/checkpoint_0.pt`, which isn't shipped.
- `bank-ai-v2-mcts/metrics.json` is committed, so the v2 badge can say "running" when nothing is training.
- Folder lives in OneDrive: deletes can fail on locks, so the server renames to `*._deleted_<time>` instead (git-ignored).

## Rules
Tim owns visuals, wording, layout and names. $0 spend. Commit with explicit paths.

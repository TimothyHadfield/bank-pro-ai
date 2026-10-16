# Bank Pro AI — progress

## START HERE
Read this file, then `CLAUDE.md`. Last updated 2026-10-16 (bug sweep).

## Goal right now
Published showcase repo; no active feature work. Keep it working and tested.

## Status
| Area | State |
|---|---|
| v1 PPO training + unified dashboard (port 5001) | Works; swept 2026-10-16 |
| v2 MCTS training + v2 dashboard (port 5002) | Works; tab scripts fixed 2026-10-16 |
| Play vs AI (unified mode) | Multi-human turns and bust display fixed 2026-10-16 |
| Build AI (browser + server) | Variables now match on both sides |
| Tests | v2 `tests/*.py`, v1 `test_*.py` (plain scripts) |

## Authorized next steps
None.

## Standing instructions
- "Fix only clear defects. Don't change looks, wording, layout, names, features or game rules." (sweep brief, 2026-10-16)
- Background testing only: no browser windows on Tim's screen.

## Traps
See `CLAUDE.md` Traps. The main ones: shared global scope for dashboard scripts; `last_dice` belongs to the new round after a 7; Build AI variables must match `build.js`.

## Decisions
- 2026-10-16: v1 dashboard server binds 127.0.0.1 (was 0.0.0.0), same as v2. It runs user code and can delete saves.
- 2026-10-16: save/checkpoint names must be plain names (no `.`, `..`, slashes).
- 2026-10-16: unified Play vs AI logs a bust as "7 rolled! Pot wiped! Round over!", matching v1's Play tab wording.

## NOT verified
- Full training runs (v1 and v2) after the 2026-10-16 changes; only the test suites and dashboard screens were checked.
- Phone use: the dashboards are local-only, so the iPhone can't reach them.

## Rejected / parked
- (none yet)

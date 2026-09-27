# Bank — Official Game Rules

This is the authoritative reference for the rules of Bank. Any disagreement between code and this document means the code is wrong.

---

## Overview

Bank is a multiplayer dice game played over **10 rounds**. Two dice are rolled each turn. All players share **ONE common pot** (a running total) per round. The goal is to have the highest permanent (banked) score at the end of 10 rounds.

Player count: **2 to 6 players** (variable, configurable).

---

## Flow of a single round

1. The round's pot starts at **0**. The roll counter starts at **1**.
2. Two dice are rolled. The result is applied to the pot according to the rules below.
3. If the roll was a 7 (after immunity has expired), the pot is wiped to 0, the round ends immediately, and NO remaining players get anything. Go to step 7.
4. **During immunity (rolls 1–3):** NO decisions are made. Players CANNOT bank during immunity. The dice are rolled automatically and results applied to the pot. Skip to step 2 (next roll).
5. **After immunity ends** (after roll 3, and after every subsequent roll): all players who have not yet banked this round simultaneously choose: **BANK** or **CONTINUE**. The first decision point is **AFTER roll 3 completes, BEFORE roll 4 is rolled.**
   - Players who choose **BANK**: the current pot value is added to their permanent score. They sit out for the rest of this round. They do NOT participate in any further decisions this round.
   - Players who choose **CONTINUE**: they stay in and will face the next roll.
6. If all players have banked, the round ends. Otherwise, increment the roll counter and go back to step 2.
7. A new round begins (go to step 1). After 10 rounds, the game is over. **Highest permanent score wins.**

---

## Roll outcome rules

The behavior depends on whether the roll is within the first 3 rolls of the current round (immunity active) or after (immunity expired).

### During immunity (rolls 1, 2, and 3 of the current round)
- **Normal roll** (not a 7, not doubles): Add the sum of the two dice to the pot. Example: rolling 4+5=9, pot goes from 30 to 39.
- **Rolling a 7:** Instead of busting, **ADD 70 to the pot**. The round continues normally. Example: rolling 3+4=7, pot goes from 30 to 100.
- **Rolling doubles:** Just add the sum of the two dice to the pot. Doubles have **NO special effect during immunity**. Example: rolling 3+3, pot goes from 30 to 36 (not doubled).

### After immunity (rolls 4, 5, 6, ... of the current round)
- **Normal roll** (not a 7, not doubles): Add the sum of the two dice to the pot. Same as during immunity.
- **Rolling a 7:** The pot is **wiped to 0**. The round ends IMMEDIATELY. All players who had not yet banked this round get nothing. Players who banked earlier in this round keep what they banked.
- **Rolling doubles:** The pot is **MULTIPLIED BY 2**. Nothing is added — the pot just doubles. It does not matter what the double is (two 1s, two 6s, etc.) — the effect is always: pot = pot × 2. Example: pot is 80, doubles are rolled, pot becomes 160.

---

## Important clarifications

- The pot is **SHARED**. If the pot is 120 and Player A banks, Player A gets 120 added to their permanent score. The pot remains at 120 for the remaining players. If the pot later grows to 200 and Player B banks, Player B gets 200. The pot value at the moment you bank is what you get.
- Banking is **SIMULTANEOUS**. All remaining players decide at the same time. **No player sees another player's decision before making their own.** This means the AI cannot condition its decision on what other players chose this turn.
- Players **CANNOT bank during immunity** (rolls 1–3). Since 7s are protected and doubles have no special effect, it is always optimal to continue. The first decision point is roll 4.
- **Immunity resets every round.** Each round's first 3 rolls have immunity, regardless of what happened in previous rounds.
- A 7 after immunity ends the round for **EVERYONE still in**. Only players who already banked earlier that round keep their points from that round.

---

## What NOT to do (game-logic mistakes)

- Do **NOT** give each player their own pot. The pot is shared.
- Do **NOT** let players see each other's bank/continue decisions before committing their own.
- Do **NOT** add the dice sum to the pot when doubles are rolled after immunity. Doubles after immunity ONLY multiply the pot by 2.
- Do **NOT** forget that immunity resets every round.

---

## Winning

After 10 rounds, the player(s) with the highest permanent (banked) score win. **Ties for first share the win** — there can be more than one winner. Players in 2nd, 3rd, 4th, etc. all lose. There is no partial credit for non-first finishes.

"""
Server-side compiler for Build AI pseudo-Python.

Mirrors the JavaScript compiler at dashboard/build.js. Lets a user-written strategy
participate in a server-side tournament alongside trained networks.

User code uses Python-ish syntax with two extra keywords:
    bank        -> "return 'bank'"      (immediately bank)
    continue    -> "return 'continue'"  (immediately continue)

Variables exposed to user code (matching the JS variant):
    pot, round, roll, players_in, my_score, num_players,
    p1_score..p5_score        (all banked scores sorted high to low, 0 past the last player)
    p1_banked..p5_banked      (1 if that ranked player already banked this round, else 0)

Plus any custom variables the user defined in their code.

Safety: this runs user code via exec() in a restricted namespace. The dashboard server
is local-only, so we don't fully sandbox — but we do limit the builtins available.
"""

from __future__ import annotations

import re
from typing import Callable


_VAR_NAMES = (
    "pot", "round", "roll", "players_in", "my_score", "num_players",
    "p1_score", "p2_score", "p3_score", "p4_score", "p5_score",
    "p1_banked", "p2_banked", "p3_banked", "p4_banked", "p5_banked",
)


_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len,
    "int": int, "float": float, "bool": bool, "round": round,
    "sum": sum, "range": range,
    "True": True, "False": False, "None": None,
}


def compile_build_ai(code: str) -> Callable[[dict], str]:
    """
    Compile pseudo-Python user code into a callable `decide(state) -> "bank"|"continue"`.

    Raises ValueError if the code can't be compiled.
    """
    # Translate `bank` / `continue` action keywords into Python `return` statements.
    # We only do this when the keyword stands alone as a statement (whitespace +
    # optional inline comment).
    transformed_lines: list[str] = []
    for raw in code.split("\n"):
        stripped = raw.rstrip()
        # Detect action statement: optional indent, then exactly "bank" or "continue",
        # optionally followed by a # comment.
        m = re.match(r"^(\s*)(bank|continue)\s*(#.*)?$", stripped)
        if m:
            indent, kw, _ = m.groups()
            transformed_lines.append(f'{indent}return "{kw}"')
        else:
            transformed_lines.append(stripped)

    # Build the wrapped function. Variables come from the state dict.
    var_unpack = ""
    for v in _VAR_NAMES:
        var_unpack += f"    {v} = state.get('{v}', 0)\n"

    body = "\n".join("    " + line for line in transformed_lines)
    full_source = (
        "def _decide(state):\n"
        + var_unpack
        + body
        + "\n    return 'continue'\n"  # default fallback
    )

    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(full_source, "<build_ai>", "exec"), namespace)
    except SyntaxError as e:
        raise ValueError(f"Build AI syntax error: {e.msg} at line {e.lineno}")

    decide_fn = namespace.get("_decide")
    if not callable(decide_fn):
        raise ValueError("Build AI compilation produced no decide function")
    return decide_fn


def state_for_build_ai(game, player_id: int) -> dict:
    """
    Build the variable-bag dict that the compiled decide function reads from.

    Compatible with both v1 and v2 BankGame engines (uses fields they both expose).
    """
    # Both v1 and v2 BankGame have these fields
    pot = int(game.pot)
    round_ = int(game.round_number)
    roll = int(game.roll_number)
    num_players = int(game.num_players)
    players_in = len(game.players_in_round)
    my_score = int(game.banked_scores[player_id])

    # Same as build.js BuildBankGame.getVars(): p1..p5 are ALL players' scores sorted
    # high to low (p1_score = leader, which may be you), and p1..p5_banked follow that
    # same rank order. Stable sort keeps seat order for ties, like JS Array.sort.
    all_scores = [int(s) for s in game.banked_scores]
    ranked = sorted(range(num_players), key=lambda i: -all_scores[i])
    opp_scores = [all_scores[i] for i in ranked][:5]
    # "banked" means: banked already this round (= NOT in players_in_round)
    opp_banked = [0 if i in game.players_in_round else 1 for i in ranked][:5]
    while len(opp_scores) < 5:
        opp_scores.append(0)
        opp_banked.append(0)

    return {
        "pot": pot,
        "round": round_,
        "roll": roll,
        "players_in": players_in,
        "my_score": my_score,
        "num_players": num_players,
        "p1_score": opp_scores[0], "p2_score": opp_scores[1], "p3_score": opp_scores[2],
        "p4_score": opp_scores[3], "p5_score": opp_scores[4],
        "p1_banked": opp_banked[0], "p2_banked": opp_banked[1], "p3_banked": opp_banked[2],
        "p4_banked": opp_banked[3], "p5_banked": opp_banked[4],
    }

"""The scripted tour: `uv run poe demo`.

Runs every tool against the committed demo fixture league (no credentials, no network) and
prints a readable transcript to stdout. This is the same in-process `mcp.Client` path
`tests/mcp/` and `scripts/bench_tokens.py` use: a real protocol round-trip, just not over stdio,
so what it prints is genuine tool output, not a hand-written example. Its output is the
source for the transcripts in the README: copy real runs of this script, never invent one.
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ["FFMCP_MODE"] = "demo"
os.environ.setdefault("FFMCP_TEAM_ID", "1")

from ffmcp.server import build_server
from mcp import Client

# Tool output uses real em dashes and ± signs. Windows' legacy console codepage is not UTF-8, so
# without this a captured transcript comes out mojibake'd on Windows specifically.
sys.stdout.reconfigure(encoding="utf-8")

# (section heading, tool, arguments): one pass through every tool, grouped by the question it
# answers, in the order a manager would actually ask them on a Sunday morning.
TOUR: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("My team", "get_my_team", {}),
    ("Who should I start? -> optimize_lineup", "optimize_lineup", {}),
    ("This week's matchup", "analyze_matchup", {}),
    (
        "Start/sit: the two flex candidates",
        "compare_players",
        {"names": ["Dominic Okafor", "Elijah Villanueva"]},
    ),
    ("Who should I pick up? -> find_waiver_targets", "find_waiver_targets", {}),
    ("Who is due to regress? -> buy_low_sell_high", "buy_low_sell_high", {}),
    ("One player, in depth", "player_report", {"name": "Damon Rios"}),
    ("Should I make this trade? -> find_trades", "find_trades", {}),
    (
        "Evaluate a specific offer -> evaluate_trade",
        "evaluate_trade",
        {
            "give": ["Quentin Hargrove"],
            "get": ["Caleb Castellan"],
            "partner_team_id": 2,
        },
    ),
    (
        "How likely am I to make the playoffs? -> simulate_season",
        "simulate_season",
        {"n_sims": 2000},
    ),
    ("League standings", "league_standings", {}),
    ("Who is actually good? -> power_rankings", "power_rankings", {}),
    ("How far should I trust the projections? -> projection_accuracy", "projection_accuracy", {}),
)


async def _run_tour() -> None:
    async with Client(build_server()) as client:
        for heading, tool, arguments in TOUR:
            result = await client.call_tool(tool, arguments)
            if result.is_error:
                raise RuntimeError(f"{tool}{arguments} returned an error: {result.content}")
            text = result.content[0].text  # type: ignore[union-attr]

            print(f"\n=== {heading} ===")
            call_args = f"({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
            print(f"$ {tool}{call_args}")
            print(text)


def main() -> None:
    asyncio.run(_run_tour())


if __name__ == "__main__":
    main()

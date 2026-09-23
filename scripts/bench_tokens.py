"""Regenerate the README token-efficiency benchmark table.

Runs every tool at every declared detail level against the demo fixture league. No network, no
credentials: it uses the same fixture the test suite and a reviewer's ``uv run poe demo`` both
use. Writes a markdown table between the ``<!-- BENCH:START -->`` / ``<!-- BENCH:END -->``
markers in ``README.md``.

Columns:
  - **tool**, **detail**
  - **tokens**: the actual response, counted by ``ffmcp.tokencount.estimate_tokens`` (the same
    deterministic estimator ``tests/test_token_budgets.py`` asserts against, so this number and
    the CI gate can never silently drift apart).
  - **upstream bytes**: the size of the raw fixture payload a provider would have fetched to
    answer the question (``tests/fixtures/demo_league.json`` in demo mode; the same file for
    every row is a simplification, noted below the table).
  - **tokens-if-naive**: tokens for a passthrough implementation that serialized that raw
    payload straight into the response, instead of reducing it in Python first.

``--check`` regenerates the table in memory and exits non-zero if it differs from what is
committed, without writing. CI runs this so a stale table fails the build.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ["FFMCP_MODE"] = "demo"
os.environ.setdefault("FFMCP_TEAM_ID", "1")

from ffmcp.budgets import TOKEN_BUDGETS
from ffmcp.server import build_server
from ffmcp.tokencount import estimate_tokens
from mcp import Client

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

START_MARKER = "<!-- BENCH:START -->"
END_MARKER = "<!-- BENCH:END -->"

# (label, tool, arguments, budget key). One row per budget entry declared in ffmcp/budgets.py.
ROWS: tuple[tuple[str, str, dict[str, object], str], ...] = (
    ("get_my_team", "get_my_team", {}, "get_my_team:compact"),
    ("get_my_team", "get_my_team", {"detail": "standard"}, "get_my_team:standard"),
    ("get_my_team", "get_my_team", {"detail": "full"}, "get_my_team:full"),
    ("optimize_lineup", "optimize_lineup", {}, "optimize_lineup"),
    ("analyze_matchup", "analyze_matchup", {}, "analyze_matchup:compact"),
    ("analyze_matchup", "analyze_matchup", {"detail": "standard"}, "analyze_matchup:standard"),
    ("find_trades", "find_trades", {}, "find_trades"),
    (
        "evaluate_trade",
        "evaluate_trade",
        {"give": ["Quentin Hargrove"], "get": ["Caleb Castellan"], "partner_team_id": 2},
        "evaluate_trade",
    ),
    ("find_waiver_targets", "find_waiver_targets", {}, "find_waiver_targets:compact"),
    (
        "find_waiver_targets",
        "find_waiver_targets",
        {"detail": "standard"},
        "find_waiver_targets:standard",
    ),
    (
        "find_waiver_targets",
        "find_waiver_targets",
        {"detail": "full", "limit": 25},
        "find_waiver_targets:full",
    ),
    ("buy_low_sell_high", "buy_low_sell_high", {}, "buy_low_sell_high"),
    ("simulate_season", "simulate_season", {"n_sims": 2000}, "simulate_season"),
    ("league_standings", "league_standings", {}, "league_standings:compact"),
    ("power_rankings", "power_rankings", {}, "power_rankings"),
    ("player_report", "player_report", {"name": "Caleb Fairbanks"}, "player_report"),
    (
        "compare_players",
        "compare_players",
        {"names": ["Dominic Okafor", "Elijah Villanueva"]},
        "compare_players",
    ),
    ("projection_accuracy", "projection_accuracy", {}, "projection_accuracy"),
)


def _detail_of(budget_key: str) -> str:
    return budget_key.split(":", 1)[1] if ":" in budget_key else "-"


def _naive_upstream_text() -> str:
    """What a passthrough implementation would have serialized: the raw fixture league payload.

    Using the whole league payload for every row is a simplification. A real passthrough tool
    would vary in what it dumped. It is a conservative one for the point being made: even
    `optimize_lineup`'s five-swap answer, the cheapest possible shape, is being compared against
    the single upstream fetch that makes it possible.
    """
    league = (FIXTURES_DIR / "demo_league.json").read_text(encoding="utf-8")
    market = (FIXTURES_DIR / "demo_market.json").read_text(encoding="utf-8")
    return league + market


async def _measure() -> list[dict[str, object]]:
    naive_text = _naive_upstream_text()
    naive_tokens = estimate_tokens(naive_text)
    upstream_bytes = len(naive_text.encode("utf-8"))

    measurements: list[dict[str, object]] = []
    async with Client(build_server()) as client:
        for label, tool, arguments, budget_key in ROWS:
            result = await client.call_tool(tool, arguments)
            if result.is_error:
                raise RuntimeError(f"{tool}{arguments} returned an error: {result.content}")
            text = result.content[0].text  # type: ignore[union-attr]
            measurements.append(
                {
                    "tool": label,
                    "detail": _detail_of(budget_key),
                    "tokens": estimate_tokens(text),
                    "upstream_bytes": upstream_bytes,
                    "naive_tokens": naive_tokens,
                }
            )
    return measurements


def _render_table(measurements: list[dict[str, object]]) -> str:
    header = "| tool | detail | tokens | upstream bytes | tokens-if-naive |"
    sep = "|---|---|---:|---:|---:|"
    lines = [header, sep]
    for row in measurements:
        lines.append(
            f"| `{row['tool']}` | {row['detail']} | {row['tokens']} "
            f"| {row['upstream_bytes']:,} | ~{row['naive_tokens']:,} |"
        )
    table = "\n".join(lines)
    note = (
        "\n\n_Every row's naive comparison is against the full fixture league payload "
        "(a 10-team league snapshot, standing in for a real upstream fetch) serialized as-is, "
        "rather than the reduced, narrow response above it. "
        "`tokens` is the deterministic local estimator. "
        "`upstream bytes`/`tokens-if-naive` are the same for "
        "every row because this benchmark runs against one fixture league. A live league's "
        "numbers vary by roster size and week, but the shape of the reduction does not._"
    )
    return f"{START_MARKER}\n{table}{note}\n{END_MARKER}"


def _update_readme(new_block: str, *, check: bool) -> bool:
    content = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        raise SystemExit(f"README.md is missing {START_MARKER}/{END_MARKER} markers")
    start = content.index(START_MARKER)
    end = content.index(END_MARKER) + len(END_MARKER)
    updated = content[:start] + new_block + content[end:]

    if updated == content:
        return False
    if check:
        return True
    README_PATH.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if README.md's table would change, without writing it",
    )
    args = parser.parse_args()

    measurements = asyncio.run(_measure())
    block = _render_table(measurements)
    changed = _update_readme(block, check=args.check)

    if args.check:
        if changed:
            print("README.md's benchmark table is stale; run `uv run poe bench`.", file=sys.stderr)
            sys.exit(1)
        print("README.md's benchmark table is up to date.")
        return

    print("README.md's benchmark table regenerated." if changed else "No change.")
    for row in measurements:
        over = row["tokens"] > TOKEN_BUDGETS.get(  # type: ignore[operator]
            f"{row['tool']}:{row['detail']}" if row["detail"] != "-" else row["tool"], 10**9
        )
        flag = " OVER BUDGET" if over else ""
        print(f"  {row['tool']:<20} {row['detail']:<10} {row['tokens']:>5} tok{flag}")


if __name__ == "__main__":
    main()

"""Every tool x detail level, run against the demo fixture league, comes in under its declared
budget (``src/ffmcp/budgets.py``), counted by the same deterministic local estimator
``scripts/bench_tokens.py`` uses, so this test needs no network and no API key, and the number
the README reports is the exact number this test enforces. This failing is a real failure, not
a warning.
"""

from __future__ import annotations

import pytest

from ffmcp.budgets import TOKEN_BUDGETS
from ffmcp.server import build_server
from ffmcp.tokencount import estimate_tokens
from mcp import Client

# (tool name, arguments, budget key into TOKEN_BUDGETS) for every tool x detail level the
# budgets table declares, against tests/fixtures/demo_league.json.
CALLS: tuple[tuple[str, dict[str, object], str], ...] = (
    ("get_my_team", {}, "get_my_team:compact"),
    ("get_my_team", {"detail": "standard"}, "get_my_team:standard"),
    ("get_my_team", {"detail": "full"}, "get_my_team:full"),
    ("optimize_lineup", {}, "optimize_lineup"),
    ("analyze_matchup", {}, "analyze_matchup:compact"),
    ("analyze_matchup", {"detail": "standard"}, "analyze_matchup:standard"),
    ("find_trades", {}, "find_trades"),
    (
        "evaluate_trade",
        {"give": ["Quentin Hargrove"], "get": ["Caleb Castellan"], "partner_team_id": 2},
        "evaluate_trade",
    ),
    ("find_waiver_targets", {}, "find_waiver_targets:compact"),
    ("find_waiver_targets", {"detail": "standard"}, "find_waiver_targets:standard"),
    ("find_waiver_targets", {"detail": "full", "limit": 25}, "find_waiver_targets:full"),
    ("simulate_season", {"n_sims": 2000}, "simulate_season"),
    ("league_standings", {}, "league_standings:compact"),
    ("power_rankings", {}, "power_rankings"),
    ("buy_low_sell_high", {}, "buy_low_sell_high"),
    ("player_report", {"name": "Caleb Fairbanks"}, "player_report"),
    ("compare_players", {"names": ["Dominic Okafor", "Elijah Villanueva"]}, "compare_players"),
    ("projection_accuracy", {}, "projection_accuracy"),
)

assert {key for _, _, key in CALLS} == set(TOKEN_BUDGETS), (
    "CALLS must cover every budget in ffmcp.budgets.TOKEN_BUDGETS, and nothing else"
)


@pytest.fixture(autouse=True)
def _demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMCP_MODE", "demo")
    monkeypatch.setenv("FFMCP_TEAM_ID", "1")


@pytest.mark.parametrize("name,arguments,budget_key", CALLS, ids=[c[2] for c in CALLS])
async def test_tool_response_is_within_budget(
    name: str, arguments: dict[str, object], budget_key: str
) -> None:
    async with Client(build_server()) as client:
        result = await client.call_tool(name, arguments)

    assert not result.is_error, f"{name}{arguments} returned an error: {result.content}"
    text = result.content[0].text  # type: ignore[union-attr]
    tokens = estimate_tokens(text)
    budget = TOKEN_BUDGETS[budget_key]
    assert tokens <= budget, f"{budget_key} used ~{tokens} tokens; budget is {budget}"

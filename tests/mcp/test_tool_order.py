"""``tools/list`` order matches the documented tool catalog exactly.

Registration order is deliberate: a stable ``tools/list`` lets a host's
prompt cache hit across turns. ``server.build_server()`` registers one tool per call, in this
order, specifically so file-by-file grouping in ``mcp/tools_*.py`` can't silently reorder it.
"""

from __future__ import annotations

from conftest import demo_client

DOCUMENTED_ORDER = (
    "get_my_team",
    "optimize_lineup",
    "analyze_matchup",
    "find_trades",
    "evaluate_trade",
    "find_waiver_targets",
    "buy_low_sell_high",
    "simulate_season",
    "league_standings",
    "power_rankings",
    "player_report",
    "compare_players",
    "projection_accuracy",
)


async def test_tools_list_matches_documented_order() -> None:
    async with demo_client() as client:
        result = await client.list_tools()
    assert tuple(tool.name for tool in result.tools) == DOCUMENTED_ORDER


DOCUMENTED_RESOURCE_ORDER = (
    "ffmcp://league/settings",
    "ffmcp://league/teams",
    "ffmcp://glossary",
)


async def test_resources_list_matches_documented_order() -> None:
    async with demo_client() as client:
        result = await client.list_resources()
        templates = await client.list_resource_templates()

    assert tuple(resource.uri for resource in result.resources) == DOCUMENTED_RESOURCE_ORDER
    assert [t.uri_template for t in templates.resource_templates] == [
        "ffmcp://team/{team_id}/roster{?week,detail}"
    ]


DOCUMENTED_PROMPT_ORDER = ("weekly_checkin", "trade_workshop", "playoff_push")


async def test_prompts_list_matches_documented_order() -> None:
    async with demo_client() as client:
        result = await client.list_prompts()
    assert tuple(prompt.name for prompt in result.prompts) == DOCUMENTED_PROMPT_ORDER

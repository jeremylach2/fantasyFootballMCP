"""Every tool is callable end-to-end in demo mode; none raises on
fixture data. "Raises" here means crashes the server: a deliberately short ``ToolError`` (an
ambiguous player name, an out-of-range week) is a correct response, not a failure; see
``tests/mcp/test_error_adaptation.py`` for that path specifically.
"""

from __future__ import annotations

import pytest
from conftest import demo_client
from mcp.types import TextResourceContents

# One (tool name, arguments) per tool, against the demo fixture league.
CALLS: tuple[tuple[str, dict[str, object]], ...] = (
    ("get_my_team", {}),
    ("get_my_team", {"detail": "standard"}),
    ("get_my_team", {"detail": "full"}),
    ("optimize_lineup", {}),
    ("analyze_matchup", {}),
    ("analyze_matchup", {"detail": "standard"}),
    ("find_trades", {}),
    (
        "evaluate_trade",
        {"give": ["Quentin Hargrove"], "get": ["Caleb Castellan"], "partner_team_id": 2},
    ),
    ("find_waiver_targets", {}),
    ("find_waiver_targets", {"detail": "full"}),
    ("simulate_season", {"n_sims": 200}),
    ("league_standings", {}),
    ("player_report", {"name": "Caleb Fairbanks"}),
    ("compare_players", {"names": ["Dominic Okafor", "Elijah Villanueva"]}),
)


@pytest.mark.parametrize("name,arguments", CALLS, ids=[c[0] for c in CALLS])
async def test_tool_call_succeeds_on_fixture_data(name: str, arguments: dict[str, object]) -> None:
    async with demo_client() as client:
        result = await client.call_tool(name, arguments)

    assert not result.is_error, f"{name}{arguments} returned an error: {result.content}"
    assert result.content, f"{name}{arguments} returned no content"


def _text(contents: object) -> str:
    content = contents[0]  # type: ignore[index]
    assert isinstance(content, TextResourceContents)
    return content.text


async def test_static_resources_are_readable() -> None:
    async with demo_client() as client:
        for uri in ("ffmcp://league/settings", "ffmcp://league/teams", "ffmcp://glossary"):
            result = await client.read_resource(uri)
            assert result.contents and _text(result.contents)


async def test_team_roster_template_is_readable() -> None:
    async with demo_client() as client:
        result = await client.read_resource("ffmcp://team/2/roster")
        detailed = await client.read_resource("ffmcp://team/2/roster?detail=full")

    assert "Team Bravo" in _text(result.contents)
    assert "TREND" in _text(detailed.contents)


async def test_every_prompt_returns_a_message() -> None:
    async with demo_client() as client:
        for name in ("weekly_checkin", "trade_workshop", "playoff_push"):
            result = await client.get_prompt(name)
            assert result.messages

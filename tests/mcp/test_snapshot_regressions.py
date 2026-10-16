"""Regression tests against a real league: ``tests/fixtures/snapshot_*``, an anonymized
16-team league at week 6 of 2026 (``scripts/record_fixtures.py``). "Me" is team 12.

Why a second fixture: all 230 tests passed while every bug in ``docs/audit-2026-10-15.md``'s P0
list was live, because the synthetic demo league differs from live ESPN in exactly the ways
that hid them. ESPN projects a bye as ``0.0``, not nothing; injured free agents carry a season
rate of zero; released players sit on no NFL team; and a 16-team league's waiver wire is thin.
Each test below reproduces one audit finding on that real shape and fails on the old code.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import demo_client


@pytest.fixture(autouse=True)
def _snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMCP_FIXTURE", "snapshot")
    monkeypatch.setenv("FFMCP_TEAM_ID", "12")


async def _call(tool: str, arguments: dict[str, object] | None = None) -> str:
    async with demo_client() as client:
        result = await client.call_tool(tool, arguments or {})
    assert not result.is_error, result.content
    return result.content[0].text  # type: ignore[union-attr]


def _row(table: str, name: str) -> str:
    return next(line for line in table.splitlines() if name in line)


def _odds(table: str, team: str) -> float:
    match = re.search(rf"^\*?{team}\s.*?(\d+\.\d)%", table, re.MULTILINE)
    assert match is not None, f"{team} not in:\n{table}"
    return float(match.group(1))


# --- P0-1: the season simulation ------------------------------------------------------------


async def test_this_weeks_byes_do_not_sink_a_season() -> None:
    """Team 16 has three starters on bye and its quarterback out this week. Holding this week
    constant all season simulated it at 7.9% (ESPN: 54%); team 8, with a soft week, at 87.4%
    (ESPN: 44.5%)."""
    table = await _call("simulate_season")
    assert _odds(table, "Team Oscar") > 30.0  # team 16
    assert _odds(table, "Team Hotel") < 80.0  # team 8


async def test_every_tool_reports_the_same_playoff_odds() -> None:
    """Three tools gave three numbers for team 12 (67.6%, 93.8%, 88-97%)."""
    simulated = _odds(await _call("simulate_season"), "Team Lima")
    standings = await _call("league_standings")
    listed = float(re.search(r"Team Lima.*?(\d+\.\d)%", standings).group(1))  # type: ignore[union-attr]
    assert listed == simulated

    matchup = await _call("analyze_matchup")
    win = float(re.search(r"(\d+)% win prob", matchup).group(1)) / 100  # type: ignore[union-attr]
    stakes = re.search(r"win -> (\d+)% playoff odds, loss -> (\d+)%", matchup)
    assert stakes is not None
    if_win, if_loss = float(stakes.group(1)), float(stakes.group(2))
    assert abs(win * if_win + (1 - win) * if_loss - simulated) < 4.0  # within simulation noise


# --- P0-2 / P0-3: weeks and byes --------------------------------------------------------------


async def test_a_bye_projected_at_zero_is_flagged() -> None:
    """Vikings D/ST and McPherson (CIN) are on bye in week 6, which ESPN projects as 0.0."""
    roster = await _call("get_my_team")
    assert _row(roster, "Vikings D/ST").endswith("BYE")
    assert _row(roster, "E. McPherson").endswith("BYE")
    report = await _call("player_report", {"name": "Vikings D/ST"})
    assert "Status: Bye (week 6)" in report


async def test_a_future_week_shows_that_weeks_byes() -> None:
    """``get_my_team(week=11)`` printed week 6's projections, Love 25.7 included."""
    roster = await _call("get_my_team", {"week": 11})
    for name in ("J. Love", "D. London", "C. Watson"):
        assert _row(roster, name).endswith("BYE")
    assert "season-average projections" in roster


async def test_a_past_week_is_a_hindsight_recap() -> None:
    """``optimize_lineup(week=5)`` returned week 6's numbers labelled week 5."""
    async with demo_client() as client:
        result = await client.call_tool("optimize_lineup", {"week": 5})
    advice = result.structured_content
    assert advice is not None
    assert advice["week"] == 5
    assert advice["actual"] is not None
    assert advice["actual"]["points_scored"] > 0


# --- P0-4: waivers --------------------------------------------------------------------------


async def test_no_waiver_move_costs_the_season() -> None:
    """The top suggestion was a kicker worth -24 rest of season, and every pickup cut the
    backup defense, because defenses have the lowest season rates."""
    table = await _call("find_waiver_targets")
    lines = table.splitlines()
    assert lines[0].split() == ["PLAYER", "POS", "TM", "PROJ", "ROS", "VAL", "DROP"]
    rows = [line for line in lines[1:] if not line.startswith(("(", "Stash", "Handcuffs"))]
    assert rows
    for line in rows:
        position, ros, drop = line[17:21].strip(), int(line[32:37]), line[44:].strip()
        assert ros > 0, line
        if position != "D/ST":
            assert not drop.endswith("D/ST"), line


async def test_a_kicker_pickup_cuts_the_second_kicker() -> None:
    table = await _call("find_waiver_targets", {"position": "K", "limit": 3})
    for line in table.splitlines()[1:]:
        if re.match(r"\S.*\sK\s", line):
            assert line.rstrip().endswith("E. McPherson"), line


async def test_an_injured_stash_is_not_priced_at_a_season_of_zeros() -> None:
    """Achane (IR, 38% owned) was valued at -91.7 rest of season."""
    table = await _call("find_waiver_targets", {"position": "RB"})
    stash = _row(table, "Stash")
    assert "D. Achane RB MIA (38% owned, no return projected)" in stash
    assert "J. Brooks RB CAR (41% owned, back wk 9" in stash
    report = await _call("player_report", {"name": "Achane"})
    assert "Outlook: no return projected this season" in report
    assert "-9" not in report.split("Value over replacement:")[1].split("\n")[0]


async def test_released_players_are_never_suggested() -> None:
    table = await _call("find_waiver_targets", {"limit": 25})
    assert " FA " not in table


# --- P0-5: trades ---------------------------------------------------------------------------


async def test_no_trade_asks_the_partner_to_give_up_far_more_than_market_value() -> None:
    """The top idea was my backup QB (market 267) for Mike Evans (3,190), marked "accept"."""
    calls: tuple[dict[str, object], ...] = ({}, {"positions_wanted": ["RB"]})
    for args in calls:
        trades = await _call("find_trades", args)
        lines = [line for line in trades.splitlines() if "(id " in line]
        assert lines, trades
        for line in lines:
            given, received = (
                float(v.replace(",", ""))
                for v in re.search(r"mkt ([\d,]+) vs ([\d,]+)", line).groups()  # type: ignore[union-attr]
            )
            assert received <= 1.25 * given, line


async def test_kickers_are_left_out_of_trades_by_default() -> None:
    trades = await _call("find_trades", {"max_results": 10})
    assert " K." not in trades
    assert "Butker" not in trades and "Fairbairn" not in trades


async def test_evaluate_trade_says_the_partner_will_decline_a_lopsided_offer() -> None:
    async with demo_client() as client:
        result = await client.call_tool(
            "evaluate_trade", {"give": ["Jacoby Brissett"], "get": ["Mike Evans"]}
        )
    verdict = result.structured_content
    assert verdict is not None
    assert verdict["partner_likely_accepts"] is False
    assert "267" in verdict["market"] and "3,190" in verdict["market"]
    assert verdict["deadline"] == "Trade deadline in 47 days (Dec 2)."


# --- P0-6 / P1-8: zeroed game-time decisions -------------------------------------------------


async def test_a_zeroed_questionable_player_gets_a_real_range_and_a_warning() -> None:
    """Barkley (Q): ESPN 0.0, Sleeper 10.7, printed as "range 0.0-2.6 ... disagree by 1065%"."""
    report = await _call("player_report", {"name": "Saquon Barkley"})
    assert "range 0.0-2.6" not in report
    assert "%" not in report.split("\n")[2]
    assert "ESPN projects 0 (expects him out); Sleeper 10.7" in report

    async with demo_client() as client:
        result = await client.call_tool("optimize_lineup", {})
    assert result.structured_content is not None
    caveats = result.structured_content["caveats"]
    assert any(c.startswith("Saquon Barkley (Questionable): ESPN projects 0") for c in caveats)


# --- P1: rough edges ------------------------------------------------------------------------


async def test_names_are_matched_the_way_people_type_them() -> None:
    assert (await _call("player_report", {"name": "AJ Brown"})).startswith("A.J. Brown")
    async with demo_client() as client:
        result = await client.call_tool("player_report", {"name": "Saquon Barkly"})
    assert result.is_error
    assert "Did you mean: Saquon Barkley?" in result.content[0].text  # type: ignore[union-attr]


async def test_compare_players_answers_for_every_open_slot() -> None:
    """Two backs, a tight end and Barkley for RB/RB/FLEX used to name one winner."""
    output = await _call(
        "compare_players",
        {"names": ["D'Andre Swift", "Cam Skattebo", "George Kittle", "Saquon Barkley"]},
    )
    assert output.startswith("Start D. Swift (RB), C. Skattebo (RB), G. Kittle (TE); sit S.")


async def test_simulation_size_is_clamped() -> None:
    assert (await _call("simulate_season", {"n_sims": 10**7})).startswith("50,000 sims")


async def test_free_agent_buy_lows_are_players_someone_would_start() -> None:
    """Four free-agent quarterbacks were listed in a one-quarterback league."""
    output = await _call("buy_low_sell_high")
    free_agents = output.split("Buy low (free agents):")[1]
    assert not re.search(r"\sQB\s", free_agents)


# --- P2: new capability ---------------------------------------------------------------------


async def test_roster_planner_finds_the_week_11_hole_and_the_spare_kicker() -> None:
    plan = await _call("roster_planner")
    assert "Wk 11: J. Love, D. London, C. Watson out" in plan
    assert "only starts to cover C. Boswell's wk-9 bye" in _row(plan, "E. McPherson")


async def test_league_activity_lists_recent_moves() -> None:
    activity = await _call("league_activity")
    assert "Team Lima (id 12): add Panthers D/ST, drop Giants D/ST" in activity


async def test_defense_pickups_show_their_matchups() -> None:
    table = await _call("find_waiver_targets", {"position": "D/ST"})
    assert "Matchups (opponent implied pts, lower is better): " in table


async def test_no_live_line_before_kickoff() -> None:
    """The snapshot was taken before Thursday's game, so nothing is live yet."""
    assert "Live win prob" not in await _call("analyze_matchup")


async def test_player_report_carries_market_value() -> None:
    report = await _call("player_report", {"name": "Mike Evans"})
    assert "trade value 3,190 (FantasyCalc)" in report


def test_the_snapshot_is_anonymized() -> None:
    from pathlib import Path

    league = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures" / "snapshot_league.json").read_text(
            encoding="utf-8"
        )
    )
    assert league["settings"]["league_id"] == 1234567
    assert all(team["name"].startswith("Team ") for team in league["teams"])

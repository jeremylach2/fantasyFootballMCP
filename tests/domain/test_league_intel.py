"""All-play, luck, and lineup accuracy from a league's completed weeks, plus the projection
accuracy table."""

from __future__ import annotations

import pytest

from ffmcp.domain.calibration import source_accuracy
from ffmcp.domain.league_intel import manager_reports
from ffmcp.domain.models import (
    LeagueSettings,
    LeagueState,
    PlayerWeek,
    Roster,
    Team,
)

SETTINGS = LeagueSettings(
    league_id=1,
    season=2026,
    team_count=3,
    playoff_team_count=2,
    reg_season_weeks=14,
    slot_counts={"WR": 1, "BE": 1},
)


def _team(team_id: int, wins: int, losses: int) -> Team:
    return Team(
        team_id=team_id,
        name=f"Team {team_id}",
        abbrev=f"T{team_id}",
        wins=wins,
        losses=losses,
        points_for=0.0,
        points_against=0.0,
        standing=team_id,
        playoff_pct=50.0,
        roster=Roster(slots=()),
    )


def _row(
    team_id: int, week: int, player_id: int, slot: str, projected: float, actual: float
) -> PlayerWeek:
    return PlayerWeek(
        season=2026,
        week=week,
        player_id=player_id,
        name=f"P{player_id}",
        position="WR",
        pro_team="DET",
        eligible_slots=("WR", "BE"),
        fantasy_team_id=team_id,
        slot=slot,
        projected=projected,
        actual=actual,
    )


def _history() -> list[PlayerWeek]:
    rows = []
    for week in (1, 2):
        # Team 1 is the best every week; team 3 the worst; team 2 benches its better player.
        rows += [_row(1, week, 11, "WR", 20.0, 30.0), _row(1, week, 12, "BE", 5.0, 4.0)]
        rows += [_row(2, week, 21, "WR", 8.0, 20.0), _row(2, week, 22, "BE", 12.0, 25.0)]
        rows += [_row(3, week, 31, "WR", 10.0, 10.0), _row(3, week, 32, "BE", 3.0, 2.0)]
    return rows


def test_all_play_luck_and_lineup_accuracy() -> None:
    # Team 3 went 2-0 despite scoring the fewest points every week: pure schedule luck.
    state = LeagueState(
        settings=SETTINGS,
        teams=(_team(1, 0, 2), _team(2, 0, 2), _team(3, 2, 0)),
        current_week=3,
    )
    reports = {r.team_id: r for r in manager_reports(state, _history())}

    assert [r.team_id for r in manager_reports(state, _history())] == [1, 2, 3]
    assert (reports[1].all_play_wins, reports[1].all_play_losses) == (4, 0)
    assert (reports[3].all_play_wins, reports[3].all_play_losses) == (0, 4)
    assert reports[3].luck == pytest.approx(2.0)
    assert reports[1].luck == pytest.approx(-2.0)
    assert "lucky" in reports[3].tags and "unlucky" in reports[1].tags

    # Team 2 started an 8-point projection over a 12: two-thirds accuracy, and inattentive.
    assert reports[2].lineup_accuracy == pytest.approx(100.0 * 8.0 / 12.0)
    assert "inattentive" in reports[2].tags
    assert reports[2].bench_points_per_game == pytest.approx(5.0)
    assert reports[1].lineup_accuracy == pytest.approx(100.0)


def test_source_accuracy_compares_sources_on_identical_player_weeks() -> None:
    history = [
        _row(1, 1, 11, "WR", 10.0, 14.0),
        _row(1, 1, 12, "WR", 10.0, 6.0),
        _row(1, 2, 11, "WR", 10.0, 10.0),
    ]
    secondary = {(1, 11): 14.0, (1, 12): 10.0}  # nothing for week 2
    rows = {r.position: r for r in source_accuracy(history, secondary)}

    overall = rows["ALL"]
    assert overall.n == 3
    assert overall.primary_mae == pytest.approx(8.0 / 3)
    assert overall.primary_bias == pytest.approx(0.0)
    assert overall.secondary_n == 2
    assert overall.primary_mae_matched == pytest.approx(4.0)
    assert overall.secondary_mae == pytest.approx(2.0)
    assert overall.blend_mae == pytest.approx(3.0)

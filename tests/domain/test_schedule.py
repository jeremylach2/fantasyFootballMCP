"""Rest-of-season value knows about byes and the playoffs."""

from __future__ import annotations

import pytest

from ffmcp.domain.models import LeagueSettings, Player, Projection
from ffmcp.domain.schedule import (
    effective_weeks,
    rate_for_week,
    remaining_week_weights,
    season_lineup_points,
    season_marginal_value,
    week_projections,
    weekly_rate,
)
from ffmcp.domain.trades import apply_offer

SETTINGS = LeagueSettings(
    league_id=1,
    season=2026,
    team_count=12,
    playoff_team_count=8,
    reg_season_weeks=14,
    slot_counts={"RB": 1, "BE": 3},
)


def _rb(player_id: int, points: float | None, bye: int | None, rate: float | None = None) -> Player:
    return Player(
        player_id=player_id,
        name=f"RB{player_id}",
        position="RB",
        eligible_slots=("RB",),
        pro_team="DET",
        projection=None if points is None else Projection(week=12, points=points),
        bye_week=bye,
        season_rate=rate,
    )


def test_playoff_weeks_follow_from_the_bracket_size() -> None:
    assert SETTINGS.playoff_weeks == (15, 16, 17)
    four_team = SETTINGS.model_copy(update={"playoff_team_count": 4})
    assert four_team.playoff_weeks == (15, 16)
    two_week_rounds = SETTINGS.model_copy(update={"playoff_round_weeks": 2})
    assert two_week_rounds.playoff_weeks == tuple(range(15, 21))


def test_playoff_weeks_are_weighted_by_the_chance_of_being_there() -> None:
    weights = remaining_week_weights(SETTINGS, 12, playoff_probability=0.25)
    assert [(w.week, w.weight) for w in weights] == [
        (12, 1.0),
        (13, 1.0),
        (14, 1.0),
        (15, 0.25),
        (16, 0.25),
        (17, 0.25),
    ]


def test_this_week_uses_the_matchup_and_later_weeks_the_season_average() -> None:
    soft_matchup = _rb(1, 11.0, bye=None, rate=7.0)
    assert rate_for_week(soft_matchup, 12) == 11.0
    assert rate_for_week(soft_matchup, 13) == 7.0
    assert weekly_rate(soft_matchup) == 7.0
    assert weekly_rate(_rb(2, None, bye=12, rate=15.0)) == 15.0
    assert weekly_rate(_rb(3, 9.0, bye=None)) == 9.0


def test_one_good_matchup_is_not_extrapolated_across_the_season() -> None:
    weights = remaining_week_weights(SETTINGS, 12, playoff_probability=0.0)
    hot_this_week = _rb(1, 14.0, bye=None, rate=6.0)
    assert season_lineup_points([hot_this_week], ("RB",), weights) == pytest.approx(14.0 + 2 * 6.0)


def test_a_one_week_injury_does_not_zero_the_rest_of_the_season() -> None:
    weights = remaining_week_weights(SETTINGS, 12, playoff_probability=0.0)
    out_this_week = _rb(1, 0.0, bye=None, rate=15.0).model_copy(update={"injury_status": "OUT"})
    assert season_lineup_points([out_this_week], ("RB",), weights) == pytest.approx(2 * 15.0)


def test_week_projections_leave_out_whoever_is_on_bye() -> None:
    players = [_rb(1, 20.0, bye=13), _rb(2, 10.0, bye=None)]
    assert week_projections(players, 13) == {2: 10.0}
    assert week_projections(players, 14) == {1: 20.0, 2: 10.0}


def test_a_backup_is_worth_exactly_the_bye_week_he_covers() -> None:
    weights = remaining_week_weights(SETTINGS, 12, playoff_probability=0.0)
    starter = _rb(1, 20.0, bye=13)
    backup = _rb(2, 10.0, bye=None)
    assert season_lineup_points([starter], ("RB",), weights) == pytest.approx(40.0)
    assert season_marginal_value([starter], backup, ("RB",), weights) == pytest.approx(10.0)


def test_effective_weeks_skip_the_bye_and_discount_the_playoffs() -> None:
    weights = remaining_week_weights(SETTINGS, 12, playoff_probability=0.5)
    assert effective_weeks(_rb(1, 10.0, bye=16), weights) == pytest.approx(3 + 0.5 * 2)


def test_a_roster_cut_never_drops_a_star_just_because_he_is_on_bye() -> None:
    star_on_bye = _rb(1, None, bye=12, rate=18.0)
    scrub = _rb(2, 3.0, bye=None)
    starter = _rb(3, 12.0, bye=None)
    arrival = _rb(4, 8.0, bye=None)
    _, drops = apply_offer([star_on_bye, scrub, starter], [], [arrival], ("RB",))
    assert [p.player_id for p in drops] == [2]

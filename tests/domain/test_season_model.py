"""The season simulation models each remaining week on its own terms, and the league's format.

``test_simulate.py`` covers the machinery (reproducibility, common random numbers, speed); this
file covers what the per-week model is *for*: a bye costs the week it falls in and no other,
the strength term keeps week-6 odds honest, and median games, divisions and two-week playoff
rounds are played the way the league plays them.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_simulate import make_league

import ffmcp.domain.simulate as simulate
from ffmcp.domain.models import LeagueState, Player, Projection, Roster, RosterSlot, Team
from ffmcp.domain.simulate import _build_season, playoff_odds_delta, simulate_rest_of_season


def _with_byes(state: LeagueState, team_id: int, positions: tuple[str, ...]) -> LeagueState:
    """``state`` with ``team_id``'s starters at ``positions`` on bye this week, the way ESPN
    reports it: projected 0.0, their season rate unchanged."""
    week = state.current_week

    def on_bye(player: Player) -> Player:
        if player.position not in positions:
            return player
        rate = player.projection.points if player.projection else None
        return player.model_copy(
            update={
                "bye_week": week,
                "season_rate": rate,
                "projection": Projection(week=week, points=0.0),
            }
        )

    teams = []
    for team in state.teams:
        if team.team_id == team_id:
            roster = team.roster
            team = team.model_copy(
                update={
                    "roster": Roster(
                        slots=tuple(
                            RosterSlot(slot=s.slot, player=on_bye(s.player) if s.player else None)
                            for s in roster.slots
                        ),
                        bench=roster.bench,
                    )
                }
            )
        teams.append(team)
    return state.model_copy(update={"teams": tuple(teams)})


def test_this_weeks_byes_cost_one_week_not_the_season() -> None:
    """The bug this model replaced: one team's optimal lineup *this week*, held constant all
    season, so three starters on bye now were on bye in every simulated week. In a live league
    that took a 54% team to 8%."""
    base = make_league([1.0] * 8, current_week=3, reg_season_weeks=12, playoff_team_count=4)
    bye = _with_byes(base, 1, ("QB", "RB", "WR"))

    healthy = simulate_rest_of_season(base, 4_000).team(1).playoff_odds
    with_byes = simulate_rest_of_season(bye, 4_000).team(1).playoff_odds

    assert with_byes < healthy  # a bye is still a real cost, this week
    assert healthy - with_byes < 12.0  # about one game's worth, not a season's
    assert with_byes > 0.6 * healthy


def test_a_projection_favourite_is_not_a_lock_in_midseason() -> None:
    state = make_league([1.15, 1.0, 1.0, 1.0, 1.0, 1.0], current_week=4, playoff_team_count=2)
    with_strength = simulate_rest_of_season(state, 4_000).team(1).playoff_odds

    original = simulate.TEAM_STRENGTH_SD
    try:
        simulate.TEAM_STRENGTH_SD = 0.0
        without = simulate_rest_of_season(state, 4_000).team(1).playoff_odds
    finally:
        simulate.TEAM_STRENGTH_SD = original

    assert with_strength < without
    assert with_strength < 95.0


def test_a_median_league_plays_two_games_a_week() -> None:
    state = make_league([1.0] * 6, current_week=5, reg_season_weeks=10)
    median = state.model_copy(
        update={"settings": state.settings.model_copy(update={"median_scoring": True})}
    )
    weeks = 10 - 5 + 1
    base_wins = sum(t.wins for t in state.teams)

    plain = simulate_rest_of_season(state, 1_000)
    doubled = simulate_rest_of_season(median, 1_000)

    assert sum(t.mean_final_wins for t in plain.teams) - base_wins == pytest.approx(3 * weeks)
    assert sum(t.mean_final_wins for t in doubled.teams) - base_wins == pytest.approx(6 * weeks)


def test_a_division_winner_makes_the_playoffs_whatever_his_record() -> None:
    state = make_league([0.8, 0.8, 1.2, 1.2], current_week=6, reg_season_weeks=10)
    divided = state.model_copy(
        update={
            "teams": tuple(
                t.model_copy(update={"division_id": 0 if t.team_id <= 2 else 1})
                for t in state.teams
            ),
            "settings": state.settings.model_copy(
                update={"playoff_team_count": 2, "division_count": 2}
            ),
        }
    )
    outcome = simulate_rest_of_season(divided, 2_000)
    weak = outcome.team(1).playoff_odds + outcome.team(2).playoff_odds
    assert weak == pytest.approx(100.0)  # exactly one of the weak division's two goes


def test_a_two_week_playoff_round_is_scored_over_both_weeks() -> None:
    state = make_league([1.0] * 4, current_week=10, reg_season_weeks=12, playoff_team_count=4)
    two_week = state.model_copy(
        update={"settings": state.settings.model_copy(update={"playoff_round_weeks": 2})}
    )
    one = _build_season(state)
    two = _build_season(two_week)
    assert one.round_weeks == ((13,), (14,))
    assert two.round_weeks == ((13, 14), (15, 16))
    assert np.allclose(two.baseline.round_means, 2 * one.baseline.round_means)


def test_trading_for_a_star_on_bye_this_week_is_priced_for_the_weeks_he_plays() -> None:
    """A trade's odds re-model the new roster week by week. Priced by this week alone, a star
    on bye contributed nothing and the trade looked worthless."""
    state = make_league([1.0] * 6, current_week=4, reg_season_weeks=12, playoff_team_count=2)
    star = Player(
        player_id=999,
        name="Star Receiver",
        position="WR",
        eligible_slots=("WR", "RB/WR/TE"),
        pro_team="DET",
        projection=Projection(week=4, points=0.0),
        bye_week=4,
        season_rate=24.0,
    )
    mine = [*state.team(1).roster.players, star]
    gain = playoff_odds_delta(state, 1, {1: mine}, n_sims=2_000)
    assert gain > 5.0


def test_the_strength_draw_is_shared_by_both_sides_of_a_comparison() -> None:
    """Common random numbers include the strength row: an unchanged roster is a zero delta."""
    state = make_league([1.0] * 6, current_week=4)
    unchanged = list(state.team(1).roster.players)
    assert playoff_odds_delta(state, 1, {1: unchanged}, n_sims=500) == 0.0


def test_teams_without_a_division_id_are_one_division() -> None:
    state = make_league([1.0] * 4)
    season = _build_season(state)
    assert set(season.division_of.tolist()) == {0}
    assert isinstance(state.teams[0], Team)

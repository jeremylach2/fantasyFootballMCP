"""The season simulation is reproducible, sane, fast, and, the point of the module, precise
enough to price a single lineup change, which it only is because of common random numbers.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from ffmcp.domain.models import (
    LeagueSettings,
    LeagueState,
    Lineup,
    Matchup,
    Player,
    Projection,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.domain.optimizer import optimize
from ffmcp.domain.simulate import (
    LineupScenario,
    simulate_rest_of_season,
    team_score_model,
    win_prob_delta,
)

SLOT_COUNTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "RB/WR/TE": 1,
    "D/ST": 1,
    "K": 1,
    "BE": 5,
}
STARTER_POSITIONS = ("QB", "RB", "RB", "WR", "WR", "TE", "WR", "D/ST", "K")
POSITION_WEIGHT = {"QB": 20.0, "RB": 14.0, "WR": 13.0, "TE": 9.0, "D/ST": 8.0, "K": 8.0}
ELIGIBLE = {
    "QB": ("QB",),
    "RB": ("RB", "RB/WR/TE"),
    "WR": ("WR", "RB/WR/TE"),
    "TE": ("TE", "RB/WR/TE"),
    "D/ST": ("D/ST",),
    "K": ("K",),
}


def make_team(team_id: int, *, strength: float, wins: int, losses: int) -> Team:
    """A team whose every starter projects ``strength`` times its positional baseline, so one
    number controls how good the team is."""
    slots: list[RosterSlot] = []
    for index, position in enumerate(STARTER_POSITIONS):
        points = round(POSITION_WEIGHT[position] * strength, 1)
        slots.append(
            RosterSlot(
                slot=position if position != "WR" or index != 6 else "RB/WR/TE",
                player=Player(
                    player_id=team_id * 100 + index,
                    name=f"T{team_id} {position}{index}",
                    position=position,
                    eligible_slots=ELIGIBLE[position],
                    pro_team="FA",
                    projection=Projection(week=1, points=points),
                ),
            )
        )
    bench = tuple(
        Player(
            player_id=team_id * 100 + 50 + index,
            name=f"T{team_id} backup {position}",
            position=position,
            eligible_slots=ELIGIBLE[position],
            pro_team="FA",
            projection=Projection(
                week=1, points=round(POSITION_WEIGHT[position] * strength - 3.0, 1)
            ),
        )
        for index, position in enumerate(("QB", "RB", "WR"))
    )
    return Team(
        team_id=team_id,
        name=f"Team {team_id}",
        abbrev=f"T{team_id}",
        wins=wins,
        losses=losses,
        points_for=100.0 * (wins + losses) * strength,
        points_against=100.0 * (wins + losses),
        standing=team_id,
        playoff_pct=50.0,
        roster=Roster(slots=tuple(slots), bench=bench),
    )


def round_robin(team_ids: list[int], weeks: range) -> list[Matchup]:
    """Circle-method round robin, repeated as often as the remaining weeks need."""
    rotation = list(team_ids)
    matchups: list[Matchup] = []
    for week in weeks:
        half = len(rotation) // 2
        for home, away in zip(rotation[:half], rotation[half:][::-1], strict=True):
            matchups.append(
                Matchup(
                    week=week,
                    home_team_id=home,
                    away_team_id=away,
                    home_score=0.0,
                    away_score=0.0,
                    home_projected=0.0,
                    away_projected=0.0,
                )
            )
        rotation = [rotation[0], *rotation[-1:], *rotation[1:-1]]
    return matchups


def make_league(
    strengths: list[float],
    *,
    records: list[tuple[int, int]] | None = None,
    current_week: int = 3,
    reg_season_weeks: int = 14,
    playoff_team_count: int = 4,
) -> LeagueState:
    records = records or [(1, 1)] * len(strengths)
    teams = [
        make_team(i + 1, strength=strength, wins=record[0], losses=record[1])
        for i, (strength, record) in enumerate(zip(strengths, records, strict=True))
    ]
    return LeagueState(
        settings=LeagueSettings(
            league_id=1234567,
            season=2026,
            team_count=len(teams),
            playoff_team_count=playoff_team_count,
            reg_season_weeks=reg_season_weeks,
            slot_counts=SLOT_COUNTS,
            scoring_type="PPR",
        ),
        teams=tuple(teams),
        current_week=current_week,
        matchups=tuple(
            round_robin([t.team_id for t in teams], range(current_week, reg_season_weeks + 1))
        ),
    )


def test_seeded_runs_are_exactly_reproducible() -> None:
    state = make_league([1.0, 0.95, 0.9, 0.85])
    first = simulate_rest_of_season(state, 500, np.random.default_rng(7))
    second = simulate_rest_of_season(state, 500, np.random.default_rng(7))
    assert first == second

    # And the fixed default seed makes the no-argument call reproducible too, deliberately.
    assert simulate_rest_of_season(state, 500) == simulate_rest_of_season(state, 500)

    different = simulate_rest_of_season(state, 500, np.random.default_rng(8))
    assert different != first


def test_a_dominant_undefeated_team_outranks_a_hopeless_winless_one() -> None:
    state = make_league(
        [1.35, 1.0, 1.0, 0.6],
        records=[(3, 0), (2, 1), (1, 2), (0, 3)],
        current_week=4,
        playoff_team_count=2,
    )
    outcome = simulate_rest_of_season(state, 2_000, np.random.default_rng(1))
    best, worst = outcome.team(1), outcome.team(4)

    assert best.playoff_odds > worst.playoff_odds
    assert best.mean_final_wins > worst.mean_final_wins
    assert best.title_odds > worst.title_odds
    assert best.seed_distribution[0] > worst.seed_distribution[0]


def test_playoff_odds_sum_to_the_number_of_playoff_spots() -> None:
    for playoff_spots in (2, 4, 6):
        state = make_league(
            [1.1, 1.0, 1.0, 0.95, 0.9, 0.9, 0.85, 0.8], playoff_team_count=playoff_spots
        )
        outcome = simulate_rest_of_season(state, 1_000, np.random.default_rng(3))
        total = sum(team.playoff_odds for team in outcome.teams)
        assert total == pytest.approx(100.0 * playoff_spots, abs=1e-6)
        assert sum(team.title_odds for team in outcome.teams) == pytest.approx(100.0, abs=1e-6)
        # Every team lands in exactly one seed in every simulated season.
        for team in outcome.teams:
            assert sum(team.seed_distribution) == pytest.approx(100.0, abs=1e-6)


def better_and_worse_lineups(state: LeagueState) -> tuple[Lineup, Lineup]:
    """The optimal lineup, and the best lineup available with the starting QB benched.

    The gap is 3 points out of ~115, the size of a real start/sit decision, which is the whole
    reason the delta needs variance reduction to be visible at all.
    """
    team = state.teams[0]
    slots = state.settings.starting_slots
    optimal = optimize(team.roster.players, slots)
    started_qb = next(p for p in optimal.started if p.position == "QB")
    weakened = optimize(
        [p for p in team.roster.players if p.player_id != started_qb.player_id], slots
    )
    assert 2.0 < optimal.projected_points - weakened.projected_points < 4.0
    return optimal, weakened


def test_common_random_numbers_make_a_lineup_gain_measurable() -> None:
    """This test *is* the argument for common random numbers.

    The counterfactual lineup is strictly better than the baseline in one week. Under shared
    draws every simulated season is identical except for that team's score in that one week, so
    the measured delta can never be negative and is positive whenever any season flips. Under
    independent draws the same comparison is two noisy estimates subtracted: the sign is close
    to a coin flip and the spread is an order of magnitude wider than the effect being measured.
    """
    state = make_league([1.0, 1.0, 1.0, 1.0], current_week=3, playoff_team_count=2)
    optimal, weakened = better_and_worse_lineups(state)
    baseline = LineupScenario(team_id=1, week=3, lineup=weakened)
    counterfactual = LineupScenario(team_id=1, week=3, lineup=optimal)

    def deltas(*, common: bool) -> list[float]:
        return [
            win_prob_delta(
                state,
                baseline,
                counterfactual,
                np.random.default_rng(seed),
                n_sims=800,
                common_random_numbers=common,
            )
            for seed in range(100)
        ]

    shared = deltas(common=True)
    independent = deltas(common=False)

    assert all(delta >= 0.0 for delta in shared), "CRN can never price a strict gain as a loss"
    assert sum(delta > 0.0 for delta in shared) >= 95

    assert statistics.pstdev(independent) > 3 * statistics.pstdev(shared)
    assert sum(delta < 0.0 for delta in independent) > 5, (
        "independent draws should get the sign wrong often enough to be useless"
    )


def test_win_prob_delta_rejects_mismatched_or_unscheduled_scenarios() -> None:
    state = make_league([1.0, 1.0, 1.0, 1.0], current_week=3)
    optimal, weakened = better_and_worse_lineups(state)

    with pytest.raises(ValueError, match="same team"):
        win_prob_delta(
            state,
            LineupScenario(team_id=1, week=3, lineup=weakened),
            LineupScenario(team_id=2, week=3, lineup=optimal),
        )
    with pytest.raises(ValueError, match="remaining schedule"):
        win_prob_delta(
            state,
            LineupScenario(team_id=1, week=1, lineup=weakened),
            LineupScenario(team_id=1, week=1, lineup=optimal),
        )


def test_progress_is_reported_about_every_ten_percent() -> None:
    state = make_league([1.0, 1.0, 1.0, 1.0])
    calls: list[tuple[int, int]] = []

    def record(done: int, total: int) -> None:
        calls.append((done, total))

    simulate_rest_of_season(state, 1_000, np.random.default_rng(0), progress=record)

    assert calls[-1] == (1_000, 1_000)
    assert len(calls) == 10
    assert calls == sorted(calls)


def test_ten_thousand_sims_for_a_twelve_team_league_run_in_under_two_seconds() -> None:
    state = make_league([1.0 + 0.02 * i for i in range(12)], current_week=3, reg_season_weeks=14)
    started = time.perf_counter()
    outcome = simulate_rest_of_season(state, 10_000, np.random.default_rng(42))
    elapsed = time.perf_counter() - started

    assert len(outcome.teams) == 12
    assert elapsed < 2.0, f"10,000 sims took {elapsed:.2f}s"


def test_team_score_model_tracks_the_optimal_lineup() -> None:
    state = make_league([1.0, 0.5, 1.0, 1.0])
    strong = team_score_model(state.teams[0], state.settings.starting_slots)
    weak = team_score_model(state.teams[1], state.settings.starting_slots)

    assert strong.mean > weak.mean
    assert strong.sd > weak.sd > 0.0
    # A full starting lineup should land in the range real PPR team-weeks actually show.
    assert 20.0 < strong.sd < 40.0


def test_a_league_with_no_remaining_games_still_produces_odds() -> None:
    state = make_league([1.0, 0.9, 0.8, 0.7], current_week=15, reg_season_weeks=14)
    outcome = simulate_rest_of_season(state, 200, np.random.default_rng(5))
    assert outcome.weeks_simulated == ()
    assert sum(team.playoff_odds for team in outcome.teams) == pytest.approx(400.0, abs=1e-6)

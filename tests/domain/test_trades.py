"""Trades are valued from both sides, fleeces never surface, and the search
funnel stays inside its time budget.

The league below is built to contain exactly the trade managers actually make: a team stuck
with two elite quarterbacks in a one-QB league and nothing at running back, across the table
from a team with a weak quarterback and more backs than it can start.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ffmcp.domain.models import (
    LeagueSettings,
    LeagueState,
    Matchup,
    Player,
    Projection,
    Roster,
    RosterSlot,
    Team,
    TradeOffer,
)
from ffmcp.domain.trades import (
    apply_offer,
    evaluate,
    marginal_value,
    positional_impact,
    search,
)
from ffmcp.providers.demo import DemoLeagueProvider

SLOT_COUNTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "BE": 4}
ELIGIBLE = {
    "QB": ("QB",),
    "RB": ("RB", "RB/WR/TE"),
    "WR": ("WR", "RB/WR/TE"),
    "TE": ("TE", "RB/WR/TE"),
}
SLOTS = ("QB", "RB", "RB", "WR", "WR", "TE", "RB/WR/TE")

# (position, projection) per team, best first within a position.
QB_RICH_RB_POOR = [
    ("QB", 26.0),
    ("QB", 24.0),
    ("RB", 6.0),
    ("RB", 5.0),
    ("RB", 4.0),
    ("WR", 18.0),
    ("WR", 16.0),
    ("WR", 12.0),
    ("TE", 10.0),
]
RB_RICH_QB_POOR = [
    ("QB", 9.0),
    ("RB", 20.0),
    ("RB", 17.0),
    ("RB", 15.0),
    ("RB", 14.0),
    ("WR", 13.0),
    ("WR", 11.0),
    ("TE", 9.0),
    ("TE", 8.0),
]
BALANCED = [
    ("QB", 18.0),
    ("RB", 13.0),
    ("RB", 12.0),
    ("RB", 9.0),
    ("WR", 15.0),
    ("WR", 14.0),
    ("WR", 10.0),
    ("TE", 11.0),
    ("TE", 7.0),
]


def make_team(team_id: int, roster: list[tuple[str, float]], *, wins: int = 1) -> Team:
    players = [
        Player(
            player_id=team_id * 100 + index,
            name=f"T{team_id} {position}{index}",
            position=position,
            eligible_slots=ELIGIBLE[position],
            pro_team="FA",
            projection=Projection(week=3, points=points),
        )
        for index, (position, points) in enumerate(roster)
    ]
    return Team(
        team_id=team_id,
        name=f"Team {team_id}",
        abbrev=f"T{team_id}",
        wins=wins,
        losses=2 - wins,
        points_for=220.0,
        points_against=210.0,
        standing=team_id,
        playoff_pct=50.0,
        # Everyone is on the bench to start with; the optimizer decides who plays.
        roster=Roster(slots=(RosterSlot(slot="QB", player=players[0]),), bench=tuple(players[1:])),
    )


def make_state(current_week: int = 3, reg_season_weeks: int = 8) -> LeagueState:
    teams = (
        make_team(1, QB_RICH_RB_POOR),
        make_team(2, RB_RICH_QB_POOR),
        make_team(3, BALANCED),
        make_team(4, BALANCED, wins=2),
    )
    matchups = [
        Matchup(
            week=week,
            home_team_id=home,
            away_team_id=away,
            home_score=0.0,
            away_score=0.0,
            home_projected=0.0,
            away_projected=0.0,
        )
        for week in range(current_week, reg_season_weeks + 1)
        for home, away in (((1, 2), (3, 4)) if week % 2 else ((1, 3), (2, 4)))
    ]
    return LeagueState(
        settings=LeagueSettings(
            league_id=1234567,
            season=2026,
            team_count=4,
            playoff_team_count=2,
            reg_season_weeks=reg_season_weeks,
            slot_counts=SLOT_COUNTS,
            scoring_type="PPR",
        ),
        teams=teams,
        current_week=current_week,
        matchups=tuple(matchups),
    )


def player_at(state: LeagueState, team_id: int, position: str, rank: int = 0) -> Player:
    """The ``rank``-th best player at a position on a team, counting from zero."""
    ranked = sorted(
        (p for p in state.team(team_id).roster.players if p.position == position),
        key=lambda p: -(p.projection.points if p.projection else 0.0),
    )
    return ranked[rank]


def test_marginal_value_prices_a_player_against_the_roster_not_the_projection() -> None:
    """A 24-point backup quarterback is worth nothing; a 15-point back is worth ten."""
    state = make_state()
    mine = list(state.team(1).roster.players)

    backup_qb = player_at(state, 1, "QB", rank=1)
    assert backup_qb.projection is not None and backup_qb.projection.points == 24.0
    assert marginal_value(mine, backup_qb, SLOTS) == 0.0

    starter_qb = player_at(state, 1, "QB", rank=0)
    assert marginal_value(mine, starter_qb, SLOTS) == 2.0  # the drop to his own backup

    their_back = player_at(state, 2, "RB", rank=2)
    assert their_back.projection is not None and their_back.projection.points == 15.0
    assert marginal_value(mine, their_back, SLOTS) == 10.0


def test_evaluate_prices_both_sides_of_a_trade() -> None:
    state = make_state()
    offer = TradeOffer(
        partner_team_id=2,
        give=(player_at(state, 1, "QB", rank=1),),
        get=(player_at(state, 2, "RB", rank=2),),
    )
    result = evaluate(offer, state.team(1).roster, state.team(2).roster, SLOTS)

    assert result.my_value_delta == 10.0
    assert result.partner_value_delta == 14.0
    assert result.mutually_beneficial


def test_a_fleece_is_never_mutually_beneficial() -> None:
    state = make_state()
    fleece = TradeOffer(
        partner_team_id=2,
        give=(player_at(state, 1, "RB", rank=2),),  # my worst back
        get=(player_at(state, 2, "RB", rank=0),),  # their best
    )
    result = evaluate(fleece, state.team(1).roster, state.team(2).roster, SLOTS)

    assert result.my_value_delta > 0
    assert result.partner_value_delta < 0
    assert not result.mutually_beneficial


def test_evaluate_is_antisymmetric_under_undoing_the_trade() -> None:
    """Reversing give and get on the post-trade rosters returns exactly the negated deltas."""
    state = make_state()
    mine, theirs = list(state.team(1).roster.players), list(state.team(2).roster.players)
    offer = TradeOffer(
        partner_team_id=2,
        give=(player_at(state, 1, "QB", rank=1),),
        get=(player_at(state, 2, "RB", rank=2),),
    )
    forward = evaluate(offer, mine, theirs, SLOTS)

    mine_after, _ = apply_offer(mine, offer.give, offer.get, SLOTS)
    theirs_after, _ = apply_offer(theirs, offer.get, offer.give, SLOTS)
    undo = evaluate(offer.mirrored, mine_after, theirs_after, SLOTS)

    assert undo.my_value_delta == -forward.my_value_delta
    assert undo.partner_value_delta == -forward.partner_value_delta


def test_evaluate_gives_the_same_answer_from_the_other_side_of_the_table() -> None:
    state = make_state()
    offer = TradeOffer(
        partner_team_id=2,
        give=(player_at(state, 1, "QB", rank=1),),
        get=(player_at(state, 2, "RB", rank=2),),
    )
    mine, theirs = state.team(1).roster, state.team(2).roster
    forward = evaluate(offer, mine, theirs, SLOTS)
    from_their_seat = evaluate(offer.mirrored, theirs, mine, SLOTS)

    assert from_their_seat.my_value_delta == forward.partner_value_delta
    assert from_their_seat.partner_value_delta == forward.my_value_delta


def test_search_offers_the_quarterback_surplus_team_a_running_back() -> None:
    state = make_state()
    results = search(state, my_team_id=1, rng=np.random.default_rng(1), n_sims=400)

    assert results
    qb_for_rb = [
        result
        for result in results
        if any(p.position == "QB" for p in result.offer.give)
        and any(p.position == "RB" for p in result.offer.get)
    ]
    assert qb_for_rb, "the obvious trade was not found"

    best = results[0]
    assert best.my_value_delta > 0
    assert best.partner_value_delta > 0
    assert best.my_playoff_odds_delta is not None


def test_search_never_returns_a_one_sided_offer() -> None:
    state = make_state()
    results = search(state, my_team_id=1, max_results=20, rng=np.random.default_rng(2), n_sims=400)

    assert results
    for result in results:
        assert result.mutually_beneficial
        assert result.partner_value_delta > 0, "a fleece reached the output"


def test_search_respects_partner_and_position_filters() -> None:
    state = make_state()
    results = search(
        state,
        my_team_id=1,
        partner_team_id=2,
        positions_wanted=["RB"],
        rng=np.random.default_rng(3),
        n_sims=400,
    )

    assert results
    for result in results:
        assert result.offer.partner_team_id == 2
        assert all(player.position == "RB" for player in result.offer.get)


def test_search_reports_progress_and_is_reproducible() -> None:
    state = make_state()
    calls: list[tuple[int, int]] = []

    def record(done: int, total: int) -> None:
        calls.append((done, total))

    first = search(state, my_team_id=1, rng=np.random.default_rng(4), n_sims=400, progress=record)
    second = search(state, my_team_id=1, rng=np.random.default_rng(4), n_sims=400)

    assert first == second
    assert calls and calls[-1][0] == calls[-1][1]


def test_a_two_for_one_forces_a_drop_rather_than_a_free_roster_spot() -> None:
    state = make_state()
    mine = list(state.team(1).roster.players)
    offer = TradeOffer(
        partner_team_id=2,
        give=(player_at(state, 1, "QB", rank=1),),
        get=(player_at(state, 2, "RB", rank=2), player_at(state, 2, "RB", rank=3)),
    )
    after, drops = apply_offer(mine, offer.give, offer.get, SLOTS)

    assert len(after) == len(mine)
    assert len(drops) == 1
    # The cut comes off the bench, so it costs the lineup nothing.
    assert drops[0].projection is not None and drops[0].projection.points == 4.0


async def test_search_on_the_fixture_league_completes_well_inside_five_seconds() -> None:
    provider = DemoLeagueProvider()
    week = await provider.get_current_week()
    state = LeagueState(
        settings=await provider.get_settings(),
        teams=tuple(await provider.get_teams()),
        current_week=week,
        matchups=tuple(await provider.get_matchups(week)),
    )

    started = time.perf_counter()
    results = search(state, my_team_id=state.teams[0].team_id, rng=np.random.default_rng(0))
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"search took {elapsed:.2f}s"
    # The committed fixture is a placeholder whose starters all out-project their own benches,
    # so it holds no surplus to trade and the honest answer is an empty list. This is a timing
    # and no-crash guard against real league shapes; the behaviour lives in the tests above.
    assert all(result.mutually_beneficial for result in results)


def test_search_on_a_league_with_nothing_to_offer_returns_nothing() -> None:
    """Four identical rosters have no asymmetry to exploit, and silence is the right answer."""
    state = make_state()
    identical = tuple(
        team.model_copy(update={"roster": state.teams[2].roster}) for team in state.teams
    )
    flat = state.model_copy(update={"teams": identical})

    assert search(flat, my_team_id=1, rng=np.random.default_rng(5), n_sims=200) == []


def test_search_rejects_a_team_id_that_is_not_in_the_league() -> None:
    with pytest.raises(KeyError):
        search(make_state(), my_team_id=99)


def test_positional_impact_ignores_a_backup_who_was_never_going_to_start() -> None:
    """The whole point of measuring impact through the lineup.

    Team 1 starts one quarterback and rosters two. Trading the second away costs the lineup
    nothing at QB, however many points he is projected for; the trade's real effect is at
    running back. Summing the traded players' raw projections would report the opposite.
    """
    state = make_state()
    mine = list(state.team(1).roster.players)
    backup_qb = player_at(state, 1, "QB", rank=1)
    their_rb = player_at(state, 2, "RB", rank=1)

    deltas = positional_impact(mine, [backup_qb], [their_rb], SLOTS)

    assert "QB" not in deltas, f"benched backup QB should not move the lineup: {deltas}"
    assert deltas["RB"] > 0.0


def test_search_does_not_repeat_one_trade_padded_with_worthless_bench_players() -> None:
    """Adding a zero-marginal-value body to either side leaves both sides' numbers untouched,
    so the package builder can emit the same trade several times over. A manager should see it
    once, in its smallest form."""
    state = make_state()
    results = search(state, 1, max_results=5, rng=np.random.default_rng(7), n_sims=200)

    seen = {
        (
            evaluation.offer.partner_team_id,
            round(evaluation.my_value_delta, 2),
            round(evaluation.partner_value_delta, 2),
        )
        for evaluation in results
    }
    assert len(seen) == len(results), f"duplicate offers by value: {results}"

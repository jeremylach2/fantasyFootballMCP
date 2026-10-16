"""Pure helpers in the adapter layer: lineup-lock detection and format caveats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ffmcp.domain.models import LeagueSettings, Player, Projection, RosterSlot
from ffmcp.mcp._shared import lineup_locks, quarterback_count, week_note
from ffmcp.mcp.tools_league import format_caveats

NOW = datetime(2026, 10, 18, 18, 0, tzinfo=UTC)


def _wr(player_id: int, kickoff: datetime | None) -> Player:
    return Player(
        player_id=player_id,
        name=f"WR{player_id}",
        position="WR",
        eligible_slots=("WR", "RB/WR/TE"),
        pro_team="DET",
        projection=Projection(week=6, points=10.0),
        kickoff=kickoff,
    )


def test_started_games_lock_their_players_where_they_are() -> None:
    early, late = NOW - timedelta(hours=1), NOW + timedelta(hours=3)
    slots = ("WR", "WR", "RB/WR/TE")
    roster_slots = (
        RosterSlot(slot="RB/WR/TE", player=_wr(1, early)),
        RosterSlot(slot="WR", player=_wr(2, late)),
        RosterSlot(slot="WR", player=_wr(3, early)),
    )
    bench = (_wr(4, early), _wr(5, late), _wr(6, None))
    pinned, frozen = lineup_locks(roster_slots, bench, slots, NOW)
    assert pinned == {1: 2, 3: 0}  # indexes into the declared slot order, not ESPN's
    assert frozen == {4}


def test_a_superflex_league_counts_two_quarterbacks() -> None:
    base = dict(league_id=1, season=2026, team_count=12, playoff_team_count=6, reg_season_weeks=14)
    assert quarterback_count(LeagueSettings(**base, slot_counts={"QB": 1})) == 1  # type: ignore[arg-type]
    assert quarterback_count(LeagueSettings(**base, slot_counts={"QB": 1, "OP": 1})) == 2  # type: ignore[arg-type]


def test_only_later_weeks_carry_the_season_average_caveat() -> None:
    assert week_note(6, 6) is None
    assert week_note(5, 6) is None
    note = week_note(11, 6)
    assert note is not None and note.startswith("Week 11 uses season-average projections")


def test_an_unmodelled_tiebreak_is_said_once() -> None:
    base = dict(
        league_id=1,
        season=2026,
        team_count=12,
        playoff_team_count=6,
        reg_season_weeks=14,
        slot_counts={"QB": 1},
    )
    assert format_caveats(LeagueSettings(**base, playoff_seed_tie_rule="TOTAL_POINTS_SCORED")) == []  # type: ignore[arg-type]
    caveats = format_caveats(LeagueSettings(**base, playoff_seed_tie_rule="H2H_RECORD"))  # type: ignore[arg-type]
    assert caveats == [
        "Tied records are seeded by total points here; this league uses h2h record, so close "
        "seeds are approximate."
    ]

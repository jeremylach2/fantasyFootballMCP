"""Pickups, drops and byes, priced over the rest of the season (``domain.roster``)."""

from __future__ import annotations

from ffmcp.domain.models import Player, Projection
from ffmcp.domain.roster import (
    best_drop,
    bye_weeks,
    drop_candidates,
    rank_pickups,
    rank_stashes,
)
from ffmcp.domain.schedule import WeekWeight

SLOTS = ("QB", "RB", "RB", "D/ST", "K")
WEEK = 6
WEIGHTS = tuple(WeekWeight(week=w, weight=1.0) for w in range(WEEK, 15))
ELIGIBLE = {"QB": ("QB",), "RB": ("RB",), "K": ("K",), "D/ST": ("D/ST",)}


def _p(
    player_id: int,
    position: str,
    rate: float,
    *,
    bye: int | None = None,
    now: float | None = None,
    status: str | None = None,
    owned: float | None = None,
    outlook: tuple[tuple[int, float], ...] = (),
) -> Player:
    from ffmcp.domain.models import MarketSignal

    points = rate if now is None else now
    return Player(
        player_id=player_id,
        name=f"{position} {player_id}",
        position=position,
        eligible_slots=ELIGIBLE[position],
        pro_team="DET",
        projection=Projection(week=WEEK, points=0.0 if bye == WEEK else points),
        bye_week=bye,
        season_rate=rate,
        injury_status=status,
        injured=status is not None,
        market=MarketSignal(percent_owned=owned) if owned is not None else None,
        outlook=tuple(Projection(week=w, points=pts) for w, pts in outlook),
    )


def _roster() -> list[Player]:
    """Two kickers and two defenses, the shape the audit found: the starting kicker's bye is
    week 9, the backup kicker's this week; the backup defense is just worse."""
    return [
        _p(1, "QB", 22.0, bye=11),
        _p(2, "RB", 15.0, bye=10),
        _p(3, "RB", 13.0, bye=8),
        _p(4, "RB", 8.0, bye=12),
        _p(10, "D/ST", 5.4, bye=WEEK),
        _p(11, "D/ST", 2.8, bye=13, now=5.7),
        _p(20, "K", 9.2, bye=9),
        _p(21, "K", 7.7, bye=WEEK),
    ]


def test_a_kicker_pickup_cuts_the_spare_kicker_not_a_defense() -> None:
    """The replaced rule cut the lowest season rate on the roster, which is always a defense."""
    kicker = _p(30, "K", 9.9, bye=7)
    drop, _ = best_drop(_roster(), kicker, SLOTS, WEIGHTS)
    assert drop.player_id == 21


def test_pickups_are_ranked_by_the_season_and_never_cost_it() -> None:
    candidates = [
        _p(30, "K", 9.9, bye=7),  # +0.7/wk over the starter, and covers his bye
        _p(31, "RB", 14.0, bye=9),  # a real upgrade on RB2/RB3
        _p(32, "D/ST", 1.0, bye=14),  # worthless
    ]
    pickups, near_misses = rank_pickups(_roster(), candidates, SLOTS, WEIGHTS, limit=5)
    assert [p.player.player_id for p in pickups] == [31, 30]
    assert all(p.season_value > 0 for p in pickups)
    assert all(p.drop is not None and p.drop.player_id != p.player.player_id for p in pickups)
    assert near_misses == []


def test_when_nothing_helps_the_closest_moves_are_shown_best_first() -> None:
    candidates = [_p(40, "RB", 3.0, owned=1.0), _p(41, "RB", 3.0, owned=30.0)]
    pickups, near_misses = rank_pickups(_roster(), candidates, SLOTS, WEIGHTS, limit=5)
    assert pickups == []
    # Tied on value: more owned first, never the lower player id.
    assert [p.player.player_id for p in near_misses] == [41, 40]


def test_pruning_by_upper_bound_matches_pricing_everyone() -> None:
    candidates = [_p(50 + i, "RB", 8.0 + i, bye=7 + i % 6) for i in range(12)]
    top, _ = rank_pickups(_roster(), candidates, SLOTS, WEIGHTS, limit=3)
    every, _ = rank_pickups(_roster(), candidates, SLOTS, WEIGHTS, limit=len(candidates))
    assert [p.player.player_id for p in top] == [p.player.player_id for p in every[:3]]


def test_an_ir_stash_is_valued_from_his_expected_return() -> None:
    back = _p(60, "RB", 0.0, status="INJURY_RESERVE", outlook=((7, 0.0), (8, 0.0), (9, 16.0)))
    for week in range(10, 15):
        back = back.model_copy(
            update={"outlook": (*back.outlook, Projection(week=week, points=16.0))}
        )
    gone = _p(61, "RB", 0.0, status="INJURY_RESERVE", owned=40.0)
    stashes = rank_stashes(_roster(), [gone, back], SLOTS, WEIGHTS, limit=2)
    assert [s.player.player_id for s in stashes] == [60, 61]
    assert stashes[0].season_value > 0
    assert stashes[1].season_value == 0


def test_drop_candidates_say_whose_bye_a_backup_covers() -> None:
    drops = drop_candidates(_roster(), SLOTS, WEIGHTS, WEEK, limit=10)
    by_id = {d.player.player_id: d for d in drops}
    spare_kicker = by_id[21]
    assert spare_kicker.weeks_started == (9,)
    assert [(week, p.player_id) for week, p in spare_kicker.covers] == [(9, 20)]
    # This week's starters are never drop candidates.
    assert 11 not in by_id  # the backup defense starts this week, the first one's bye


def test_bye_weeks_name_the_regular_starters_missing() -> None:
    byes = bye_weeks(_roster(), SLOTS, range(WEEK, 15), WEEK)
    by_week = {b.week: b for b in byes}
    assert [p.player_id for p in by_week[11].missing] == [1]
    assert by_week[11].gap > 0
    assert 7 not in by_week  # nobody who starts is off in week 7

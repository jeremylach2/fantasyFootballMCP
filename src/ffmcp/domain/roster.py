"""Roster planning: who to pick up, who to cut, and where the bye weeks bite.

Every number here is a rest-of-season one, through ``domain.schedule``: each remaining week's
optimal lineup, each player absent in his bye week, playoff weeks weighted by the team's odds of
being there. That is what makes these three questions answerable at all:

* **Pickups** are ranked by what the whole move (the add *and* its drop) does to the season,
  not by this week. In a deep league almost no free agent beats a starter this week, so a
  this-week ranking shows nothing but kickers and streaming defenses, while the moves that matter
  (bye cover, a back returning from injury) never appear. This week still counts: it is the
  first week of the season sum.
* **The drop** is whichever roster player the season misses least, tried one at a time. The rule
  it replaced (lowest season rate, any position) cut the backup defense for every pickup,
  because defenses score least, whatever the roster needed.
* **Drop candidates** are bench players ranked by the same measure, with the weeks they would
  actually start, so "your second kicker only matters in week 9" is stated, not left to guess.
* **Byes** are each remaining week's optimal lineup against a typical week's, with the regular
  starters missing named.
"""

from __future__ import annotations

from collections.abc import Sequence

from ffmcp.domain.models import Frozen, Player
from ffmcp.domain.optimizer import is_ruled_out, optimize, projected_points
from ffmcp.domain.schedule import (
    WeekWeight,
    healthy,
    season_lineup_points,
    week_lineup,
    weekly_rate,
)

STASH_STATUSES = frozenset({"INJURY_RESERVE", "IR"})
"""Designations that let a player go straight into an IR slot, no drop needed."""

MIN_SEASON_GAIN = 0.5
"""Rest-of-season points a pickup must add to be listed: the same floor trades use. A move
worth a third of a point over ten weeks is a rounding error, not advice."""


class Pickup(Frozen):
    """One free agent, priced as a complete move for this roster."""

    player: Player
    week_value: float
    """Optimal-lineup points the move adds this week (add and drop together)."""
    season_value: float
    """The same over the rest of the season: the ranking key."""
    drop: Player | None = None
    """The roster player the move cuts, or ``None`` when no cut is needed (a stash into an open
    IR slot)."""


class DropCandidate(Frozen):
    """A roster player and what losing him would cost."""

    player: Player
    season_value: float
    """Weighted lineup points the rest of the season loses without him."""
    weeks_started: tuple[int, ...]
    """Remaining weeks in which he would be in the optimal lineup."""
    covers: tuple[tuple[int, Player], ...] = ()
    """``(week, player)`` for each of those weeks where he stands in for a regular starter at
    his position who is on bye."""


class ByeWeek(Frozen):
    """One remaining week, against a typical one."""

    week: int
    projected: float
    typical: float
    missing: tuple[Player, ...]
    """Regular starters on bye this week."""

    @property
    def gap(self) -> float:
        return self.typical - self.projected


def best_drop(
    roster: Sequence[Player],
    add: Player,
    slots: Sequence[str],
    weights: Sequence[WeekWeight],
) -> tuple[Player, float]:
    """The roster player to cut for ``add``, and the season points after the move.

    Tried one at a time: whoever the season misses least goes. This week's starters are only
    considered when nobody else is left, and ``add`` himself never is (cutting the player you
    just claimed is not a move). Ties go to the lower rest-of-season rate, then the lower id,
    so the answer is stable.
    """
    after = [*roster, add]
    starting = {p.player_id for p in optimize(after, slots).started}
    pool = [p for p in roster if p.player_id not in starting] or list(roster)
    scored = [
        (
            season_lineup_points([q for q in after if q.player_id != p.player_id], slots, weights),
            p,
        )
        for p in pool
    ]
    points, victim = max(
        scored,
        key=lambda pair: (round(pair[0], 4), -(weekly_rate(pair[1]) or 0.0), -pair[1].player_id),
    )
    return victim, points


def rank_pickups(
    roster: Sequence[Player],
    candidates: Sequence[Player],
    slots: Sequence[str],
    weights: Sequence[WeekWeight],
    *,
    limit: int,
) -> tuple[list[Pickup], list[Pickup]]:
    """``(pickups, near_misses)``: the best ``limit`` moves that help the season, and, when
    there are none, the few that come closest.

    Exact for every listed move, but not computed for every candidate: adding a player with no
    drop is an upper bound on any move that adds him (a drop can only cost), so candidates are
    priced in order of that bound and the search stops once the bound cannot beat the moves
    already found. Ties sort by this week's projection, then ownership, never by id, which
    would favour the oldest veterans on the wire.
    """
    if not candidates:
        return [], []
    base_season = season_lineup_points(roster, slots, weights)
    base_week = optimize(roster, slots).projected_points

    def bound(player: Player) -> float:
        return season_lineup_points([*roster, player], slots, weights) - base_season

    ordered = sorted(
        ((bound(player), player) for player in candidates),
        key=lambda pair: (-round(pair[0], 4), *_tiebreak(pair[1])),
    )

    priced: list[Pickup] = []
    for upper, player in ordered:
        good = sorted(
            (p.season_value for p in priced if p.season_value >= MIN_SEASON_GAIN), reverse=True
        )
        if len(good) >= limit and upper <= good[limit - 1]:
            break
        if not good and len(priced) >= limit and upper < MIN_SEASON_GAIN:
            break
        drop, after_points = best_drop(roster, player, slots, weights)
        after = [p for p in (*roster, player) if p.player_id != drop.player_id]
        priced.append(
            Pickup(
                player=player,
                week_value=round(optimize(after, slots).projected_points - base_week, 4),
                season_value=round(after_points - base_season, 4),
                drop=drop,
            )
        )
    priced.sort(key=lambda p: (-p.season_value, *_tiebreak(p.player)))
    pickups = [p for p in priced if p.season_value >= MIN_SEASON_GAIN][:limit]
    near_misses = [] if pickups else priced[:3]
    return pickups, near_misses


def rank_stashes(
    roster: Sequence[Player],
    candidates: Sequence[Player],
    slots: Sequence[str],
    weights: Sequence[WeekWeight],
    *,
    limit: int,
) -> list[Pickup]:
    """Injured-reserve free agents worth holding in an open IR slot: no drop now, valued at what
    they add once back (their outlook supplies the return week). Best first."""
    base_season = season_lineup_points(roster, slots, weights)
    stashes = [
        Pickup(
            player=player,
            week_value=0.0,
            season_value=round(
                season_lineup_points([*roster, player], slots, weights) - base_season, 4
            ),
        )
        for player in candidates
        if (player.injury_status or "").upper() in STASH_STATUSES
    ]
    stashes.sort(key=lambda p: (-p.season_value, *_tiebreak(p.player)))
    return stashes[:limit]


def drop_candidates(
    roster: Sequence[Player],
    slots: Sequence[str],
    weights: Sequence[WeekWeight],
    current_week: int,
    *,
    limit: int,
) -> list[DropCandidate]:
    """Bench players, cheapest to lose first: the answer to "who should I drop?". This week's
    starters and the roster's regular starters are left out: nobody cuts a player he is about to
    start, or one who is only on the bench because this is his bye week."""
    if not weights:
        return []
    regular = _regular_starters(roster, slots)
    starting = {p.player_id for p in optimize(roster, slots).started}
    starting |= {p.player_id for p in regular}
    base = season_lineup_points(roster, slots, weights)
    weeks = [w.week for w in weights]
    lineups = {week: week_lineup(roster, slots, week, current_week) for week in weeks}

    candidates = []
    for player in roster:
        if player.player_id in starting:
            continue
        without = [p for p in roster if p.player_id != player.player_id]
        started_in = tuple(
            week
            for week in weeks
            if any(p.player_id == player.player_id for p in lineups[week].started)
        )
        covers = tuple(
            (week, starter)
            for week in started_in
            for starter in regular
            if starter.position == player.position
            and starter.player_id != player.player_id
            and starter.bye_week == week
        )
        candidates.append(
            DropCandidate(
                player=player,
                season_value=round(base - season_lineup_points(without, slots, weights), 4),
                weeks_started=started_in,
                covers=covers,
            )
        )
    candidates.sort(key=lambda c: (c.season_value, *_tiebreak(c.player)))
    return candidates[:limit]


def bye_weeks(
    roster: Sequence[Player], slots: Sequence[str], weeks: Sequence[int], current_week: int
) -> list[ByeWeek]:
    """Each of ``weeks`` (regular-season weeks from now on) whose optimal lineup is missing a
    regular starter to a bye, against a typical week's lineup.

    Both sides are at season rates, everyone healthy, this week included: the question is what
    the byes cost, and comparing this week's matchup projections with a season-average week
    would mix that up with whether this week's matchups happen to be good."""
    regular = _regular_starters(roster, slots)
    well = [healthy(p) for p in roster]
    typical = optimize(well, slots, _typical_projections(roster)).projected_points
    result = []
    for week in weeks:
        missing = tuple(p for p in regular if p.bye_week == week)
        if not missing:
            continue
        rates = {
            p.player_id: rate
            for p in well
            if p.bye_week != week and (rate := _rate_in(p, week)) is not None
        }
        lineup = optimize(well, slots, rates)
        result.append(
            ByeWeek(
                week=week,
                projected=lineup.projected_points,
                typical=typical,
                missing=missing,
            )
        )
    return result


def _rate_in(player: Player, week: int) -> float | None:
    """His season rate, or his published outlook for ``week`` where there is one."""
    for projection in player.outlook:
        if projection.week == week:
            return projection.points
    return weekly_rate(player)


def _typical_projections(roster: Sequence[Player]) -> dict[int, float]:
    """Everyone healthy at his season rate, nobody on bye: an ordinary week."""
    return {p.player_id: rate for p in roster if (rate := weekly_rate(p)) is not None}


def _regular_starters(roster: Sequence[Player], slots: Sequence[str]) -> list[Player]:
    """Who starts in an ordinary week."""
    return list(optimize([healthy(p) for p in roster], slots, _typical_projections(roster)).started)


def _tiebreak(player: Player) -> tuple[float, float]:
    points = projected_points(player)
    owned = player.market.percent_owned if player.market else None
    return (
        -(points if points is not None and not is_ruled_out(player) else 0.0),
        -(owned or 0.0),
    )

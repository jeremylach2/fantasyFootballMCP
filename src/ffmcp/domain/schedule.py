"""Rest-of-season value that knows about the calendar.

The simple version of season value, "this week's value times the weeks remaining", is wrong in
two ways a manager feels:

* **Byes.** A player is worth nothing in his bye week, and two starters sharing a bye leave a
  hole that neither's projection shows. Values here are summed week by week, with each player
  absent in his own bye week, through the optimal lineup: so a backup who covers a bye earns
  exactly the points he would start for, and no more.
* **The playoffs.** Weeks after the regular season count only if you are still playing. Each
  playoff week is weighted by the team's probability of making the playoffs, so a contender
  values a receiver with a juicy week-16 matchup and a long-shot values the next three weeks.

Which projection each week uses matters as much as the calendar. This week uses this week's
projection, which knows the matchup. Every later week uses the player's *season-average*
projection instead: holding this week's number flat for the rest of the year would turn one
soft matchup into a season of them (in a live league, a defense projected 6.8 against a weak
offense averages 3.7 across its schedule). Later weeks also treat the player as healthy, since
a one-week "out" should not zero his season; injuries that last are already in ESPN's season
average, which drops toward zero for a player not expected back. Where a second source publishes
a week-by-week outlook (``Player.outlook``), that week's number wins: it is the only way an
injured player's expected return reaches the value at all.

``lineup_inputs`` is the one definition of "what a given week looks like" for a roster. The
season simulation (``domain.simulate``), trade and waiver valuation all go through it, so they
cannot disagree about who is on bye or which projection a week uses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence

from ffmcp.domain.models import Frozen, LeagueSettings, Lineup, Player, Projection
from ffmcp.domain.optimizer import optimize, projected_points


class WeekWeight(Frozen):
    week: int
    weight: float
    """1.0 for a regular-season week; the playoff probability for a playoff week."""


def remaining_week_weights(
    settings: LeagueSettings, current_week: int, *, playoff_probability: float
) -> tuple[WeekWeight, ...]:
    """Every week from ``current_week`` to the championship, with its weight.

    ``playoff_probability`` is a fraction in ``[0, 1]``, from ``domain.simulate``.
    """
    probability = min(max(playoff_probability, 0.0), 1.0)
    regular = [
        WeekWeight(week=week, weight=1.0)
        for week in range(current_week, settings.reg_season_weeks + 1)
    ]
    playoffs = [
        WeekWeight(week=week, weight=probability)
        for week in settings.playoff_weeks
        if week >= current_week
    ]
    return tuple(regular + playoffs)


def weekly_rate(player: Player) -> float | None:
    """Points per week this player is worth over the rest of the season: his season-average
    projection where the provider has one, otherwise his current weekly projection."""
    if player.season_rate is not None:
        return player.season_rate
    return projected_points(player)


def rate_for_week(player: Player, week: int) -> float | None:
    """This week's projection for this week; a published outlook for a later week where there
    is one; the season rate otherwise."""
    if player.projection is not None and player.projection.week == week:
        return projected_points(player)
    for projection in player.outlook:
        if projection.week == week:
            return projection.points
    return weekly_rate(player)


def average_rate(player: Player, weeks: Sequence[int]) -> float | None:
    """His average projection over ``weeks``, healthy, byes left out: an injured player's
    expected return (via his outlook) included, which no single rate captures."""
    well = healthy(player)
    rates = [
        rate
        for week in weeks
        if well.bye_week != week and (rate := rate_for_week(well, week)) is not None
    ]
    return sum(rates) / len(rates) if rates else None


def week_projections(players: Sequence[Player], week: int) -> dict[int, float]:
    """The ``projections`` map for one week: everyone at his rate for that week, except whoever
    is on bye, who is absent from the map, which is how ``domain.optimizer`` reads "no game"."""
    result: dict[int, float] = {}
    for player in players:
        rate = rate_for_week(player, week)
        if rate is not None and player.bye_week != week:
            result[player.player_id] = rate
    return result


def healthy(player: Player) -> Player:
    """``player`` with any injury designation cleared: how a later week sees him."""
    if player.injury_status is None and not player.injured:
        return player
    return player.model_copy(update={"injury_status": None, "injured": False})


def as_of_week(player: Player, week: int, current_week: int) -> Player:
    """``player`` as a later week sees him, for showing that week: healthy, projected at his
    rate for that week (no projection on his bye), no live score, no kickoff. The current week
    returns him unchanged. The same assumptions ``lineup_inputs`` optimizes with, so a later
    week's roster view and its optimal lineup agree."""
    if week == current_week:
        return player
    well = healthy(player)
    rate = None if well.bye_week == week else rate_for_week(well, week)
    return well.model_copy(
        update={
            "projection": None
            if rate is None
            else Projection(week=week, points=rate, source="season"),
            "live_points": None,
            "kickoff": None,
        }
    )


def lineup_inputs(
    players: Sequence[Player], week: int, current_week: int
) -> tuple[list[Player], dict[int, float] | None]:
    """The roster and ``projections`` map to optimize ``week`` with.

    The current week is the roster as it stands: this week's projections, this week's injury
    designations (``None`` map). Any later week is the roster healthy, at each player's rate for
    that week, byes removed.
    """
    if week == current_week:
        return list(players), None
    well = [healthy(player) for player in players]
    return well, week_projections(well, week)


def week_lineup(
    players: Sequence[Player], slots: Sequence[str], week: int, current_week: int
) -> Lineup:
    """The optimal lineup for ``week`` as ``lineup_inputs`` sees it."""
    roster, projections = lineup_inputs(players, week, current_week)
    return optimize(roster, slots, projections)


def _week_key(projections: Mapping[int, float] | None) -> Hashable:
    """Weeks whose inputs are identical have identical optimal lineups. ``None`` is the current
    week, which uses the roster's own attached projections."""
    return None if projections is None else frozenset(projections.items())


def season_lineup_points(
    players: Sequence[Player], slots: Sequence[str], weights: Sequence[WeekWeight]
) -> float:
    """Weighted optimal-lineup points over ``weights``' weeks.

    The first week in ``weights`` is the current one. Later weeks whose inputs are identical
    (the same players on bye, no outlook that week) have identical optimal lineups, so each
    distinct week is optimized once: typically four or five optimizations for a whole season
    rather than one per week.
    """
    if not weights:
        return 0.0
    current = min(weight.week for weight in weights)
    groups: dict[Hashable, float] = defaultdict(float)
    inputs: dict[Hashable, tuple[list[Player], dict[int, float] | None]] = {}
    for weight in weights:
        roster, projections = lineup_inputs(players, weight.week, current)
        key = _week_key(projections)
        groups[key] += weight.weight
        inputs.setdefault(key, (roster, projections))
    total = 0.0
    for key, group_weight in groups.items():
        roster, projections = inputs[key]
        total += group_weight * optimize(roster, slots, projections).projected_points
    return total


def season_positional_points(
    players: Sequence[Player], slots: Sequence[str], weights: Sequence[WeekWeight]
) -> dict[str, float]:
    """Weighted *started* points per position over ``weights``' weeks: what the lineup gets
    from each position across the rest of the season, byes included."""
    if not weights:
        return {}
    current = min(weight.week for weight in weights)
    totals: dict[str, float] = defaultdict(float)
    for weight in weights:
        roster, projections = lineup_inputs(players, weight.week, current)
        for player in optimize(roster, slots, projections).started:
            points = projected_points(player, projections) or 0.0
            totals[player.position] += weight.weight * points
    return dict(totals)


def season_marginal_value(
    roster: Sequence[Player],
    player: Player,
    slots: Sequence[str],
    weights: Sequence[WeekWeight],
) -> float:
    """What adding ``player`` to ``roster`` is worth over the rest of the season, in weighted
    lineup points: ``domain.trades.marginal_value`` with a calendar."""
    without = [other for other in roster if other.player_id != player.player_id]
    return season_lineup_points([*without, player], slots, weights) - season_lineup_points(
        without, slots, weights
    )


def effective_weeks(player: Player, weights: Sequence[WeekWeight]) -> float:
    """Weighted weeks in which ``player`` has a game: the multiplier for a per-week value."""
    return sum(weight.weight for weight in weights if weight.week != player.bye_week)

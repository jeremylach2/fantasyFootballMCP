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
average, which drops toward zero for a player not expected back.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ffmcp.domain.models import Frozen, LeagueSettings, Player
from ffmcp.domain.optimizer import optimize


class WeekWeight(Frozen):
    week: int
    weight: float
    """1.0 for a regular-season week; the playoff probability for a playoff week."""


def remaining_week_weights(
    settings: LeagueSettings, current_week: int, *, playoff_probability: float
) -> tuple[WeekWeight, ...]:
    """Every week from ``current_week`` to the championship, with its weight.

    ``playoff_probability`` is a fraction in ``[0, 1]``: simulated, or ESPN's own estimate.
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
    return player.projection.points if player.projection is not None else None


def rate_for_week(player: Player, week: int) -> float | None:
    """This week's projection for this week; the season rate for any other."""
    if player.projection is not None and player.projection.week == week:
        return player.projection.points
    return weekly_rate(player)


def week_projections(players: Sequence[Player], week: int) -> dict[int, float]:
    """The ``projections`` map for one week: everyone at his rate for that week, except whoever
    is on bye, who is absent from the map, which is how ``domain.optimizer`` reads "no game"."""
    result: dict[int, float] = {}
    for player in players:
        rate = rate_for_week(player, week)
        if rate is not None and player.bye_week != week:
            result[player.player_id] = rate
    return result


def _healthy(player: Player) -> Player:
    if player.injury_status is None and not player.injured:
        return player
    return player.model_copy(update={"injury_status": None, "injured": False})


def season_lineup_points(
    players: Sequence[Player], slots: Sequence[str], weights: Sequence[WeekWeight]
) -> float:
    """Weighted optimal-lineup points over ``weights``' weeks.

    The first week in ``weights`` is the current one: it uses this week's projections and
    injury designations. Later weeks in which the same set of these players is on bye have
    identical optimal lineups, so they are grouped and each distinct week is optimized once:
    typically four or five optimizations for a whole season rather than one per week.
    """
    if not weights:
        return 0.0
    current = min(weight.week for weight in weights)
    healthy = [_healthy(player) for player in players]
    groups: dict[tuple[bool, frozenset[int]], float] = defaultdict(float)
    week_of_group: dict[tuple[bool, frozenset[int]], int] = {}
    for weight in weights:
        on_bye = frozenset(p.player_id for p in players if p.bye_week == weight.week)
        key = (weight.week == current, on_bye)
        groups[key] += weight.weight
        week_of_group.setdefault(key, weight.week)
    total = 0.0
    for key, group_weight in groups.items():
        week = week_of_group[key]
        roster = players if key[0] else healthy
        lineup = optimize(roster, slots, week_projections(roster, week))
        total += group_weight * lineup.projected_points
    return total


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

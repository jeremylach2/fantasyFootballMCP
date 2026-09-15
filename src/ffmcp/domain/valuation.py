"""Player valuation: value over replacement, plus a small market nudge.

A projection alone says nothing about what a player is *worth*. Twelve points from a tight end
in a one-TE league is a luxury; twelve from a running back in a league that starts two of them
and a flex is barely a starter. Value over replacement fixes the comparison by subtracting the
points freely available at that position: the best player nobody would have to start.

The replacement level is derived from the league's own roster settings rather than a hardcoded
table, because that is the whole point: a 14-team league that starts three receivers has a very
different replacement receiver from an 8-team league that starts two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ffmcp.domain.models import (
    NON_STARTING_SLOTS,
    LeagueSettings,
    LeagueState,
    MarketSignal,
    Player,
)
from ffmcp.domain.optimizer import SLOT_POSITIONS, projected_points

MARKET_ADJUSTMENT_CAP = 0.05
"""Hard ceiling on how far market sentiment may move a valuation: 5%.

The cap is the design. Sleeper's trending-adds count measures how many managers clicked "add"
in the last day, which is a real signal about competition for a player and a poor one about how
many points he will score. Uncapped, a popular name would outrank a better player on hype
alone, which is precisely the failure this module exists to avoid. Capped at 5% of a player's
value, it can reorder two players already within a few percent of each other, which is where a
tiebreaker belongs, and can never overturn a real difference in projection."""

_TRENDING_SATURATION = 50_000.0
"""Adds in the lookback window at which the nudge reaches ~76% of its cap. Calibrated against
Sleeper's 24-hour counts, where a genuinely hot waiver pickup runs into the tens of thousands.
An assumption, like everything else about the market term."""


def starters_per_team(position: str, settings: LeagueSettings) -> float:
    """How many starters at ``position`` a single team fields, counting flex slots fractionally.

    A ``RB/WR/TE`` slot is credited a third to each of the three positions. Real flexes skew to
    running backs and receivers, so this slightly overstates tight-end demand. It is an even
    split because there is no data here to justify any other number, and saying so is better
    than inventing weights.
    """
    total = 0.0
    for slot, count in settings.slot_counts.items():
        if slot in NON_STARTING_SLOTS or count <= 0:
            continue
        positions = SLOT_POSITIONS.get(slot)
        if positions is not None and position in positions:
            total += count / len(positions)
    return total


def replacement_level(
    position: str,
    league_settings: LeagueSettings,
    player_pool: Sequence[Player],
    projections: Mapping[int, float] | None = None,
) -> float:
    """Points available for free at ``position``: the best player who would not be starting.

    Rank everyone at the position and step past the number the league starts league-wide. With
    twelve teams starting one quarterback each, the twelfth-best is the last starter and the
    thirteenth is the replacement. ``player_pool`` should be the whole relevant universe
    (rostered players plus free agents); a pool too shallow to reach the cutoff falls back to
    its own worst player rather than pretending replacement level is zero.
    """
    ranked = sorted(
        (
            points
            for player in player_pool
            if player.position == position
            for points in (projected_points(player, projections),)
            if points is not None
        ),
        reverse=True,
    )
    if not ranked:
        return 0.0
    cutoff = round(league_settings.team_count * starters_per_team(position, league_settings))
    return ranked[min(cutoff, len(ranked) - 1)]


def replacement_levels(
    league_settings: LeagueSettings,
    player_pool: Sequence[Player],
    projections: Mapping[int, float] | None = None,
) -> dict[str, float]:
    """Every position's replacement level in one pass, for callers working in a loop."""
    return {
        position: replacement_level(position, league_settings, player_pool, projections)
        for position in {player.position for player in player_pool}
    }


def market_adjustment(player: Player, market_signal: MarketSignal | None = None) -> float:
    """A multiplier in ``[1, 1 + MARKET_ADJUSTMENT_CAP]`` reflecting waiver-wire enthusiasm.

    Saturating (``tanh``) rather than linear, so the difference between a quiet player and a
    trending one matters while the difference between trending and viral does not. Falls back
    to the signal attached to the player, and to a flat 1.0 when there is no signal at all: an
    unknown player is not a penalised one.
    """
    signal = market_signal if market_signal is not None else player.market
    if signal is None or signal.trending_adds is None:
        return 1.0
    adds = max(0, signal.trending_adds)
    return 1.0 + MARKET_ADJUSTMENT_CAP * math.tanh(adds / _TRENDING_SATURATION)


def value_over_replacement(
    player: Player,
    league_settings: LeagueSettings,
    player_pool: Sequence[Player],
    *,
    weeks_remaining: int = 1,
    projections: Mapping[int, float] | None = None,
    market_signal: MarketSignal | None = None,
    replacement: float | None = None,
) -> float:
    """Points this player is worth above replacement over the remaining schedule.

    Pass ``replacement`` (from ``replacement_levels``) when valuing many players at once; it is
    the only expensive part.

    The market nudge multiplies the surplus rather than the projection, which is what keeps the
    cap meaning what it says: at most 5% of *value*, not 5% of a projection that may dwarf the
    surplus it sits on. It applies only to players who are above replacement, because a burst of
    waiver adds is evidence of interest and never evidence of decline.
    """
    points = projected_points(player, projections)
    if points is None:
        return 0.0
    level = (
        replacement
        if replacement is not None
        else replacement_level(player.position, league_settings, player_pool, projections)
    )
    surplus = points - level
    nudge = market_adjustment(player, market_signal) if surplus > 0.0 else 1.0
    return surplus * nudge * weeks_remaining


def remaining_strength_of_schedule(team_id: int, state: LeagueState) -> float | None:
    """Average win percentage of ``team_id``'s remaining opponents, as a sanity check on the
    schedule rather than a points prediction.

    ``None`` when no remaining matchup names an opponent for this team. An empty schedule is a
    fact worth showing as absent, not as a manufactured 0%.
    """
    percentages: list[float] = []
    for matchup in state.matchups:
        if not state.current_week <= matchup.week <= state.settings.reg_season_weeks:
            continue
        if matchup.home_team_id == team_id:
            opponent_id = matchup.away_team_id
        elif matchup.away_team_id == team_id:
            opponent_id = matchup.home_team_id
        else:
            continue
        if opponent_id is None:
            continue
        opponent = state.team(opponent_id)
        games = opponent.wins + opponent.losses + opponent.ties
        if games > 0:
            percentages.append(100.0 * (opponent.wins + 0.5 * opponent.ties) / games)
    if not percentages:
        return None
    return sum(percentages) / len(percentages)

"""How uncertain is a projection? Weekly scoring spread, measured rather than assumed.

A projection is the middle of a distribution, and nearly every decision this server makes
depends on how wide that distribution is: the win probability of a matchup, whether a
boom-or-bust receiver beats a steady one, how much a trade moves playoff odds. This module owns
that width.

Every constant below was fit by ``scripts/calibrate.py`` against a real 16-team PPR league's
2025 season (3,073 player-weeks with an ESPN projection and a game played). They replaced
hand-picked values that understated player spread and then papered over it with a 1.6x
"correlation" factor. That model put a typical lineup's weekly sd near 33 points; the measured
figure is 22.4, so it was pushing every matchup's win probability toward a coin flip.

Two layers, each kept because it improved held-out log-likelihood in that script:

1. **Position** sets the spread: sd as a fraction of the projection.
2. **Source disagreement** widens it. When ESPN and a second projection source disagree about a
   player, both are more likely to be wrong: the most-disputed 5% of projections missed by 2.5x
   as much as the least-disputed half.

Two plausible layers were tested and left out, because the data did not support them:
correcting a player's *mean* by his own recent misses made the next weeks' predictions worse at
every shrinkage strength (last month's miss is noise, not a trait), and personalising his
*spread* from his own history was no better than his position's spread at any shrinkage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ffmcp.domain.models import Player

POSITION_SD_FRACTION: Mapping[str, float] = {
    "QB": 0.47,
    "RB": 0.59,
    "WR": 0.59,
    "TE": 0.67,
    "K": 0.58,
    "D/ST": 1.14,
}
"""Weekly residual sd (actual minus ESPN projection) as a fraction of the projection, by
position. Quarterbacks remain the steadiest and defenses by far the least predictable; a
defense's sd exceeding its projection is not a typo, it is what pick-sixes do."""

DEFAULT_SD_FRACTION = 0.60
MIN_PLAYER_SD = 2.0
"""Floor for any player's sd. A kicker projected for 3 points can still score 12."""

TEAM_SD_FACTOR = 0.92
"""Measured ratio of a real lineup's weekly sd (22.4) to the sum-in-quadrature of its players'
sds (24.5). Slightly *below* one: some starters are negatively correlated (your defense does
well exactly when the opposing offense does not), and that offsets the positive stacks. The old
model's 1.6 was not modelling correlation at all, it was compensating for per-player numbers
that were too small."""

DISAGREEMENT_SENSITIVITY = 1.0
"""How strongly source disagreement widens the sd. Fit by the same script (tried 0 to 2)."""

TYPICAL_DISAGREEMENT = 0.08
"""Median relative disagreement between ESPN and Sleeper in 2025. The multiplier is normalised
around it, so a typical player keeps his position's sd and only unusual disputes move it."""

_MAX_DISAGREEMENT = 1.0

_Z_P10 = 1.2815515655446004
"""Standard-normal quantile for the 10th/90th percentiles used as floor and ceiling."""


def disagreement(primary: float | None, secondary: float | None) -> float | None:
    """Relative gap between two projections of the same player, or ``None`` if either is
    missing. Relative to the primary, floored at one point so a projection of 0.3 does not turn
    a trivial difference into an enormous ratio."""
    if primary is None or secondary is None:
        return None
    return abs(primary - secondary) / max(primary, 1.0)


def disagreement_multiplier(relative_gap: float | None) -> float:
    """Scale factor on a player's sd, 1.0 for a typical (or unknown) level of disagreement."""
    if relative_gap is None:
        return 1.0
    gap = min(relative_gap, _MAX_DISAGREEMENT)
    return (1.0 + DISAGREEMENT_SENSITIVITY * gap) / (
        1.0 + DISAGREEMENT_SENSITIVITY * TYPICAL_DISAGREEMENT
    )


def position_fraction(position: str) -> float:
    return POSITION_SD_FRACTION.get(position, DEFAULT_SD_FRACTION)


def player_sd(player: Player, points: float, *, relative_gap: float | None = None) -> float:
    """One player's weekly sd given his projection and the (optional) disagreement between
    projection sources."""
    spread = position_fraction(player.position) * points * disagreement_multiplier(relative_gap)
    return max(spread, MIN_PLAYER_SD)


def player_sds(
    players: Sequence[Player],
    *,
    projections: Mapping[int, float] | None = None,
    alternative: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """``{player_id: sd}`` for every player with a projection this week: the map
    ``domain.simulate`` and ``domain.risk`` consume. ``projections`` overrides each player's
    attached projection, as everywhere else in ``domain/``."""
    result: dict[int, float] = {}
    for player in players:
        if projections is not None:
            points = projections.get(player.player_id)
        else:
            points = player.projection.points if player.projection is not None else None
        if points is None:
            continue
        gap = disagreement(points, alternative.get(player.player_id)) if alternative else None
        result[player.player_id] = player_sd(player, points, relative_gap=gap)
    return result


def floor_ceiling(points: float, sd: float) -> tuple[float, float]:
    """10th and 90th percentile outcomes, clipped at zero. Normal is an approximation (real
    scoring is right-skewed), so these are ranges to compare players by, not promises."""
    return max(0.0, points - _Z_P10 * sd), points + _Z_P10 * sd

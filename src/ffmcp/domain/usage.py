"""Opportunity, expected points, and luck: who is due to regress, in either direction.

Fantasy points are opportunity times efficiency times touchdown luck, and only the first of
those is stable week to week. A receiver seeing ten targets a game scores eventually; one
scoring on three catches a game stops eventually. So this module prices each player-game by its
*opportunity* alone ("expected fantasy points", xFP) and treats the gap to what he actually
scored as luck.

That is a testable claim, and it was tested (``scripts/calibrate.py``): fit on 2024, scored on
2025, xFP per game through week 4 predicted the next six weeks' scoring with a mean absolute
error of 3.44 points against 3.69 for actual points per game. Players more than 3 points per
game *below* their xFP after week 4 went on to score 3.5 more per game than they had; players
more than 3 *above* it gave back 2.9. That regression is the whole basis for the buy-low and
sell-high labels below, and the threshold is where it was measured.

The model is deliberately plain: a per-position linear regression of points on volume (pass
attempts and air yards, carries, targets and air yards), no touchdowns, no efficiency. Anything
that measured efficiency would re-import the luck it exists to strip out.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from ffmcp.domain.models import Frozen, UsageWeek

_FEATURES: Mapping[str, tuple[str, ...]] = {
    "QB": ("attempts", "passing_air_yards", "carries"),
    "RB": ("carries", "targets", "receiving_air_yards"),
    "WR": ("targets", "receiving_air_yards", "carries"),
    "TE": ("targets", "receiving_air_yards"),
}

_COEFFICIENTS: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # position: (standard-scoring coefficients, full-PPR coefficients), features then intercept
    "QB": ((0.2945, 0.0124, 1.0409, -0.3525), (0.2945, 0.0124, 1.0416, -0.3493)),
    "RB": ((0.7188, 0.5569, 0.0468, -0.3023), (0.7198, 1.3476, 0.0331, -0.3173)),
    "WR": ((0.7856, 0.0293, 0.8814, -0.0102), (1.5783, 0.0145, 0.9390, -0.0324)),
    "TE": ((0.8160, 0.0332, 0.1149), (1.6462, 0.0178, 0.0830)),
}
"""Least-squares fits on every 2024-2025 regular-season player-game in nflverse (R^2: QB 0.49,
RB 0.70, WR 0.66, TE 0.67 under PPR). Regenerate with ``scripts/calibrate.py``. A least-squares
fit is linear in its target, so a half-PPR league's coefficients are exactly the average of these
two columns, and any reception scoring interpolates between them."""

LUCK_THRESHOLD = 3.0
"""Points per game between actual and expected scoring before a player is labelled buy-low or
sell-high. The size of the gap at which the out-of-sample regression above was measured."""

MIN_GAMES = 3
"""Games before a player can be labelled: the backtest measured the regression from week 4 on,
and two games of touchdown luck is not yet a pattern."""

MIN_WATCH_GAMES = 2
"""Games before an unlabelled gap is worth watching at all. One game is one game."""
MIN_RELEVANT_PPG = 7.0
"""Points per game below which a player is not worth a trade conversation. Measured on the side
that makes him interesting: *expected* points for a buy-low (a backup's bad luck on four touches
a game is still four touches), *actual* points for a sell-high (the four-target receiver scoring
19 a game on touchdowns is exactly the player to sell)."""


def expected_points(week: UsageWeek, reception_points: float = 1.0) -> float | None:
    """xFP for one player-game, or ``None`` for a position the model does not cover (K, D/ST)."""
    features = _FEATURES.get(week.position)
    if features is None:
        return None
    standard, ppr = _COEFFICIENTS[week.position]
    coefficients = [s + reception_points * (p - s) for s, p in zip(standard, ppr, strict=True)]
    values = [float(getattr(week, name)) for name in features]
    return sum(c * v for c, v in zip(coefficients, [*values, 1.0], strict=True))


class UsageProfile(Frozen):
    """One player's season-to-date workload, and what it says about his scoring."""

    player_id: int
    name: str
    position: str
    pro_team: str
    games: int
    snap_pct: float | None
    """Mean offensive snap share across his games, 0-100."""
    recent_snap_pct: float | None
    """Snap share in his most recent game, 0-100. A role change shows here first."""
    targets_per_game: float
    carries_per_game: float
    target_share: float | None
    """Mean share of his team's targets, 0-100."""
    expected_ppg: float
    actual_ppg: float

    @property
    def luck_ppg(self) -> float:
        """Actual minus expected points per game. Positive is running hot."""
        return self.actual_ppg - self.expected_ppg

    @property
    def signal(self) -> str | None:
        """``"buy_low"``, ``"sell_high"`` or ``None``: see the module docstring for the evidence
        behind the threshold."""
        if self.games < MIN_GAMES:
            return None
        return self.gap_direction

    @property
    def gap_direction(self) -> str | None:
        """Which way this player's scoring has strayed from his workload, ignoring sample size:
        the ``signal`` before it has enough games to be a call."""
        if self.games < MIN_WATCH_GAMES:
            return None
        if self.luck_ppg <= -LUCK_THRESHOLD and self.expected_ppg >= MIN_RELEVANT_PPG:
            return "buy_low"
        if self.luck_ppg >= LUCK_THRESHOLD and self.actual_ppg >= MIN_RELEVANT_PPG:
            return "sell_high"
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def usage_profiles(
    weeks: Sequence[UsageWeek], reception_points: float = 1.0
) -> dict[int, UsageProfile]:
    """``{player_id: profile}`` for every modelled position, from all of ``weeks``."""
    by_player: dict[int, list[UsageWeek]] = defaultdict(list)
    for week in weeks:
        if week.position in _FEATURES:
            by_player[week.player_id].append(week)

    profiles: dict[int, UsageProfile] = {}
    for player_id, games in by_player.items():
        games.sort(key=lambda g: (g.season, g.week))
        latest = games[-1]
        n = len(games)
        expected = [expected_points(g, reception_points) or 0.0 for g in games]
        snap = _mean(g.snap_pct for g in games)
        share = _mean(g.target_share for g in games)
        profiles[player_id] = UsageProfile(
            player_id=player_id,
            name=latest.name,
            position=latest.position,
            pro_team=latest.pro_team,
            games=n,
            snap_pct=None if snap is None else 100.0 * snap,
            recent_snap_pct=None if latest.snap_pct is None else 100.0 * latest.snap_pct,
            targets_per_game=sum(g.targets for g in games) / n,
            carries_per_game=sum(g.carries for g in games) / n,
            target_share=None if share is None else 100.0 * share,
            expected_ppg=sum(expected) / n,
            actual_ppg=sum(g.fantasy_points for g in games) / n,
        )
    return profiles


def backup_running_back(
    starter_id: int, pro_team: str, profiles: Mapping[int, UsageProfile]
) -> UsageProfile | None:
    """The running back most likely to inherit ``starter_id``'s work if he goes down: the one
    on the same NFL team, other than the starter, playing the most snaps. Snap share is the
    right measure because it is who the coaches trust on the field today, which is what decides
    the next man up; a depth chart is often a month stale.

    ``None`` unless ``starter_id`` is himself his team's lead back. A committee's second back
    has no handcuff: the player who "inherits his work" is the team's actual starter.
    """
    backs = sorted(
        (
            profile
            for profile in profiles.values()
            if profile.position == "RB"
            and profile.pro_team == pro_team
            and profile.snap_pct is not None
        ),
        key=lambda p: (-(p.recent_snap_pct or 0.0), -(p.snap_pct or 0.0), p.player_id),
    )
    if len(backs) < 2 or backs[0].player_id != starter_id:
        return None
    return backs[1]

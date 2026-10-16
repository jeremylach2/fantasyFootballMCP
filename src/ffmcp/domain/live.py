"""Live win probability: the score so far, plus what is still to be played.

Once games kick off, the pregame win probability answers a question nobody is asking any more.
What matters on a Sunday afternoon is the score now and how much of each lineup is still to
play. Each starter contributes his points so far plus his projection for the part of his game
that is left, and his spread shrinks with the time remaining (variance scales with the share of
the game still to come, as a sum of independent plays does).

How much of a game is left is estimated from the clock: ESPN's box score reports a game as
either not played or finished, nothing in between, so the share remaining is the time since
kickoff against ``GAME_LENGTH``. That is an approximation, stated as one: halftime, overtime and
a slow fourth quarter all bend it a little, and it is exact at kickoff and at the final whistle,
which is where it matters most.

``at`` is passed in: this module, like the rest of ``domain/``, never reads a clock.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from ffmcp.domain.models import Player
from ffmcp.domain.optimizer import projected_points
from ffmcp.domain.simulate import PlayerSds, ScoreModel, player_score_sd
from ffmcp.domain.variance import TEAM_SD_FACTOR

GAME_LENGTH = timedelta(minutes=190)
"""Kickoff to final whistle for a typical NFL game, a little over three hours."""


def share_left(player: Player, at: datetime) -> float:
    """The share of ``player``'s game still to play at ``at``: 1 before kickoff, 0 once it is
    over, and 0 for a player with no game this week."""
    if player.kickoff is None or projected_points(player) is None:
        return 0.0
    if at <= player.kickoff:
        return 1.0
    return max(0.0, 1.0 - (at - player.kickoff) / GAME_LENGTH)


def live_score_model(
    team_id: int, starters: Sequence[Player], at: datetime, sds: PlayerSds | None = None
) -> tuple[ScoreModel, float]:
    """The team's final-score distribution given what has happened by ``at``, and how many of
    its projected points are still to play."""
    mean = variance = remaining = 0.0
    for player in starters:
        left = share_left(player, at)
        projected = projected_points(player) or 0.0
        mean += (player.live_points or 0.0) + projected * left
        remaining += projected * left
        variance += player_score_sd(player, projected, sds) ** 2 * left
    sd = math.sqrt(variance) * TEAM_SD_FACTOR
    return ScoreModel(team_id=team_id, mean=mean, sd=sd), remaining


def any_kicked_off(players: Sequence[Player], at: datetime) -> bool:
    return any(p.kickoff is not None and p.kickoff <= at for p in players)

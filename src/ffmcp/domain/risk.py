"""Start the lineup most likely to *win*, which is not always the one projected to score most.

Head-to-head fantasy pays for beating one opponent, not for points. When you are projected to
lose by fifteen, the steady receiver who reliably scores his projection reliably loses you the
week, and a boom-or-bust one with a slightly lower projection is the better start: you need the
boom. Projected to win by fifteen, the logic flips and you want the floor.

Formally, with your score ~ N(mu, sigma) and theirs ~ N(mu_o, sigma_o),

    P(win) = Phi((mu - mu_o) / sqrt(sigma^2 + sigma_o^2))

An underdog (mu < mu_o) raises P(win) by raising sigma; a favourite raises it by lowering sigma.
Maximising it exactly is a hard combinatorial problem, because sigma is not additive over
players. So this module leans on the exact optimizer instead: it re-runs the matching with each
player scored ``mean + tilt * sd`` for a handful of tilts, positive (chase variance) and negative
(protect a lead), and keeps whichever resulting lineup has the highest true win probability.
Tilt 0 is the plain points-maximising lineup, so the answer can never be worse than it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ffmcp.domain.models import Frozen, Lineup, Player
from ffmcp.domain.optimizer import optimize, projected_points
from ffmcp.domain.simulate import PlayerSds, ScoreModel, lineup_score_model, matchup_win_probability

TILTS: tuple[float, ...] = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
"""Standard deviations of risk to trade per point of mean. Beyond one sd either way the lineups
stop changing in practice: every player is already sorted by ceiling (or floor)."""

MIN_WIN_PROB_GAIN = 0.5
"""Percentage points a tilted lineup must add before it replaces the points-maximising one.
The win probability is a model output, not a measurement; recommending a lower-projected start
for a tenth of a point of it would be advice built on the model's rounding."""

_MIN_TILTED_WEIGHT = 0.01
"""A negative tilt can push a high-variance player's score below zero, and the optimizer would
rather leave a slot empty than fill it at negative value. Clamp so every slot still fills."""


class LineupChoice(Frozen):
    """The recommended lineup and the points-maximising one, each with its win probability."""

    lineup: Lineup
    win_probability: float
    points_lineup: Lineup
    points_win_probability: float
    tilt: float
    """0 when the recommendation is the points-maximising lineup; positive when it chases
    variance (underdog), negative when it protects a lead (favourite)."""

    @property
    def differs(self) -> bool:
        return self.tilt != 0.0


def best_lineup_for_matchup(
    team_id: int,
    players: Sequence[Player],
    slots: Sequence[str],
    opponent: ScoreModel,
    *,
    sds: PlayerSds,
    projections: Mapping[int, float] | None = None,
) -> LineupChoice:
    """The lineup, from ``players``, that maximises the chance of outscoring ``opponent``."""
    means = {
        player.player_id: points
        for player in players
        if (points := projected_points(player, projections)) is not None
    }

    def evaluate(tilt: float) -> tuple[Lineup, float]:
        weights = {
            player_id: max(mean + tilt * sds.get(player_id, 0.0), _MIN_TILTED_WEIGHT)
            for player_id, mean in means.items()
        }
        tilted = optimize(players, slots, weights)
        true_points = sum(means[player.player_id] for player in tilted.started)
        lineup = tilted.model_copy(update={"projected_points": round(true_points, 4)})
        model = lineup_score_model(team_id, lineup, means, sds)
        return lineup, matchup_win_probability(model, opponent)

    points_lineup, points_probability = evaluate(0.0)
    best_lineup, best_probability, best_tilt = points_lineup, points_probability, 0.0
    for tilt in TILTS:
        if tilt == 0.0:
            continue
        lineup, probability = evaluate(tilt)
        if probability > best_probability + 1e-9:
            best_lineup, best_probability, best_tilt = lineup, probability, tilt

    if best_probability - points_probability < MIN_WIN_PROB_GAIN:
        best_lineup, best_probability, best_tilt = points_lineup, points_probability, 0.0
    return LineupChoice(
        lineup=best_lineup,
        win_probability=best_probability,
        points_lineup=points_lineup,
        points_win_probability=points_probability,
        tilt=best_tilt,
    )

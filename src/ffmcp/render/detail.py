"""Small decision-shaped results: Pydantic models, returned directly rather than through
``render/tables.py``. A ``LineupAdvice`` or ``TradeVerdict`` is few enough fields that the
schema earns its keep for a client. Conversion only: the values here were computed in
``domain/``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ffmcp.domain.models import Lineup, Player
from ffmcp.domain.models import Swap as DomainSwap
from ffmcp.domain.optimizer import is_ruled_out
from ffmcp.domain.trades import TradeEvaluation
from ffmcp.render.tables import short_name, slot_label

MAX_SWAPS = 5
"""Cap ``swaps`` at 5, ordered by gain descending."""


class Swap(BaseModel):
    bench: str
    starter: str
    slot: str
    gain: float
    reason: str = Field(max_length=80)


class ActualLineup(BaseModel):
    """What the roster actually scored this week against what actually happened, rather than
    Thursday's projections. Present only once the week's games have started. See
    ``build_lineup_advice``."""

    points_scored: float
    optimal_live_total: float
    points_left_on_bench: float
    swaps: list[Swap] = Field(default_factory=list)


class LineupAdvice(BaseModel):
    week: int
    current_projected: float
    optimal_projected: float
    point_gain: float
    win_prob_delta: float
    swaps: list[Swap] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    actual: ActualLineup | None = None
    """Set only when at least one roster player's game has kicked off this week. Distinct from
    ``swaps`` above, which are always pregame: the two are never conflated in one section."""


def _swap_reason(swap: DomainSwap) -> str:
    if swap.player_out is not None and is_ruled_out(swap.player_out):
        return f"{swap.player_out.name} is {swap.player_out.injury_status or 'ruled out'}"
    if swap.player_in is not None:
        return f"+{swap.gain:.1f} projected points"
    return "no legal replacement available"


def _to_wire_swap(swap: DomainSwap) -> Swap:
    return Swap(
        bench=swap.player_out.name if swap.player_out is not None else "(none)",
        starter=swap.player_in.name if swap.player_in is not None else "(none)",
        slot=slot_label(swap.slot),
        gain=round(swap.gain, 1),
        reason=_swap_reason(swap)[:80],
    )


def _caveat(player: Player) -> str | None:
    if player.injury_status is None:
        return None
    label = player.injury_status.replace("_", " ").title()
    return f"{label}: {player.name} — check inactives"


def build_lineup_advice(
    *,
    week: int,
    current: Lineup,
    optimal: Lineup,
    win_prob_delta: float,
    swaps: Sequence[DomainSwap],
    actual_current: Lineup | None = None,
    actual_optimal: Lineup | None = None,
    actual_swaps: Sequence[DomainSwap] = (),
) -> LineupAdvice:
    """The common case, an already-optimal lineup, is cheap: ``swaps`` and ``caveats`` are
    both empty and the whole result is four numbers.

    ``actual_current``/``actual_optimal`` are ``None`` unless the caller found at least one
    live score for the week. See ``tools_lineup.py::_live_projections``.
    """
    capped = sorted(swaps, key=lambda swap: -swap.gain)[:MAX_SWAPS]
    caveats = [c for player in optimal.started if (c := _caveat(player)) is not None]
    actual = None
    if actual_current is not None and actual_optimal is not None:
        capped_actual = sorted(actual_swaps, key=lambda swap: -swap.gain)[:MAX_SWAPS]
        actual = ActualLineup(
            points_scored=round(actual_current.projected_points, 1),
            optimal_live_total=round(actual_optimal.projected_points, 1),
            points_left_on_bench=round(
                actual_optimal.projected_points - actual_current.projected_points, 1
            ),
            swaps=[_to_wire_swap(swap) for swap in capped_actual],
        )
    return LineupAdvice(
        week=week,
        current_projected=round(current.projected_points, 1),
        optimal_projected=round(optimal.projected_points, 1),
        point_gain=round(optimal.projected_points - current.projected_points, 1),
        win_prob_delta=round(win_prob_delta, 1),
        swaps=[_to_wire_swap(swap) for swap in capped],
        caveats=caveats,
        actual=actual,
    )


Verdict = Literal["accept", "lean_accept", "neutral", "lean_decline", "decline"]


class TradeVerdict(BaseModel):
    verdict: Verdict
    my_value_delta: float
    partner_value_delta: float
    my_playoff_odds_delta: float
    positional_impact: str
    risks: list[str] = Field(default_factory=list)


_VERDICT_BANDS: tuple[tuple[float, Verdict], ...] = (
    (3.0, "accept"),
    (0.5, "lean_accept"),
    (-0.5, "neutral"),
    (-3.0, "lean_decline"),
)
"""Playoff-odds-delta cutoffs, in percentage points. An assumption, stated as one: there is no
league data to calibrate "how many points of playoff odds count as a clear accept," so these
are round numbers chosen to separate a real edge from noise, not a fitted threshold."""


def classify_verdict(my_playoff_odds_delta: float) -> Verdict:
    for threshold, verdict in _VERDICT_BANDS:
        if my_playoff_odds_delta >= threshold:
            return verdict
    return "decline"


def render_positional_impact(deltas: Mapping[str, float]) -> str:
    """The one sentence a manager wants about shape: which position this trade actually moves.

    ``deltas`` comes from ``domain.trades.positional_impact``: the change in *started* points,
    not in the projections of the players swapped.
    """
    if not deltas:
        return "No change to the starting lineup."
    position, delta = max(deltas.items(), key=lambda item: (abs(item[1]), item[0]))
    direction = "Strengthens" if delta >= 0 else "Weakens"
    return f"{direction} {position} by {abs(delta):.1f} started pts/wk."


def _trade_risks(
    give: Sequence[Player], get: Sequence[Player], drops: Sequence[Player]
) -> list[str]:
    risks = [
        f"{player.name} is {player.injury_status.replace('_', ' ').title()}"
        for player in (*give, *get)
        if player.injury_status is not None
    ]
    if drops:
        names = ", ".join(short_name(player.name) for player in drops)
        risks.append(f"Must drop {names} to stay at roster size")
    return risks[:3]


def build_trade_verdict(
    evaluation: TradeEvaluation,
    drops: Sequence[Player],
    positional_deltas: Mapping[str, float],
) -> TradeVerdict:
    odds = evaluation.my_playoff_odds_delta or 0.0
    return TradeVerdict(
        verdict=classify_verdict(odds),
        my_value_delta=round(evaluation.my_value_delta, 1),
        partner_value_delta=round(evaluation.partner_value_delta, 1),
        my_playoff_odds_delta=round(odds, 1),
        positional_impact=render_positional_impact(positional_deltas),
        risks=_trade_risks(evaluation.offer.give, evaluation.offer.get, drops),
    )

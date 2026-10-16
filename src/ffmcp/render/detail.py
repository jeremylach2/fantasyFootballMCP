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
from ffmcp.domain.trades import MARKET_RATIO_LIMIT, TradeEvaluation
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
    playoff_odds_delta: float
    """Change in *playoff* odds, in percentage points, from making the swaps."""
    win_probability: float | None = None
    """Chance, in percent, the recommended lineup beats this week's opponent."""
    strategy: str | None = None
    """Set only when the recommendation deliberately trades projected points for a better
    chance of winning: one sentence saying why, and what it costs."""
    swaps: list[Swap] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    actual: ActualLineup | None = None
    """Set only when at least one roster player's game has kicked off this week. Distinct from
    ``swaps`` above, which are always pregame: the two are never conflated in one section."""


def _swap_reason(swap: DomainSwap, tilt: float = 0.0, *, actual: bool = False) -> str:
    if actual:
        return f"{swap.gain:+.1f} points scored"
    if swap.player_out is not None and is_ruled_out(swap.player_out):
        return f"{swap.player_out.name} is {swap.player_out.injury_status or 'ruled out'}"
    if swap.player_in is not None and swap.gain < 0.0:
        style = "higher ceiling" if tilt > 0.0 else "steadier floor"
        return f"{swap.gain:.1f} projected, but a {style} raises win probability"
    if swap.player_in is not None:
        return f"+{swap.gain:.1f} projected points"
    return "no legal replacement available"


def _to_wire_swap(swap: DomainSwap, tilt: float = 0.0, *, actual: bool = False) -> Swap:
    return Swap(
        bench=swap.player_out.name if swap.player_out is not None else "(none)",
        starter=swap.player_in.name if swap.player_in is not None else "(none)",
        slot=slot_label(swap.slot),
        gain=round(swap.gain, 1),
        reason=_swap_reason(swap, tilt, actual=actual)[:80],
    )


def strategy_note(
    *,
    my_projected: float,
    opponent_projected: float,
    tilt: float,
    points_win_probability: float,
    win_probability: float,
    points_cost: float,
) -> str | None:
    """The one sentence explaining a lineup that is not the points-maximising one."""
    if tilt == 0.0:
        return None
    if tilt > 0.0:
        situation = f"Projected underdog ({my_projected:.1f} vs {opponent_projected:.1f})"
        move = "higher-ceiling starts"
    else:
        situation = f"Projected favourite ({my_projected:.1f} vs {opponent_projected:.1f})"
        move = "steadier starts"
    return (
        f"{situation}: {move} lift win chance {points_win_probability:.0f}% -> "
        f"{win_probability:.0f}% for {points_cost:.1f} fewer projected points."
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
    playoff_odds_delta: float,
    swaps: Sequence[DomainSwap],
    actual_current: Lineup | None = None,
    actual_optimal: Lineup | None = None,
    actual_swaps: Sequence[DomainSwap] = (),
    win_probability: float | None = None,
    strategy: str | None = None,
    tilt: float = 0.0,
    extra_caveats: Sequence[str] = (),
) -> LineupAdvice:
    """The common case, an already-optimal lineup, is cheap: ``swaps`` and ``caveats`` are
    both empty and the whole result is four numbers.

    ``actual_current``/``actual_optimal`` are ``None`` unless the caller found at least one
    live score for the week. See ``tools_lineup.py::_live_projections``.
    """
    capped = sorted(swaps, key=lambda swap: -swap.gain)[:MAX_SWAPS]
    caveats = [c for player in optimal.started if (c := _caveat(player)) is not None]
    caveats.extend(extra_caveats)
    actual = None
    if actual_current is not None and actual_optimal is not None:
        capped_actual = sorted(actual_swaps, key=lambda swap: -swap.gain)[:MAX_SWAPS]
        actual = ActualLineup(
            points_scored=round(actual_current.projected_points, 1),
            optimal_live_total=round(actual_optimal.projected_points, 1),
            points_left_on_bench=round(
                actual_optimal.projected_points - actual_current.projected_points, 1
            ),
            swaps=[_to_wire_swap(swap, actual=True) for swap in capped_actual],
        )
    return LineupAdvice(
        week=week,
        current_projected=round(current.projected_points, 1),
        optimal_projected=round(optimal.projected_points, 1),
        point_gain=round(optimal.projected_points - current.projected_points, 1),
        playoff_odds_delta=round(playoff_odds_delta, 1),
        win_probability=None if win_probability is None else round(win_probability, 1),
        strategy=strategy,
        swaps=[_to_wire_swap(swap, tilt) for swap in capped],
        caveats=caveats,
        actual=actual,
    )


_MATERIAL = 1.0
"""Started points per week a position must move by to be named in a trade's impact."""

Verdict = Literal["accept", "lean_accept", "neutral", "lean_decline", "decline"]


class TradeVerdict(BaseModel):
    verdict: Verdict
    """Whether *I* should want this trade, by what it does to my playoff odds."""
    my_value_delta: float
    partner_value_delta: float
    my_playoff_odds_delta: float
    positional_impact: str
    partner_likely_accepts: bool | None = None
    """Whether the other manager plausibly says yes: they gain lineup points *and* give up no
    more market value than they get back. ``None`` when market values are unavailable."""
    market: str | None = None
    risks: list[str] = Field(default_factory=list)
    deadline: str | None = None


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


def render_positional_impact(deltas: Mapping[str, float], *, ros: bool = False) -> str:
    """The one sentence a manager wants about shape: which position this trade actually moves.

    ``deltas`` comes from ``domain.trades.positional_impact``: the change in *started* points,
    not in the projections of the players swapped. ``ros`` says they are a rest-of-season
    average rather than this week alone, which this week's byes can make very different.
    """
    if not deltas:
        return "No change to the starting lineup."
    suffix = " ROS" if ros else ""
    material = sorted(
        ((position, delta) for position, delta in deltas.items() if abs(delta) >= _MATERIAL),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:3]
    if len(material) > 1:
        # Several positions move, typically because the flex changes hands: a trade that
        # costs a third running back start and gives the flex to a tight end is "RB -7.0,
        # TE +6.9", and naming only the larger half would describe a different trade.
        moves = ", ".join(f"{position} {delta:+.1f}" for position, delta in material)
        return f"Started pts/wk{suffix}: {moves}."
    position, delta = max(deltas.items(), key=lambda item: (abs(item[1]), item[0]))
    direction = "Strengthens" if delta >= 0 else "Weakens"
    return f"{direction} {position} by {abs(delta):.1f} started pts/wk{suffix}."


def market_sentence(evaluation: TradeEvaluation) -> str | None:
    """What the trade market thinks of the offer, from my side: what I give against what I
    get, and whether that is a realistic ask."""
    given, received = evaluation.market_given, evaluation.market_received
    if given is None or received is None:
        return None
    sides = f"you give {given:,.0f}, get {received:,.0f} (FantasyCalc)"
    if evaluation.market_plausible:
        return f"Market: {sides}."
    return (
        f"Market: {sides}; they give up more than {MARKET_RATIO_LIMIT:g}x what they get, so "
        "expect a decline."
    )


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
    *,
    ros: bool = False,
    deadline: str | None = None,
) -> TradeVerdict:
    odds = evaluation.my_playoff_odds_delta or 0.0
    known = evaluation.market_given is not None and evaluation.market_received is not None
    return TradeVerdict(
        verdict=classify_verdict(odds),
        my_value_delta=round(evaluation.my_value_delta, 1),
        partner_value_delta=round(evaluation.partner_value_delta, 1),
        my_playoff_odds_delta=round(odds, 1),
        positional_impact=render_positional_impact(positional_deltas, ros=ros),
        partner_likely_accepts=(
            evaluation.partner_value_delta > 0 and evaluation.market_plausible if known else None
        ),
        market=market_sentence(evaluation),
        risks=_trade_risks(evaluation.offer.give, evaluation.offer.get, drops),
        deadline=deadline,
    )

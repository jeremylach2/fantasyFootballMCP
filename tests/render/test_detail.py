"""Golden-output tests for ``render/detail.py``'s two small structured results."""

from __future__ import annotations

from ffmcp.domain.models import Lineup, Player, Projection, RosterSlot, TradeOffer
from ffmcp.domain.optimizer import diff
from ffmcp.domain.trades import TradeEvaluation
from ffmcp.render.detail import build_lineup_advice, build_trade_verdict, classify_verdict


def _player(player_id: int, name: str, position: str, points: float, **kw: object) -> Player:
    return Player(
        player_id=player_id,
        name=name,
        position=position,
        eligible_slots=(position,),
        pro_team="BUF",
        projection=Projection(week=3, points=points),
        **kw,  # type: ignore[arg-type]
    )


def test_lineup_advice_is_cheap_when_already_optimal() -> None:
    bench = _player(1, "Josh Allen", "QB", 20.0)
    lineup = Lineup(slots=(RosterSlot(slot="QB", player=bench),), projected_points=20.0)

    advice = build_lineup_advice(
        week=3, current=lineup, optimal=lineup, playoff_odds_delta=0.0, swaps=[]
    )

    assert advice.swaps == []
    assert advice.caveats == []
    assert advice.point_gain == 0.0


def test_lineup_advice_caveats_one_per_injury_status() -> None:
    healthy = _player(1, "Josh Allen", "QB", 20.0)
    questionable = _player(2, "Puka Nacua", "WR", 15.0, injury_status="QUESTIONABLE")
    lineup = Lineup(
        slots=(RosterSlot(slot="QB", player=healthy), RosterSlot(slot="WR", player=questionable)),
        projected_points=35.0,
    )

    advice = build_lineup_advice(
        week=3, current=lineup, optimal=lineup, playoff_odds_delta=0.0, swaps=[]
    )

    assert advice.caveats == ["Questionable: Puka Nacua — check inactives"]


def test_lineup_advice_has_no_actual_section_without_live_data() -> None:
    lineup = Lineup(
        slots=(RosterSlot(slot="QB", player=_player(1, "Josh Allen", "QB", 20.0)),),
        projected_points=20.0,
    )
    advice = build_lineup_advice(
        week=8, current=lineup, optimal=lineup, playoff_odds_delta=0.0, swaps=[]
    )
    assert advice.actual is None


def test_lineup_advice_actual_section_reports_the_live_recap() -> None:
    """A Monday-morning recap where the bench outscored the starter."""
    bench_riser = _player(1, "Bench Riser", "WR", 4.0)
    cold_starter = _player(2, "Cold Starter", "WR", 18.0)
    actual_current = Lineup(
        slots=(RosterSlot(slot="WR", player=cold_starter),), projected_points=2.0
    )
    actual_optimal = Lineup(
        slots=(RosterSlot(slot="WR", player=bench_riser),), projected_points=22.0
    )
    live_swaps = diff(actual_current, actual_optimal, {1: 22.0, 2: 2.0})

    advice = build_lineup_advice(
        week=8,
        current=actual_current,
        optimal=actual_current,
        playoff_odds_delta=0.0,
        swaps=[],
        actual_current=actual_current,
        actual_optimal=actual_optimal,
        actual_swaps=live_swaps,
    )

    assert advice.actual is not None
    assert advice.actual.points_scored == 2.0
    assert advice.actual.optimal_live_total == 22.0
    assert advice.actual.points_left_on_bench == 20.0
    assert len(advice.actual.swaps) == 1
    assert advice.actual.swaps[0].bench == "Cold Starter"
    assert advice.actual.swaps[0].starter == "Bench Riser"
    # The pregame section is untouched: the two are never conflated.
    assert advice.swaps == []
    assert advice.point_gain == 0.0


def test_classify_verdict_bands() -> None:
    assert classify_verdict(5.0) == "accept"
    assert classify_verdict(1.0) == "lean_accept"
    assert classify_verdict(0.0) == "neutral"
    assert classify_verdict(-1.0) == "lean_decline"
    assert classify_verdict(-5.0) == "decline"


def test_trade_verdict_names_the_biggest_positional_shift_and_a_forced_drop() -> None:
    give = (_player(1, "Josh Allen", "QB", 20.0),)
    get = (_player(2, "Saquon Barkley", "RB", 18.0, injury_status="QUESTIONABLE"),)
    evaluation = TradeEvaluation(
        offer=TradeOffer(partner_team_id=2, give=give, get=get),
        my_value_delta=4.0,
        partner_value_delta=-4.0,
        my_playoff_odds_delta=6.0,
    )
    drop = _player(3, "Bench Guy", "RB", 2.0)

    verdict = build_trade_verdict(
        evaluation, drops=(drop,), positional_deltas={"RB": 18.0, "QB": -20.0}
    )

    assert verdict.verdict == "accept"
    # The lineup loses more at QB than it gains at RB, so QB is the shift worth naming.
    assert verdict.positional_impact == "Weakens QB by 20.0 started pts/wk."
    assert verdict.risks == [
        "Saquon Barkley is Questionable",
        "Must drop B. Guy to stay at roster size",
    ]

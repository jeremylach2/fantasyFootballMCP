"""A week's playoff stakes, and a trade list of distinct ideas rather than variants."""

from __future__ import annotations

from test_simulate import make_league

from ffmcp.domain.models import Player, TradeOffer
from ffmcp.domain.simulate import week_stakes
from ffmcp.domain.trades import TradeEvaluation, _distinct


def test_winning_this_week_is_worth_playoff_odds_to_a_bubble_team() -> None:
    state = make_league([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], current_week=12, playoff_team_count=3)
    stakes = week_stakes(state, 1, 12, n_sims=4_000)
    assert stakes is not None
    assert stakes.playoff_odds_if_win > stakes.playoff_odds_if_loss
    assert stakes.leverage > 5.0


def test_a_week_outside_the_remaining_schedule_has_no_stakes() -> None:
    state = make_league([1.0, 1.0, 1.0, 1.0], current_week=12)
    assert week_stakes(state, 1, 3, n_sims=500) is None


def _p(player_id: int) -> Player:
    return Player(
        player_id=player_id,
        name=f"P{player_id}",
        position="RB",
        eligible_slots=("RB",),
        pro_team="FA",
    )


def _evaluation(partner: int, give: list[int], get: list[int]) -> TradeEvaluation:
    offer = TradeOffer(
        partner_team_id=partner,
        give=tuple(_p(i) for i in give),
        get=tuple(_p(i) for i in get),
    )
    return TradeEvaluation(offer=offer, my_value_delta=5.0, partner_value_delta=5.0)


def test_variants_of_a_trade_already_shown_are_skipped() -> None:
    offers = [
        _evaluation(2, [1], [10]),
        _evaluation(2, [1], [10, 11]),  # same core with a sweetener: a variant
        _evaluation(2, [1, 3], [10]),  # likewise
        _evaluation(2, [4], [12]),  # a different trade with the same partner
        _evaluation(3, [1], [30]),  # the same player offered to someone else
    ]
    kept = _distinct(offers, 10)
    assert [
        ([p.player_id for p in e.offer.give], [p.player_id for p in e.offer.get]) for e in kept
    ] == [([1], [10]), ([4], [12]), ([1], [30])]
    assert len(_distinct(offers, 2)) == 2

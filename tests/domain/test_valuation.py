"""Replacement level follows the league's own settings, and the market term
stays inside the cap that stops hype from outvoting projections.
"""

from __future__ import annotations

from ffmcp.domain.models import LeagueSettings, MarketSignal, Player, Projection
from ffmcp.domain.valuation import (
    MARKET_ADJUSTMENT_CAP,
    market_adjustment,
    replacement_level,
    starters_per_team,
    value_over_replacement,
)


def make_settings(*, team_count: int = 12, **slots: int) -> LeagueSettings:
    counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "BE": 6}
    counts.update(slots)
    return LeagueSettings(
        league_id=1234567,
        season=2026,
        team_count=team_count,
        playoff_team_count=6,
        reg_season_weeks=14,
        slot_counts=counts,
    )


def pool(position: str, count: int, *, top: float = 30.0, step: float = 0.5) -> list[Player]:
    """``count`` players at one position, projecting ``top`` down to ``top - count * step``."""
    return [
        Player(
            player_id=hash(position) % 1000 * 1000 + index,
            name=f"{position} {index + 1}",
            position=position,
            eligible_slots=(position,),
            pro_team="FA",
            projection=Projection(week=1, points=round(top - index * step, 2)),
        )
        for index in range(count)
    ]


def test_starters_per_team_counts_flex_slots_fractionally() -> None:
    assert starters_per_team("RB", make_settings()) == 2.0
    assert starters_per_team("RB", make_settings(**{"RB/WR/TE": 1})) == 2.0 + 1 / 3
    assert starters_per_team("QB", make_settings(OP=1)) == 1.25
    assert starters_per_team("QB", make_settings()) == 1.0
    # Bench and IR are not starting slots and must never create demand.
    assert starters_per_team("RB", make_settings(BE=20)) == 2.0


def test_replacement_level_falls_as_the_league_gets_bigger() -> None:
    backs = pool("RB", 60)
    small = replacement_level("RB", make_settings(team_count=8), backs)
    large = replacement_level("RB", make_settings(team_count=12), backs)
    assert large < small
    # 8 teams x 2 starters means the 17th back is replacement; 12 teams means the 25th.
    assert small == backs[16].projection.points  # type: ignore[union-attr]
    assert large == backs[24].projection.points  # type: ignore[union-attr]


def test_replacement_level_falls_as_the_league_starts_more_of_a_position() -> None:
    backs = pool("RB", 60)
    two = replacement_level("RB", make_settings(RB=2), backs)
    three = replacement_level("RB", make_settings(RB=3), backs)
    with_flex = replacement_level("RB", make_settings(RB=2, **{"RB/WR/TE": 1}), backs)
    assert three < with_flex < two


def test_replacement_level_survives_a_pool_too_shallow_to_reach_the_cutoff() -> None:
    assert replacement_level("RB", make_settings(), []) == 0.0
    thin = pool("RB", 3)
    assert replacement_level("RB", make_settings(), thin) == thin[-1].projection.points  # type: ignore[union-attr]


def test_market_adjustment_is_capped_no_matter_how_hyped_the_player_is() -> None:
    player = pool("WR", 1)[0]
    assert market_adjustment(player) == 1.0
    assert market_adjustment(player, MarketSignal(trending_adds=0)) == 1.0

    warm = market_adjustment(player, MarketSignal(trending_adds=5_000))
    hot = market_adjustment(player, MarketSignal(trending_adds=80_000))
    absurd = market_adjustment(player, MarketSignal(trending_adds=10**9))

    assert 1.0 < warm < hot < 1.0 + MARKET_ADJUSTMENT_CAP
    assert absurd <= 1.0 + MARKET_ADJUSTMENT_CAP
    # A negative count is upstream nonsense, not a reason to penalise a player.
    assert market_adjustment(player, MarketSignal(trending_adds=-50)) == 1.0


def test_market_hype_can_never_outweigh_a_real_projection_gap() -> None:
    """The cap's whole job. Hype may reorder two players already within 5% of each other, and
    may not touch a ranking decided by a gap wider than that."""
    settings = make_settings()
    backs = pool("RB", 40)
    viral = MarketSignal(trending_adds=10**9)

    hyped = backs[6]
    plain = value_over_replacement(hyped, settings, backs)
    boosted = value_over_replacement(hyped, settings, backs, market_signal=viral)
    assert boosted - plain <= MARKET_ADJUSTMENT_CAP * plain + 1e-9

    clearly_better = backs[5]  # ~5.5% more value than the hyped player
    assert boosted < value_over_replacement(clearly_better, settings, backs)

    # A near-tie is exactly what the nudge is allowed to settle.
    near_tie = backs[6].model_copy(
        update={"player_id": 999, "projection": Projection(week=1, points=27.05)}
    )
    assert boosted > value_over_replacement(near_tie, settings, backs)


def test_value_over_replacement_scales_with_the_remaining_schedule() -> None:
    settings = make_settings()
    backs = pool("RB", 40)
    weekly = value_over_replacement(backs[0], settings, backs)
    rest_of_season = value_over_replacement(backs[0], settings, backs, weeks_remaining=10)

    assert weekly > 0
    assert rest_of_season == weekly * 10
    # A replacement-level player is worth approximately nothing, which is the point.
    assert value_over_replacement(backs[24], settings, backs) == 0.0


def test_value_over_replacement_treats_a_player_with_no_game_as_worthless() -> None:
    settings = make_settings()
    backs = pool("RB", 40)
    on_bye = backs[0].model_copy(update={"projection": None})
    assert value_over_replacement(on_bye, settings, backs) == 0.0

"""Expected points from volume, luck as the gap, and labels only where the evidence is."""

from __future__ import annotations

import pytest

from ffmcp.domain.models import UsageWeek
from ffmcp.domain.usage import (
    LUCK_THRESHOLD,
    MIN_GAMES,
    backup_running_back,
    expected_points,
    usage_profiles,
)


def _game(
    player_id: int,
    week: int,
    *,
    position: str = "WR",
    team: str = "DET",
    targets: float = 0.0,
    air: float = 0.0,
    carries: float = 0.0,
    points: float = 0.0,
    snaps: float | None = 0.8,
) -> UsageWeek:
    return UsageWeek(
        season=2026,
        week=week,
        player_id=player_id,
        name=f"P{player_id}",
        position=position,
        pro_team=team,
        snap_pct=snaps,
        targets=targets,
        receiving_air_yards=air,
        carries=carries,
        fantasy_points=points,
    )


def test_a_target_is_worth_more_under_ppr_than_standard() -> None:
    game = _game(1, 1, targets=10.0, air=90.0)
    ppr = expected_points(game, reception_points=1.0)
    half = expected_points(game, reception_points=0.5)
    standard = expected_points(game, reception_points=0.0)
    assert ppr is not None and half is not None and standard is not None
    assert ppr > half > standard
    assert half == pytest.approx((ppr + standard) / 2)  # least squares is linear in its target


def test_kickers_and_defenses_are_not_modelled() -> None:
    assert expected_points(_game(1, 1, position="K")) is None


def test_scoring_well_above_volume_is_labelled_sell_high_only_after_enough_games() -> None:
    games = [_game(1, week, targets=4.0, air=30.0, points=19.0) for week in range(1, MIN_GAMES + 1)]
    profile = usage_profiles(games)[1]
    assert profile.luck_ppg >= LUCK_THRESHOLD
    assert profile.signal == "sell_high"
    early = usage_profiles(games[: MIN_GAMES - 1])[1]
    assert early.signal is None


def test_heavy_volume_with_little_scoring_is_buy_low() -> None:
    games = [_game(1, week, targets=11.0, air=110.0, points=9.0) for week in range(1, 5)]
    assert usage_profiles(games)[1].signal == "buy_low"


def test_a_low_volume_player_is_never_a_trade_target_however_unlucky() -> None:
    games = [_game(1, week, targets=2.0, air=10.0, points=0.0) for week in range(1, 5)]
    assert usage_profiles(games)[1].signal is None


def test_before_enough_games_the_gap_is_visible_but_not_a_call() -> None:
    games = [_game(1, week, targets=11.0, air=110.0, points=9.0) for week in range(1, MIN_GAMES)]
    profile = usage_profiles(games)[1]
    assert profile.signal is None
    assert profile.gap_direction == "buy_low"
    assert usage_profiles(games[:1])[1].gap_direction is None  # one game is one game


def test_the_handcuff_is_the_lead_backs_highest_snap_teammate() -> None:
    games = [
        _game(1, 1, position="RB", snaps=0.75, carries=18.0),
        _game(2, 1, position="RB", snaps=0.30, carries=6.0),
        _game(3, 1, position="RB", snaps=0.05, carries=1.0),
        _game(4, 1, position="RB", team="CHI", snaps=0.9, carries=20.0),
    ]
    profiles = usage_profiles(games)
    backup = backup_running_back(1, "DET", profiles)
    assert backup is not None and backup.player_id == 2


def test_a_committee_back_has_no_handcuff() -> None:
    games = [
        _game(1, 1, position="RB", snaps=0.75, carries=18.0),
        _game(2, 1, position="RB", snaps=0.30, carries=6.0),
    ]
    assert backup_running_back(2, "DET", usage_profiles(games)) is None

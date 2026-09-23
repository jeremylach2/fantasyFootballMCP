"""The variance model: measured position spreads, widened where projection sources disagree."""

from __future__ import annotations

import pytest

from ffmcp.domain.models import Player, Projection
from ffmcp.domain.variance import (
    MIN_PLAYER_SD,
    POSITION_SD_FRACTION,
    disagreement,
    disagreement_multiplier,
    floor_ceiling,
    player_sd,
    player_sds,
)


def _player(player_id: int, position: str, points: float | None) -> Player:
    return Player(
        player_id=player_id,
        name=f"P{player_id}",
        position=position,
        eligible_slots=(position,),
        pro_team="DET",
        projection=None if points is None else Projection(week=3, points=points),
    )


def test_a_player_gets_his_positions_measured_spread() -> None:
    wr = _player(1, "WR", 15.0)
    assert player_sd(wr, 15.0) == pytest.approx(POSITION_SD_FRACTION["WR"] * 15.0)


def test_quarterbacks_are_steadier_than_defenses() -> None:
    assert POSITION_SD_FRACTION["QB"] < POSITION_SD_FRACTION["WR"] < POSITION_SD_FRACTION["D/ST"]


def test_the_sd_never_falls_below_the_floor() -> None:
    assert player_sd(_player(1, "K", 1.0), 1.0) == MIN_PLAYER_SD


def test_disagreement_is_relative_and_symmetric_in_sign() -> None:
    assert disagreement(10.0, 8.0) == pytest.approx(0.2)
    assert disagreement(10.0, 12.0) == pytest.approx(0.2)
    assert disagreement(None, 8.0) is None
    assert disagreement(0.2, 1.2) == pytest.approx(1.0)  # floored denominator, not 5.0


def test_typical_disagreement_leaves_the_spread_alone_and_large_disagreement_widens_it() -> None:
    assert disagreement_multiplier(None) == 1.0
    assert disagreement_multiplier(0.08) == pytest.approx(1.0)
    assert disagreement_multiplier(0.5) > 1.3
    assert disagreement_multiplier(5.0) == disagreement_multiplier(1.0)  # capped


def test_player_sds_skips_players_without_a_game() -> None:
    sds = player_sds([_player(1, "WR", 10.0), _player(2, "WR", None)], alternative={1: 5.0})
    assert set(sds) == {1}
    assert sds[1] > player_sd(_player(1, "WR", 10.0), 10.0)


def test_floor_and_ceiling_bracket_the_projection_and_never_go_negative() -> None:
    low, high = floor_ceiling(10.0, 5.0)
    assert low == pytest.approx(10.0 - 1.2816 * 5.0, abs=1e-3)
    assert high == pytest.approx(10.0 + 1.2816 * 5.0, abs=1e-3)
    assert floor_ceiling(3.0, 10.0)[0] == 0.0

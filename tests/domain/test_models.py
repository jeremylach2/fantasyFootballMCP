"""Domain models are frozen and carry no ESPN/Sleeper vocabulary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ffmcp.domain.models import Player, Projection


def test_player_is_frozen() -> None:
    player = Player(
        player_id=1,
        name="QB One",
        position="QB",
        eligible_slots=("QB",),
        pro_team="BUF",
    )
    with pytest.raises(ValidationError):
        player.name = "someone else"


def test_projection_defaults_source_to_espn() -> None:
    assert Projection(week=1, points=10.0).source == "espn"

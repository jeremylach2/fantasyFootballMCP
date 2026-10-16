"""Live win probability: the score so far plus what is left (``domain.live``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ffmcp.domain.live import GAME_LENGTH, live_score_model, share_left
from ffmcp.domain.models import Lineup, Player, Projection, RosterSlot
from ffmcp.domain.simulate import lineup_score_model, matchup_win_probability

KICKOFF = datetime(2026, 10, 18, 17, 0, tzinfo=UTC)


def _player(player_id: int, points: float, live: float | None, kickoff: datetime | None) -> Player:
    return Player(
        player_id=player_id,
        name=f"P{player_id}",
        position="WR",
        eligible_slots=("WR",),
        pro_team="DET",
        projection=Projection(week=6, points=points),
        live_points=live,
        kickoff=kickoff,
    )


def test_the_share_left_runs_from_kickoff_to_the_final_whistle() -> None:
    player = _player(1, 15.0, None, KICKOFF)
    assert share_left(player, KICKOFF - timedelta(hours=1)) == 1.0
    assert share_left(player, KICKOFF + GAME_LENGTH / 2) == pytest.approx(0.5)
    assert share_left(player, KICKOFF + GAME_LENGTH + timedelta(minutes=5)) == 0.0
    assert share_left(_player(2, 15.0, None, None), KICKOFF) == 0.0  # no game this week


def test_before_kickoff_the_live_model_is_the_pregame_one() -> None:
    starters = [_player(1, 15.0, None, KICKOFF), _player(2, 10.0, None, KICKOFF)]
    live, left = live_score_model(1, starters, KICKOFF - timedelta(minutes=1))
    lineup = Lineup(
        slots=tuple(RosterSlot(slot="WR", player=p) for p in starters), projected_points=25.0
    )
    pregame = lineup_score_model(1, lineup)
    assert live.mean == pytest.approx(pregame.mean)
    assert live.sd == pytest.approx(pregame.sd)
    assert left == pytest.approx(25.0)


def test_a_lead_with_nothing_left_to_play_is_a_win() -> None:
    after = KICKOFF + GAME_LENGTH + timedelta(minutes=1)
    mine, my_left = live_score_model(1, [_player(1, 15.0, 31.0, KICKOFF)], after)
    theirs, _ = live_score_model(2, [_player(2, 25.0, 12.0, KICKOFF)], after)
    assert my_left == 0.0
    assert matchup_win_probability(mine, theirs) == 100.0


def test_halfway_through_the_score_counts_and_the_spread_shrinks() -> None:
    halfway = KICKOFF + GAME_LENGTH / 2
    live, left = live_score_model(1, [_player(1, 20.0, 14.0, KICKOFF)], halfway)
    pregame, _ = live_score_model(1, [_player(1, 20.0, None, KICKOFF)], KICKOFF)
    assert left == pytest.approx(10.0)
    assert live.mean == pytest.approx(24.0)
    assert live.sd == pytest.approx(pregame.sd / 2**0.5)

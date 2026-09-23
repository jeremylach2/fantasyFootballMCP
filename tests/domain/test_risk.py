"""Win-probability lineups: chase variance as an underdog, protect a lead as a favourite, and
never trade points for a rounding-error gain."""

from __future__ import annotations

from ffmcp.domain.models import Player, Projection
from ffmcp.domain.risk import best_lineup_for_matchup
from ffmcp.domain.simulate import ScoreModel

SLOTS = ("QB", "WR")


def _player(player_id: int, position: str, points: float) -> Player:
    return Player(
        player_id=player_id,
        name=f"P{player_id}",
        position=position,
        eligible_slots=(position,),
        pro_team="DET",
        projection=Projection(week=3, points=points),
    )


QB = _player(1, "QB", 20.0)
STEADY = _player(2, "WR", 14.0)
BOOM_BUST = _player(3, "WR", 13.0)
ROSTER = [QB, STEADY, BOOM_BUST]
SDS = {1: 8.0, 2: 3.0, 3: 14.0}


def _started(choice_players: tuple[Player, ...]) -> set[int]:
    return {p.player_id for p in choice_players}


def test_a_big_underdog_starts_the_boom_or_bust_receiver() -> None:
    opponent = ScoreModel(team_id=2, mean=60.0, sd=10.0)
    choice = best_lineup_for_matchup(1, ROSTER, SLOTS, opponent, sds=SDS)

    assert _started(choice.points_lineup.started) == {1, 2}
    assert _started(choice.lineup.started) == {1, 3}
    assert choice.tilt > 0.0
    assert choice.win_probability > choice.points_win_probability
    assert choice.lineup.projected_points == 33.0  # true projection, not the tilted score


def test_a_big_favourite_keeps_the_steady_receiver() -> None:
    opponent = ScoreModel(team_id=2, mean=15.0, sd=10.0)
    choice = best_lineup_for_matchup(1, ROSTER, SLOTS, opponent, sds=SDS)
    assert _started(choice.lineup.started) == {1, 2}
    assert not choice.differs


def test_a_negligible_gain_does_not_cost_projected_points() -> None:
    opponent = ScoreModel(team_id=2, mean=34.5, sd=10.0)  # nearly even: variance barely matters
    choice = best_lineup_for_matchup(1, ROSTER, SLOTS, opponent, sds=SDS)
    assert choice.tilt == 0.0
    assert choice.lineup == choice.points_lineup


def test_every_slot_still_fills_when_a_negative_tilt_would_score_a_player_below_zero() -> None:
    opponent = ScoreModel(team_id=2, mean=0.0, sd=1.0)
    choice = best_lineup_for_matchup(1, [QB, BOOM_BUST], SLOTS, opponent, sds={1: 8.0, 3: 40.0})
    assert all(slot.player is not None for slot in choice.lineup.slots)

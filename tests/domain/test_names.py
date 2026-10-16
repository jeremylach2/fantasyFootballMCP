"""Names are matched the way people type them (``domain.names``)."""

from __future__ import annotations

from ffmcp.domain.models import Player
from ffmcp.domain.names import match_players, normalize_name, suggestions


def _p(player_id: int, name: str) -> Player:
    return Player(
        player_id=player_id, name=name, position="WR", eligible_slots=("WR",), pro_team="DET"
    )


POOL = [
    _p(1, "A.J. Brown"),
    _p(2, "Marvin Harrison Jr."),
    _p(3, "Saquon Barkley"),
    _p(4, "Amon-Ra St. Brown"),
    _p(5, "D'Andre Swift"),
    _p(6, "Sterling Shepard"),
]


def test_normalizing_drops_case_punctuation_and_suffixes() -> None:
    assert normalize_name("A.J. Brown") == normalize_name("aj brown") == "aj brown"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("Amon-Ra St. Brown") == normalize_name("Amon Ra St Brown")
    assert normalize_name("D" + chr(0x2019) + "Andre Swift") == "dandre swift"


def test_typed_names_find_espns_spelling() -> None:
    assert [p.player_id for p in match_players(POOL, "AJ Brown")] == [1]
    assert [p.player_id for p in match_players(POOL, "marvin harrison")] == [2]
    assert [p.player_id for p in match_players(POOL, "Dandre Swift")] == [5]


def test_an_ambiguous_fragment_returns_every_match() -> None:
    assert {p.player_id for p in match_players(POOL, "brown")} == {1, 4}


def test_a_typo_gets_suggestions_not_a_guess() -> None:
    assert match_players(POOL, "Saquon Barkly") == []
    assert suggestions(POOL, "Saquon Barkly") == ["Saquon Barkley"]
    assert suggestions(POOL, "Barkly") == ["Saquon Barkley"]
    assert suggestions(POOL, "Zzzz Qqqq") == []

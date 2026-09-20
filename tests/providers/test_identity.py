"""ESPN<->Sleeper identity resolution never raises on a miss."""

from __future__ import annotations

from ffmcp.providers.identity import IdentityIndex, normalize


def test_normalize_strips_suffix_and_punctuation() -> None:
    assert normalize("Odell Beckham Jr.", "wr", "buf") == ("odell beckham", "WR", "BUF")


def test_resolve_hit() -> None:
    index = IdentityIndex.build({"9001": {"name": "Josh Allen", "pos": "QB", "team": "BUF"}})
    assert index.resolve(101, "Josh Allen", "QB", "BUF") == "9001"
    assert index.unresolved == 0


def test_resolve_miss_increments_unresolved_and_does_not_raise() -> None:
    index = IdentityIndex.build({"9001": {"name": "Josh Allen", "pos": "QB", "team": "BUF"}})
    result = index.resolve(999, "Nobody Real", "RB", "ZZZ")
    assert result is None
    assert index.unresolved == 1


def test_defense_resolves_by_team_abbreviation_not_name() -> None:
    # ESPN: position "D/ST", name "Falcons D/ST". Sleeper: position "DEF", name the full team
    # name. Neither token would ever match through normalize(); only the shared team
    # abbreviation does.
    index = IdentityIndex.build({"ATL": {"name": "Atlanta Falcons", "pos": "DEF", "team": "ATL"}})
    assert index.resolve(1, "Falcons D/ST", "D/ST", "ATL") == "ATL"
    assert index.unresolved == 0


def test_defense_resolves_across_the_washington_abbreviation_mismatch() -> None:
    # ESPN calls the team WSH; Sleeper calls it WAS.
    index = IdentityIndex.build(
        {"WAS": {"name": "Washington Commanders", "pos": "DEF", "team": "WAS"}}
    )
    assert index.resolve(1, "Commanders D/ST", "D/ST", "WSH") == "WAS"


def test_defense_does_not_fall_back_to_the_name_based_key() -> None:
    # A defense entry must never land in by_key, even though it satisfies the same shape.
    index = IdentityIndex.build({"ATL": {"name": "Atlanta Falcons", "pos": "DEF", "team": "ATL"}})
    assert index.by_key == {}
    assert index.by_defense_team == {"ATL": "ATL"}

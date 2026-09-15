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

"""Scaffold-level checks: the package imports and the server constructs."""

from __future__ import annotations

import ffmcp
from ffmcp.server import CACHE_HINTS, build_server


def test_package_exposes_version() -> None:
    assert ffmcp.__version__


def test_server_builds() -> None:
    mcp = build_server()
    assert mcp.name == "fantasy-football"


def test_cache_hints_target_real_methods() -> None:
    """Guard against a typo silently disabling caching."""
    from mcp.server.caching import CACHEABLE_METHODS

    assert set(CACHE_HINTS) <= set(CACHEABLE_METHODS)


def test_league_data_is_never_publicly_cacheable() -> None:
    """resources/read carries roster data; a shared cache must not keep it."""
    assert CACHE_HINTS["resources/read"].scope == "private"

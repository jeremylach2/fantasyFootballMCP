"""SleeperMarketProvider against httpx2.MockTransport: zero real network.

``respx`` targets httpx 0.x and does not intercept ``httpx2``, so tests
here stub responses directly with ``httpx2.MockTransport``.
"""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest

from ffmcp.domain.models import Player
from ffmcp.providers.cache import Cache
from ffmcp.providers.sleeper import SleeperMarketProvider

PLAYERS = {
    "9001": {
        "full_name": "Josh Allen",
        "position": "QB",
        "team": "BUF",
    },
    "9002": {
        "full_name": "No Position Guy",
        "position": None,  # e.g. a practice-squad entry; must be filtered out
        "team": "BUF",
    },
}

TRENDING = [{"player_id": "9001", "count": 4200}]

PROJECTIONS = [
    {"player_id": "9001", "stats": {"pts_std": 18.0, "pts_ppr": 22.0}},
    {"player_id": "9003", "stats": {"pts_std": 5.0}},  # no PPR figure: skipped, not guessed
]


def _handler(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == "/v1/state/nfl":
        return httpx2.Response(200, json={"season": "2026", "week": 3})
    if request.url.path == "/v1/players/nfl":
        return httpx2.Response(200, json=PLAYERS)
    if request.url.path == "/v1/players/nfl/trending/add":
        return httpx2.Response(200, json=TRENDING)
    if request.url.path == "/projections/nfl/2026/3":
        return httpx2.Response(200, json=PROJECTIONS)
    return httpx2.Response(404)


def _provider(tmp_path: Path) -> SleeperMarketProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_handler))
    return SleeperMarketProvider(client, Cache(tmp_path))


async def test_get_current_week(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    assert await provider.get_current_week() == 3


async def test_get_trending_adds(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    assert await provider.get_trending_adds() == {"9001": 4200}


async def test_player_index_drops_entries_without_a_position(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    identity = await provider._get_identity()  # white-box: exercising the reduction directly
    assert identity.resolve(1, "Josh Allen", "QB", "BUF") == "9001"
    assert identity.resolve(2, "No Position Guy", "RB", "BUF") is None


async def test_get_market_signal_resolves_and_reports_trending(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    signal = await provider.get_market_signal(
        espn_player_id=1, name="Josh Allen", position="QB", pro_team="BUF"
    )
    assert signal is not None
    assert signal.trending_adds == 4200


async def test_get_market_signal_returns_none_on_identity_miss(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    signal = await provider.get_market_signal(
        espn_player_id=999, name="Nobody Real", position="RB", pro_team="ZZZ"
    )
    assert signal is None


async def test_upstream_error_is_translated(tmp_path: Path) -> None:
    def broken_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503)

    from ffmcp.errors import UpstreamUnavailable

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(broken_handler))
    provider = SleeperMarketProvider(client, Cache(tmp_path))
    with pytest.raises(UpstreamUnavailable):
        await provider.get_current_week()


async def test_alt_projections_interpolate_the_leagues_reception_scoring(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    allen = Player(
        player_id=1, name="Josh Allen", position="QB", eligible_slots=("QB",), pro_team="BUF"
    )
    stranger = Player(
        player_id=2, name="Nobody Known", position="RB", eligible_slots=("RB",), pro_team="BUF"
    )
    assert await provider.get_alt_projections(3, [allen, stranger]) == {1: 22.0}
    half = await provider.get_alt_projections(3, [allen], reception_points=0.5)
    assert half == {1: 20.0}

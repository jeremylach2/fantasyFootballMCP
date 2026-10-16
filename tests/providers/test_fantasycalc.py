"""FantasyCalcValueProvider against httpx2.MockTransport: zero real network."""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest

from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache
from ffmcp.providers.fantasycalc import FantasyCalcValueProvider, ppr_parameter

ROWS = [
    {"player": {"name": "Mike Evans", "espnId": "16737", "position": "WR"}, "value": 3190},
    {"player": {"name": "Jacoby Brissett", "espnId": "2578570", "position": "QB"}, "value": 267},
    {"player": {"name": "No Espn Id", "position": "WR"}, "value": 50},
]

requests_seen: list[str] = []


def _handler(request: httpx2.Request) -> httpx2.Response:
    requests_seen.append(str(request.url))
    if request.url.path == "/values/current":
        return httpx2.Response(200, json=ROWS)
    return httpx2.Response(404)


def _provider(tmp_path: Path, handler: object = _handler) -> FantasyCalcValueProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))  # type: ignore[arg-type]
    return FantasyCalcValueProvider(client, Cache(tmp_path))


async def test_values_are_keyed_by_espn_id_for_the_leagues_shape(tmp_path: Path) -> None:
    requests_seen.clear()
    values = await _provider(tmp_path).get_trade_values(
        team_count=16, quarterbacks=1, reception_points=1.0
    )
    assert values == {16737: 3190.0, 2578570: 267.0}
    assert "numTeams=16" in requests_seen[0]
    assert "numQbs=1" in requests_seen[0]
    assert "ppr=1" in requests_seen[0]


async def test_values_are_fetched_once_a_day(tmp_path: Path) -> None:
    requests_seen.clear()
    provider = _provider(tmp_path)
    for _ in range(3):
        await provider.get_trade_values(team_count=16, quarterbacks=1, reception_points=1.0)
    assert len(requests_seen) == 1


async def test_an_outage_is_upstream_unavailable(tmp_path: Path) -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503)

    with pytest.raises(UpstreamUnavailable):
        await _provider(tmp_path, down).get_trade_values(
            team_count=12, quarterbacks=1, reception_points=0.5
        )


def test_reception_scoring_rounds_to_the_nearest_published_format() -> None:
    assert ppr_parameter(1.0) == 1.0
    assert ppr_parameter(0.5) == 0.5
    assert ppr_parameter(0.0) == 0.0
    assert ppr_parameter(0.7) == 0.5

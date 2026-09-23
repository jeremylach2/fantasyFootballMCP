"""Betting lines become implied team totals, and a missing key is a quiet no-op."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx2
import pytest

from ffmcp.providers.cache import Cache
from ffmcp.providers.odds import OddsApiProvider, game_lines_from_events


def _event(
    home: str, away: str, spreads: list[float], totals: list[float], when: str
) -> dict[str, Any]:
    return {
        "home_team": home,
        "away_team": away,
        "commence_time": when,
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "point": spread},
                            {"name": away, "point": -spread},
                        ],
                    },
                    {"key": "totals", "outcomes": [{"name": "Over", "point": total}]},
                ]
            }
            for spread, total in zip(spreads, totals, strict=True)
        ],
    }


def test_implied_totals_split_the_total_by_the_spread() -> None:
    lines = game_lines_from_events(
        [_event("Green Bay Packers", "Atlanta Falcons", [-5.5], [43.5], "2026-09-25T00:15:00Z")]
    )
    assert lines["GB"].implied_total == pytest.approx(24.5)
    assert lines["ATL"].implied_total == pytest.approx(19.0)
    assert lines["GB"].spread == -5.5 and lines["ATL"].spread == 5.5
    assert lines["GB"].opponent == "ATL"


def test_the_line_is_the_median_across_books_and_only_the_next_game_counts() -> None:
    lines = game_lines_from_events(
        [
            _event("Detroit Lions", "New York Jets", [-8.0], [50.0], "2026-10-04T17:00:00Z"),
            _event(
                "Detroit Lions",
                "New York Jets",
                [-6.0, -6.5, -7.0],
                [47.0, 47.5, 48.0],
                "2026-09-27T17:00:00Z",
            ),
        ]
    )
    assert lines["DET"].spread == -6.5
    assert lines["DET"].total == 47.5


def test_washington_uses_espns_abbreviation() -> None:
    lines = game_lines_from_events(
        [_event("Washington Commanders", "Seattle Seahawks", [7.0], [40.5], "2026-09-27")]
    )
    assert set(lines) == {"WSH", "SEA"}


async def test_without_a_key_nothing_is_fetched(tmp_path: Path) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request should be made without a key")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(refuse)) as client:
        provider = OddsApiProvider(client, Cache(tmp_path), api_key=None)
        assert await provider.get_game_lines() == {}

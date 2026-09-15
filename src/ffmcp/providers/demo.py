"""Fixture-backed providers used when ``FFMCP_MODE=demo`` (docs/architecture.md §6).

Selected once, in ``lifespan``. No branching on mode inside a tool. Both classes implement
the same protocols as ``providers/espn.py`` and ``providers/sleeper.py`` structurally, so a
tool adapter cannot tell the difference.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ffmcp.domain.models import LeagueSettings, MarketSignal, Matchup, Player, Team


def _fixtures_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "tests" / "fixtures"
    raise FileNotFoundError("could not locate tests/fixtures relative to the installed package")


@lru_cache(maxsize=1)
def _load_league_fixture() -> dict[str, Any]:
    path = _fixtures_dir() / "demo_league.json"
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


@lru_cache(maxsize=1)
def _load_market_fixture() -> dict[str, Any]:
    path = _fixtures_dir() / "demo_market.json"
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


class DemoLeagueProvider:
    """``LeagueProvider`` backed by the committed anonymized fixture league."""

    def __init__(self) -> None:
        data = _load_league_fixture()
        self._settings = LeagueSettings.model_validate(data["settings"])
        self._teams = [Team.model_validate(t) for t in data["teams"]]
        self._current_week = int(data["current_week"])
        self._matchups = {
            int(week): [Matchup.model_validate(m) for m in matchups]
            for week, matchups in data["matchups"].items()
        }
        self._free_agents = {
            int(week): [Player.model_validate(p) for p in players]
            for week, players in data["free_agents"].items()
        }

    async def get_current_week(self) -> int:
        return self._current_week

    async def get_settings(self) -> LeagueSettings:
        return self._settings

    async def get_teams(self) -> list[Team]:
        return list(self._teams)

    async def get_matchups(self, week: int) -> list[Matchup]:
        return list(self._matchups.get(week, []))

    async def get_free_agents(
        self, week: int, *, size: int = 50, position: str | None = None
    ) -> list[Player]:
        players = self._free_agents.get(week, [])
        if position is not None:
            players = [p for p in players if p.position == position]
        return players[:size]


class DemoMarketProvider:
    """``MarketProvider`` backed by the committed fixture. Bypasses real identity resolution
    since the fixture already keys signals by ESPN player id."""

    def __init__(self) -> None:
        data = _load_market_fixture()
        self._current_season = int(data["current_season"])
        self._current_week = int(data["current_week"])
        self._trending: dict[str, int] = dict(data["trending_adds"])
        self._by_espn_id: dict[int, MarketSignal] = {
            int(player_id): MarketSignal.model_validate(signal)
            for player_id, signal in data["market_by_espn_id"].items()
        }

    async def get_current_season(self) -> int:
        return self._current_season

    async def get_current_week(self) -> int:
        return self._current_week

    async def get_trending_adds(
        self, *, lookback_hours: int = 24, limit: int = 25
    ) -> dict[str, int]:
        return dict(list(self._trending.items())[:limit])

    async def get_market_signal(
        self, *, espn_player_id: int, name: str, position: str, pro_team: str
    ) -> MarketSignal | None:
        return self._by_espn_id.get(espn_player_id)

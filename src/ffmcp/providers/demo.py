"""Fixture-backed providers used when ``FFMCP_MODE=demo`` (docs/architecture.md §6).

Selected once, in ``lifespan``. No branching on mode inside a tool. Both classes implement
the same protocols as ``providers/espn.py`` and ``providers/sleeper.py`` structurally, so a
tool adapter cannot tell the difference.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ffmcp.domain.models import (
    GameLine,
    LeagueSettings,
    MarketSignal,
    Matchup,
    Player,
    PlayerWeek,
    Roster,
    Team,
    UsageWeek,
)


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
def _load_insights_fixture() -> dict[str, Any]:
    """History, usage, second-opinion projections and game lines, from
    ``scripts/make_demo_insights.py``."""
    path = _fixtures_dir() / "demo_insights.json"
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _with_calendar(player: Player) -> Player:
    """Attach the fixture's bye week and, for a player on bye this week, his season rate: the
    two fields the league fixture predates."""
    insights = _load_insights_fixture()
    rate = insights["season_rates"].get(str(player.player_id))
    return player.model_copy(
        update={
            "bye_week": insights["bye_weeks"].get(player.pro_team),
            "season_rate": rate if rate is not None else player.season_rate,
        }
    )


def _team_with_calendar(team: Team) -> Team:
    roster = team.roster
    return team.model_copy(
        update={
            "roster": Roster(
                slots=tuple(
                    s.model_copy(update={"player": _with_calendar(s.player)}) if s.player else s
                    for s in roster.slots
                ),
                bench=tuple(_with_calendar(p) for p in roster.bench),
                ir=tuple(_with_calendar(p) for p in roster.ir),
            )
        }
    )


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
        self._teams = [_team_with_calendar(Team.model_validate(t)) for t in data["teams"]]
        self._current_week = int(data["current_week"])
        self._matchups = {
            int(week): [Matchup.model_validate(m) for m in matchups]
            for week, matchups in data["matchups"].items()
        }
        self._free_agents = {
            int(week): [_with_calendar(Player.model_validate(p)) for p in players]
            for week, players in data["free_agents"].items()
        }
        self._history = [
            PlayerWeek.model_validate(row) for row in _load_insights_fixture()["history"]
        ]

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

    async def get_player_history(self) -> list[PlayerWeek]:
        return list(self._history)


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

    async def get_alt_projections(
        self, week: int, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, float]:
        week_alt: dict[str, float] = _load_insights_fixture()["alt_projections"].get(str(week), {})
        return {
            player.player_id: week_alt[str(player.player_id)]
            for player in players
            if str(player.player_id) in week_alt
        }


class DemoUsageProvider:
    """``UsageProvider`` backed by the committed fixture. Points in the fixture are already
    PPR, the demo league's scoring, so ``reception_points`` is not reapplied."""

    async def get_usage(self, *, reception_points: float = 1.0) -> list[UsageWeek]:
        return [UsageWeek.model_validate(row) for row in _load_insights_fixture()["usage"]]


class DemoOddsProvider:
    """``OddsProvider`` backed by the committed fixture's invented lines."""

    async def get_game_lines(self) -> dict[str, GameLine]:
        lines: dict[str, GameLine] = {}
        for game in _load_insights_fixture()["game_lines"]:
            total, spread = float(game["total"]), float(game["home_spread"])
            for team, opponent, sign in (
                (game["home"], game["away"], 1.0),
                (game["away"], game["home"], -1.0),
            ):
                lines[team] = GameLine(
                    pro_team=team,
                    opponent=opponent,
                    implied_total=total / 2 - sign * spread / 2,
                    spread=sign * spread,
                    total=total,
                )
        return lines

"""Fixture-backed providers used when ``FFMCP_MODE=demo`` (docs/architecture.md §6).

Selected once, in ``lifespan``. No branching on mode inside a tool. Every class implements the
same protocol as its live counterpart structurally, so a tool adapter cannot tell the difference.

A fixture *set* is three files in ``tests/fixtures`` sharing a prefix (``FFMCP_FIXTURE``):
``{name}_league.json`` (settings, rosters, schedule, free agents), ``{name}_market.json``
(Sleeper-shaped market signal) and ``{name}_insights.json`` (history, usage, second-opinion
projections, game lines, and whatever else a tool reads beyond ESPN). Two sets are committed:

* ``demo``: a synthetic ten-team league, invented players and all, which the README and
  ``uv run poe demo`` are built from (``scripts/make_demo_fixture.py``);
* ``snapshot``: a real sixteen-team league at week 6 of 2026, anonymized
  (``scripts/record_fixtures.py``). It exists for the regression tests, because the synthetic
  league turned out to differ from live ESPN in exactly the ways that hid real bugs: byes
  projected ``0.0`` rather than absent, injured free agents whose season rate is zero, released
  players on no NFL team, a deep league's thin waiver wire.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
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
    Transaction,
    UsageWeek,
)

DEFAULT_FIXTURE = "demo"


def _fixtures_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "tests" / "fixtures"
    raise FileNotFoundError("could not locate tests/fixtures relative to the installed package")


@lru_cache(maxsize=8)
def _load(fixture: str, part: str) -> dict[str, Any]:
    path = _fixtures_dir() / f"{fixture}_{part}.json"
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _with_calendar(player: Player, insights: dict[str, Any]) -> Player:
    """Attach the fixture's bye week and, for a player on bye this week, his season rate: the
    two fields the synthetic league fixture predates. A recorded fixture carries both on the
    player already, and its insights file has neither table."""
    rate = insights.get("season_rates", {}).get(str(player.player_id))
    bye = insights.get("bye_weeks", {}).get(player.pro_team)
    return player.model_copy(
        update={
            "bye_week": bye if bye is not None else player.bye_week,
            "season_rate": rate if rate is not None else player.season_rate,
        }
    )


def _team_with_calendar(team: Team, insights: dict[str, Any]) -> Team:
    roster = team.roster
    return team.model_copy(
        update={
            "roster": Roster(
                slots=tuple(
                    s.model_copy(update={"player": _with_calendar(s.player, insights)})
                    if s.player
                    else s
                    for s in roster.slots
                ),
                bench=tuple(_with_calendar(p, insights) for p in roster.bench),
                ir=tuple(_with_calendar(p, insights) for p in roster.ir),
            )
        }
    )


class DemoLeagueProvider:
    """``LeagueProvider`` backed by a committed fixture league."""

    def __init__(self, fixture: str = DEFAULT_FIXTURE) -> None:
        data = _load(fixture, "league")
        insights = _load(fixture, "insights")
        self._settings = LeagueSettings.model_validate(data["settings"])
        self._teams = [_team_with_calendar(Team.model_validate(t), insights) for t in data["teams"]]
        self._current_week = int(data["current_week"])
        self._matchups = {
            int(week): [Matchup.model_validate(m) for m in matchups]
            for week, matchups in data["matchups"].items()
        }
        self._free_agents = {
            int(week): [_with_calendar(Player.model_validate(p), insights) for p in players]
            for week, players in data["free_agents"].items()
        }
        self._history = [PlayerWeek.model_validate(row) for row in insights["history"]]
        recorded = data.get("recorded_at")
        self.recorded_at = datetime.fromisoformat(recorded) if recorded else None
        """When the fixture was taken: demo mode's "now"."""
        self._activity = [Transaction.model_validate(row) for row in insights.get("activity", [])]

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

    async def get_recent_activity(self, *, size: int = 25) -> list[Transaction]:
        return self._activity[:size]


class DemoMarketProvider:
    """``MarketProvider`` backed by a committed fixture. Bypasses real identity resolution
    since the fixture already keys signals by ESPN player id."""

    def __init__(self, fixture: str = DEFAULT_FIXTURE) -> None:
        data = _load(fixture, "market")
        self._insights = _load(fixture, "insights")
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
        week_alt: dict[str, float] = self._insights["alt_projections"].get(str(week), {})
        return {
            player.player_id: week_alt[str(player.player_id)]
            for player in players
            if str(player.player_id) in week_alt
        }

    async def get_outlook(
        self, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, dict[int, float]]:
        outlook: dict[str, dict[str, float]] = self._insights.get("outlook", {})
        return {
            player.player_id: {
                int(week): points for week, points in outlook.get(str(player.player_id), {}).items()
            }
            for player in players
        }


class DemoUsageProvider:
    """``UsageProvider`` backed by a committed fixture. Points in the fixture are already in
    the league's own scoring, so ``reception_points`` is not reapplied."""

    def __init__(self, fixture: str = DEFAULT_FIXTURE) -> None:
        self._fixture = fixture

    async def get_usage(self, *, reception_points: float = 1.0) -> list[UsageWeek]:
        return [UsageWeek.model_validate(row) for row in _load(self._fixture, "insights")["usage"]]


class DemoOddsProvider:
    """``OddsProvider`` backed by a committed fixture's lines."""

    def __init__(self, fixture: str = DEFAULT_FIXTURE) -> None:
        self._fixture = fixture

    async def get_upcoming_lines(self) -> dict[str, list[GameLine]]:
        return {team: [line] for team, line in (await self.get_game_lines()).items()}

    async def get_game_lines(self) -> dict[str, GameLine]:
        lines: dict[str, GameLine] = {}
        for game in _load(self._fixture, "insights")["game_lines"]:
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


class DemoTradeValueProvider:
    """``TradeValueProvider`` backed by a committed fixture's market values."""

    def __init__(self, fixture: str = DEFAULT_FIXTURE) -> None:
        self._fixture = fixture

    async def get_trade_values(
        self, *, team_count: int, quarterbacks: int, reception_points: float
    ) -> dict[int, float]:
        values: dict[str, float] = _load(self._fixture, "insights").get("trade_values", {})
        return {int(espn_id): float(value) for espn_id, value in values.items()}

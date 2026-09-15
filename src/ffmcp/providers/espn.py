"""ESPN league adapter: wraps ``espn_api.football.League`` (synchronous) and maps its objects
to domain models. ``espn-api`` shifts with ESPN's private API, so every attribute this module
touches was verified against the installed ``espn-api==0.46.0`` by introspection, not memory.

One notable trap: ``BoxScore.home_team``/``away_team`` are team **ids** (``int``) for a
completed week but full ``Team`` instances for the current/in-progress week (``None`` either
way for a bye). ``_box_score_team_id`` normalizes both.

``espn-api`` is synchronous and does its own network I/O with ``requests``, so every call into
it runs under ``asyncio.to_thread`` so it never blocks the event loop (docs/architecture.md §7).
The ``League`` handle itself is expensive to build (it fetches the whole league on
construction) and is rebuilt lazily, at most every ``_LEAGUE_TTL_SECONDS``, rather than at
``lifespan`` startup. Startup must never touch the network (docs/architecture.md §3).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import requests
from espn_api.football import League
from espn_api.football import Player as EspnPlayer
from espn_api.football import Team as EspnTeam
from espn_api.football.box_score import BoxScore
from espn_api.football.settings import Settings as EspnSettings
from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague, ESPNUnknownError

from ffmcp.domain.models import (
    LeagueSettings,
    Matchup,
    Player,
    Projection,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.errors import CredentialsMissing, LeagueNotAccessible, UpstreamUnavailable

_LEAGUE_TTL_SECONDS = 600  # rosters change on waivers/trades. See docs/architecture.md §3.
_BENCH_SLOT = "BE"
_IR_SLOT = "IR"

_UpstreamErrors = (ESPNUnknownError, requests.RequestException)


class EspnLeagueProvider:
    """Live ``LeagueProvider`` backed by a real ESPN league."""

    def __init__(
        self,
        *,
        league_id: int,
        season: int | None,
        espn_s2: str | None,
        swid: str | None,
        resolve_season: Callable[[], Awaitable[int]] | None = None,
    ) -> None:
        """``season=None`` defers to ``resolve_season`` (FFMCP_SEASON's "current" default,
        docs/architecture.md §5) the first time a league handle is actually needed, rather
        than at construction. Startup must never touch the network (docs/architecture.md §3).
        """
        self._league_id = league_id
        self._season = season
        self._resolve_season = resolve_season
        self._espn_s2 = espn_s2
        self._swid = swid
        self._league: League | None = None
        self._league_built_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_current_week(self) -> int:
        league = await self._handle()
        return int(league.current_week)

    async def get_settings(self) -> LeagueSettings:
        league = await self._handle()
        assert self._season is not None  # resolved by _handle() before it returns
        return _map_settings(self._league_id, self._season, league.settings)

    async def get_teams(self) -> list[Team]:
        league = await self._handle()
        week = int(league.current_week)
        return [_map_team(t, week=week) for t in league.teams]

    async def get_matchups(self, week: int) -> list[Matchup]:
        league = await self._handle()
        try:
            box_scores: list[BoxScore] = await asyncio.to_thread(league.box_scores, week)
        except _UpstreamErrors as exc:
            raise UpstreamUnavailable("ESPN") from exc
        return [_map_matchup(week, bs) for bs in box_scores]

    async def get_free_agents(
        self, week: int, *, size: int = 50, position: str | None = None
    ) -> list[Player]:
        league = await self._handle()
        try:
            players: list[EspnPlayer] = await asyncio.to_thread(
                league.free_agents, week, size, position
            )
        except _UpstreamErrors as exc:
            raise UpstreamUnavailable("ESPN") from exc
        return [_map_player(p, week=week) for p in players]

    async def _handle(self) -> League:
        async with self._lock:
            if self._season is None:
                assert self._resolve_season is not None, (
                    "no season configured and no way to resolve one"
                )
                self._season = await self._resolve_season()

            now = time.monotonic()
            stale = self._league is None or (now - self._league_built_at) > _LEAGUE_TTL_SECONDS
            if stale:
                self._league = await asyncio.to_thread(self._build_league)
                self._league_built_at = now
            assert self._league is not None
            return self._league

    def _build_league(self) -> League:
        try:
            return League(
                league_id=self._league_id,
                year=self._season,
                espn_s2=self._espn_s2,
                swid=self._swid,
            )
        except ESPNAccessDenied as exc:
            if not (self._espn_s2 and self._swid):
                raise CredentialsMissing() from exc
            raise LeagueNotAccessible("credentials were rejected or have expired") from exc
        except ESPNInvalidLeague as exc:
            raise LeagueNotAccessible(
                f"league {self._league_id} was not found for season {self._season}"
            ) from exc
        except _UpstreamErrors as exc:
            raise UpstreamUnavailable("ESPN") from exc


def _map_settings(league_id: int, season: int, settings: EspnSettings) -> LeagueSettings:
    return LeagueSettings(
        league_id=league_id,
        season=season,
        team_count=int(settings.team_count),
        playoff_team_count=int(settings.playoff_team_count),
        reg_season_weeks=int(settings.reg_season_count),
        slot_counts=dict(settings.position_slot_counts),
        scoring_type=settings.scoring_type,
    )


def _map_player(player: EspnPlayer, *, week: int) -> Player:
    week_stats = player.stats.get(week, {})
    proj_points = week_stats.get("projected_points")
    projection = (
        Projection(week=week, points=float(proj_points)) if proj_points is not None else None
    )
    live_points = week_stats.get("points")
    return Player(
        player_id=int(player.playerId),
        name=str(player.name),
        position=str(player.position),
        eligible_slots=tuple(str(s) for s in player.eligibleSlots),
        pro_team=str(player.proTeam),
        injured=bool(player.injured),
        injury_status=player.injuryStatus or None,
        projection=projection,
        live_points=float(live_points) if live_points is not None else None,
    )


def _map_team(team: EspnTeam, *, week: int) -> Team:
    starters = []
    bench = []
    ir = []
    for espn_player in team.roster:
        mapped = _map_player(espn_player, week=week)
        slot = espn_player.lineupSlot
        if slot == _BENCH_SLOT:
            bench.append(mapped)
        elif slot == _IR_SLOT:
            ir.append(mapped)
        else:
            starters.append(RosterSlot(slot=slot, player=mapped))

    return Team(
        team_id=int(team.team_id),
        name=str(team.team_name),
        abbrev=str(team.team_abbrev),
        wins=int(team.wins),
        losses=int(team.losses),
        ties=int(getattr(team, "ties", 0)),
        points_for=float(team.points_for),
        points_against=float(team.points_against),
        standing=int(team.standing),
        playoff_pct=float(team.playoff_pct),
        roster=Roster(slots=tuple(starters), bench=tuple(bench), ir=tuple(ir)),
    )


def _box_score_team_id(value: EspnTeam | int | None) -> int | None:
    """``BoxScore.home_team``/``away_team`` are ``int`` team ids for a completed week but full
    ``Team`` objects for the current/in-progress week in the installed ``espn-api==0.46.0``.
    ``None`` either way for a bye."""
    if value is None:
        return None
    if isinstance(value, EspnTeam):
        return int(value.team_id)
    return int(value)


def _map_matchup(week: int, box_score: BoxScore) -> Matchup:
    return Matchup(
        week=week,
        home_team_id=_box_score_team_id(box_score.home_team),
        away_team_id=_box_score_team_id(box_score.away_team),
        home_score=float(box_score.home_score),
        away_score=float(box_score.away_score),
        home_projected=float(box_score.home_projected),
        away_projected=float(box_score.away_projected),
        is_playoff=bool(box_score.is_playoff),
    )

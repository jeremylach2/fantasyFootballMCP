"""ESPN league adapter: wraps ``espn_api.football.League`` (synchronous) and maps its objects
to domain models. ``espn-api`` shifts with ESPN's private API, so every attribute this module
touches was verified against the installed ``espn-api==0.46.0`` by introspection, not memory.

One notable trap: ``BoxScore.home_team``/``away_team`` are team **ids** (``int``) for a
completed week but full ``Team`` instances for the current/in-progress week (``None`` either
way for a bye). ``_box_score_team_id`` normalizes both.

A second, sharper trap: ``League.box_scores(week)`` only honors ``week`` when
``week <= league.current_week``; internally it resolves a "matchup period" for the requested
week solely inside that branch, and otherwise silently falls back to the *current* matchup
period. Calling it for a future week does not error and does not raise — it returns this
week's box scores relabeled with the future week number, which would hand every remaining-season
tool (``simulate_season``, ``find_trades``, the ``SOS`` column) a schedule where every team
plays this week's opponent every week for the rest of the season. ``get_matchups`` below
therefore only uses ``box_scores`` for ``week <= current_week`` and falls back to
``League.scoreboard(week)``, which filters on ``matchupPeriodId`` unconditionally, for any
future week.

``espn-api`` is synchronous and does its own network I/O with ``requests``, so every call into
it runs under ``asyncio.to_thread`` so it never blocks the event loop (docs/architecture.md §7).
The ``League`` handle itself is expensive to build (it fetches the whole league on
construction) and is rebuilt lazily, at most every ``_LEAGUE_TTL_SECONDS``, rather than at
``lifespan`` startup. Startup must never touch the network (docs/architecture.md §3).

A third trap, for bye weeks: ``Player.schedule`` (the player's NFL schedule, from which a bye is
the one missing week) is populated for rostered players but comes back *empty* for free agents.
Byes are a property of the NFL team, not the player, so they are derived once from the rostered
players (every NFL team has someone rostered in any real league) and applied by team to
everyone, free agents included.

Completed weeks' box scores (``get_player_history``) go through the shared disk ``Cache``:
a finished week never changes after its stat corrections settle, so it is fetched once, not on
every tool call that wants the season's history.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

import requests
from espn_api.football import League
from espn_api.football import Player as EspnPlayer
from espn_api.football import Team as EspnTeam
from espn_api.football.box_score import BoxScore
from espn_api.football.matchup import Matchup as EspnScheduledMatchup
from espn_api.football.settings import Settings as EspnSettings
from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague, ESPNUnknownError

from ffmcp.domain.models import (
    LeagueSettings,
    MarketSignal,
    Matchup,
    Player,
    PlayerWeek,
    Projection,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.errors import CredentialsMissing, LeagueNotAccessible, UpstreamUnavailable
from ffmcp.providers.cache import Cache

_LEAGUE_TTL_SECONDS = 600  # rosters change on waivers/trades. See docs/architecture.md §3.
_BENCH_SLOT = "BE"
_IR_SLOT = "IR"

_UpstreamErrors = (ESPNUnknownError, requests.RequestException)

_HEALTHY_STATUSES = frozenset({"ACTIVE", "NORMAL"})
"""ESPN's words for "not injured". Translated to ``None`` here, at the boundary, so no layer
above ever mistakes a healthy player's status for an injury designation (they used to surface
as "Active: <player> — check inactives" caveats, one per healthy starter)."""

_NFL_WEEKS = range(1, 19)
_RECEPTION_STAT_ID = 53
"""ESPN's scoring-item id for a reception: its ``points`` is the league's PPR setting."""

HISTORY_TTL_SECONDS = 7 * 86_400
RECENT_HISTORY_TTL_SECONDS = 6 * 3_600
"""The week that just finished still takes stat corrections for a few days, so it is re-read
every few hours; anything older is effectively permanent."""


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
        cache: Cache | None = None,
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
        self._cache = cache
        self._bye_weeks: dict[str, int] = {}

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
        return [_map_team(t, week=week, byes=self._bye_weeks) for t in league.teams]

    async def get_matchups(self, week: int) -> list[Matchup]:
        league = await self._handle()
        try:
            if week <= int(league.current_week):
                box_scores: list[BoxScore] = await asyncio.to_thread(league.box_scores, week)
                return [_map_matchup(week, bs) for bs in box_scores]
            # See the module docstring: box_scores() cannot be trusted for a future week.
            scheduled: list[EspnScheduledMatchup] = await asyncio.to_thread(league.scoreboard, week)
        except _UpstreamErrors as exc:
            raise UpstreamUnavailable("ESPN") from exc
        return [_map_scheduled_matchup(week, m) for m in scheduled]

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
        return [_map_player(p, week=week, byes=self._bye_weeks) for p in players]

    async def get_player_history(self) -> list[PlayerWeek]:
        """Every rostered player-week of the season's completed weeks, starters and bench.

        One ``box_scores`` call per week, concurrently, each cached (see the module docstring).
        """
        league = await self._handle()
        current = int(league.current_week)
        assert self._season is not None
        season = self._season

        async def one_week(week: int) -> list[dict[str, Any]]:
            async def fetch() -> list[dict[str, Any]]:
                try:
                    box_scores: list[BoxScore] = await asyncio.to_thread(league.box_scores, week)
                except _UpstreamErrors as exc:
                    raise UpstreamUnavailable("ESPN") from exc
                return [row for bs in box_scores for row in _box_score_rows(bs)]

            if self._cache is None:
                return await fetch()
            ttl = RECENT_HISTORY_TTL_SECONDS if week == current - 1 else HISTORY_TTL_SECONDS
            key = f"v1:espn:history:{self._league_id}:{season}:{week}"
            return await self._cache.get_or_fetch(key, ttl, fetch)

        weeks = range(1, current)
        per_week = await asyncio.gather(*(one_week(week) for week in weeks))
        return [
            PlayerWeek(season=season, week=week, **row)
            for week, rows in zip(weeks, per_week, strict=True)
            for row in rows
        ]

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
                self._bye_weeks = _bye_weeks_by_team(self._league)
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
        reception_points=_reception_points(settings),
        playoff_round_weeks=int(getattr(settings, "playoff_matchup_period_length", 1) or 1),
    )


def _reception_points(settings: EspnSettings) -> float:
    """Points per reception, read off the league's raw scoring items. 0.0 when the league
    has no reception item at all, which is how ESPN represents standard scoring."""
    raw = getattr(settings, "_raw_scoring_settings", None) or {}
    items = raw.get("scoringItems")
    if not items:
        return 1.0  # no scoring data at all: assume ESPN's default, PPR
    for item in items:
        if item.get("statId") == _RECEPTION_STAT_ID:
            return float(item.get("points", 0.0))
    return 0.0


def _bye_weeks_by_team(league: League) -> dict[str, int]:
    """``{pro_team: bye week}``, from the NFL schedules of rostered players (see the module
    docstring for why free agents cannot supply their own). A team's bye is the one week of
    the NFL season its schedule skips; the most common answer across its players wins, so one
    malformed schedule cannot mislabel a whole team."""
    votes: dict[str, Counter[int]] = {}
    for team in league.teams:
        for player in team.roster:
            schedule = getattr(player, "schedule", None) or {}
            played = {int(week) for week in schedule}
            missing = [week for week in _NFL_WEEKS if week not in played]
            if schedule and len(missing) == 1:
                votes.setdefault(str(player.proTeam), Counter())[missing[0]] += 1
    return {team: counter.most_common(1)[0][0] for team, counter in votes.items()}


def _map_market_signal(player: EspnPlayer) -> MarketSignal | None:
    """ESPN's own ownership stats for *this* league, not Sleeper's cross-league signal.

    ``espn-api`` reports a field it has no data for as ``-1`` (see the installed package's
    ``Player.__init__``, which rounds ``.get(..., -1)``) rather than omitting it, so ``-1`` is
    treated as "no reading" here rather than surfaced as a real percentage. ``None`` is
    returned rather than an empty ``MarketSignal`` when neither field has a reading, so
    ``player.market`` stays ``None`` until something actually attaches to it.
    """
    owned = getattr(player, "percent_owned", None)
    started = getattr(player, "percent_started", None)
    owned = float(owned) if owned is not None and owned >= 0 else None
    started = float(started) if started is not None and started >= 0 else None
    if owned is None and started is None:
        return None
    return MarketSignal(percent_owned=owned, percent_started=started)


def _map_player(player: EspnPlayer, *, week: int, byes: dict[str, int] | None = None) -> Player:
    week_stats = player.stats.get(week, {})
    proj_points = week_stats.get("projected_points")
    projection = (
        Projection(week=week, points=float(proj_points)) if proj_points is not None else None
    )
    live_points = week_stats.get("points")
    season_rate = getattr(player, "projected_avg_points", None)
    return Player(
        player_id=int(player.playerId),
        name=str(player.name),
        position=str(player.position),
        eligible_slots=tuple(str(s) for s in player.eligibleSlots),
        pro_team=str(player.proTeam),
        injured=bool(player.injured),
        injury_status=_injury_status(player.injuryStatus),
        projection=projection,
        market=_map_market_signal(player),
        live_points=float(live_points) if live_points is not None else None,
        bye_week=(byes or {}).get(str(player.proTeam)),
        season_rate=float(season_rate) if season_rate else None,
    )


def _injury_status(status: str | None) -> str | None:
    if not status or status.upper() in _HEALTHY_STATUSES:
        return None
    return status


def _map_team(team: EspnTeam, *, week: int, byes: dict[str, int] | None = None) -> Team:
    starters = []
    bench = []
    ir = []
    for espn_player in team.roster:
        mapped = _map_player(espn_player, week=week, byes=byes)
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


def _map_scheduled_matchup(week: int, matchup: EspnScheduledMatchup) -> Matchup:
    """A not-yet-played week's pairing, from ``League.scoreboard(week)`` rather than
    ``box_scores`` (see the module docstring for why). ``home_team``/``away_team`` are set by
    ``scoreboard()`` only when a team actually matched (absent, not ``None``, for a bye), so
    both reads go through ``getattr``.

    Projected totals are 0.0: ``scoreboard()``'s ``Matchup`` carries no box-score-derived
    projection, and nothing downstream needs one for a week that has not been played yet
    (``domain.simulate`` draws its own scoring distribution from each roster's optimal lineup,
    not from ESPN's per-matchup projection).
    """
    home_team = getattr(matchup, "home_team", None)
    away_team = getattr(matchup, "away_team", None)
    return Matchup(
        week=week,
        home_team_id=int(home_team.team_id) if home_team is not None else None,
        away_team_id=int(away_team.team_id) if away_team is not None else None,
        home_score=float(matchup.home_score),
        away_score=float(matchup.away_score),
        home_projected=0.0,
        away_projected=0.0,
        is_playoff=bool(matchup.is_playoff),
    )


def _box_score_rows(box_score: BoxScore) -> list[dict[str, Any]]:
    """Both lineups of one completed box score as plain, cacheable ``PlayerWeek`` fields
    (everything but ``season``/``week``, which the caller knows)."""
    rows = []
    for side in ("home", "away"):
        team_id = _box_score_team_id(getattr(box_score, f"{side}_team"))
        for player in getattr(box_score, f"{side}_lineup"):
            projected = getattr(player, "projected_points", None)
            rows.append(
                {
                    "player_id": int(player.playerId),
                    "name": str(player.name),
                    "position": str(player.position),
                    "pro_team": str(player.proTeam),
                    "eligible_slots": [str(slot) for slot in player.eligibleSlots],
                    "fantasy_team_id": team_id,
                    "slot": str(player.slot_position),
                    "projected": float(projected) if projected is not None else None,
                    "actual": float(player.points or 0.0),
                    "played": bool(player.game_played) and not bool(player.on_bye_week),
                }
            )
    return rows

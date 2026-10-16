"""Sleeper market-signal adapter. No auth. ``httpx2`` only, since plain ``httpx`` is not
installed and ``respx`` does not intercept it.

The full player dump is ~14.6 MB and is fetched at most daily, then immediately reduced to a
narrow ``id -> {name, pos, team}`` index. The raw parsed document is never retained past the
reduction (docs/architecture.md §3).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx2

from ffmcp.domain.models import MarketSignal, Player
from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache
from ffmcp.providers.identity import IdentityIndex

BASE_URL = "https://api.sleeper.app/v1"
PROJECTIONS_URL = "https://api.sleeper.app/projections/nfl"
"""Not under ``/v1``: Sleeper serves projections from an older, undocumented path. It is what
their own app reads, and it has been stable for years, but it carries no compatibility promise,
so every caller treats a failure here as "no second opinion", never as an error."""

STATE_TTL = 3_600
PLAYERS_TTL = 6 * 3_600
"""The player dump is ~14.6 MB, but it is also where injury detail lives, and that changes
through the week. Four fetches a day keep a Friday injury report from waiting until Saturday."""
TRENDING_TTL = 900
PROJECTIONS_TTL = 3_600
PAST_PROJECTIONS_TTL = 7 * 86_400
OUTLOOK_TTL = 6 * 3_600
"""A player's week-by-week outlook moves with injury news, which lands a few times a week."""
OUTLOOK_URL = "https://api.sleeper.com/projections/nfl/player"
"""Per-player, every week of the season in one small response (~2 KB). The weekly endpoint
above is ~2 MB per week, so it is the wrong tool for ten weeks of a handful of players. Like
``PROJECTIONS_URL``, undocumented: a failure means "no outlook", never an error."""

_PROJECTED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
"""Requested explicitly: unfiltered, the payload is 5.7 MB, mostly defensive players no
fantasy lineup here can start. Filtered, it is about 2 MB."""


def _injury_detail(info: dict[str, Any]) -> str:
    """``"Knee - ACL, surgery"``, or ``""`` for a healthy player. Sleeper also has a
    practice-participation field, but in a live check it was filled for one player of 11,000,
    so it is not kept."""
    part = info.get("injury_body_part") or ""
    if not part or not info.get("injury_status"):
        return ""
    notes = info.get("injury_notes")
    return f"{part}, {notes.lower()}" if notes else str(part)


class SleeperMarketProvider:
    """Live ``MarketProvider`` backed by the public Sleeper API."""

    def __init__(self, client: httpx2.AsyncClient, cache: Cache) -> None:
        self._client = client
        self._cache = cache
        self._identity: IdentityIndex | None = None
        self._index: dict[str, dict[str, str]] = {}

    async def get_current_season(self) -> int:
        state = await self._cache.get_or_fetch("v1:sleeper:state", STATE_TTL, self._fetch_state)
        return int(state["season"])

    async def get_current_week(self) -> int:
        state = await self._cache.get_or_fetch("v1:sleeper:state", STATE_TTL, self._fetch_state)
        return int(state["week"])

    async def get_trending_adds(
        self, *, lookback_hours: int = 24, limit: int = 25
    ) -> dict[str, int]:
        key = f"v1:sleeper:trending:add:{lookback_hours}:{limit}"

        async def fetch() -> dict[str, int]:
            rows = await self._get_json(
                f"/players/nfl/trending/add?lookback_hours={lookback_hours}&limit={limit}"
            )
            return {row["player_id"]: int(row["count"]) for row in rows}

        return await self._cache.get_or_fetch(key, TRENDING_TTL, fetch)

    async def get_market_signal(
        self, *, espn_player_id: int, name: str, position: str, pro_team: str
    ) -> MarketSignal | None:
        identity = await self._get_identity()
        sleeper_id = identity.resolve(espn_player_id, name, position, pro_team)
        if sleeper_id is None:
            return None
        trending = await self.get_trending_adds()
        injury = self._index.get(sleeper_id, {}).get("injury") or None
        return MarketSignal(trending_adds=trending.get(sleeper_id), injury_detail=injury)

    async def get_alt_projections(
        self, week: int, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, float]:
        """Sleeper's projection for each of ``players`` it can identify, keyed by ESPN id.

        Sleeper publishes standard and PPR points; a league's own reception scoring is an exact
        linear interpolation between the two (they differ only by receptions).
        """
        season = await self.get_current_season()
        current = await self.get_current_week()
        ttl = PAST_PROJECTIONS_TTL if week < current else PROJECTIONS_TTL
        key = f"v1:sleeper:projections:{season}:{week}"

        async def fetch() -> dict[str, list[float]]:
            positions = "&".join(f"position[]={position}" for position in _PROJECTED_POSITIONS)
            rows = await self._get_json_url(
                f"{PROJECTIONS_URL}/{season}/{week}?season_type=regular&{positions}"
            )
            reduced: dict[str, list[float]] = {}
            for row in rows:
                stats = row.get("stats") or {}
                standard, ppr = stats.get("pts_std"), stats.get("pts_ppr")
                if standard is not None and ppr is not None:
                    reduced[str(row["player_id"])] = [float(standard), float(ppr)]
            return reduced

        by_sleeper_id = await self._cache.get_or_fetch(key, ttl, fetch)
        identity = await self._get_identity()
        result: dict[int, float] = {}
        for player in players:
            sleeper_id = identity.resolve(
                player.player_id, player.name, player.position, player.pro_team
            )
            points = by_sleeper_id.get(sleeper_id) if sleeper_id is not None else None
            if points is not None:
                standard, ppr = points
                result[player.player_id] = standard + reception_points * (ppr - standard)
        return result

    async def get_outlook(
        self, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, dict[int, float]]:
        """Each player's projection for every remaining week, from Sleeper's per-player view.

        This is what an injured player's value needs and ESPN cannot give: ESPN's season rate
        for a player on injured reserve is zero, while Sleeper projects zeros until his
        expected return and real numbers after it. A week Sleeper has no entry for (his bye, or
        no projection at all) is left out rather than read as zero.
        """
        season = await self.get_current_season()
        current = await self.get_current_week()
        identity = await self._get_identity()

        async def one(player: Player) -> tuple[int, dict[int, float]]:
            sleeper_id = identity.resolve(
                player.player_id, player.name, player.position, player.pro_team
            )
            if sleeper_id is None:
                return player.player_id, {}

            async def fetch() -> dict[str, list[float]]:
                data = await self._get_json_url(
                    f"{OUTLOOK_URL}/{sleeper_id}?season_type=regular&season={season}&grouping=week"
                )
                reduced: dict[str, list[float]] = {}
                for week, entry in (data or {}).items():
                    stats = (entry or {}).get("stats") or {}
                    standard, ppr = stats.get("pts_std"), stats.get("pts_ppr")
                    if standard is not None and ppr is not None:
                        reduced[str(week)] = [float(standard), float(ppr)]
                return reduced

            key = f"v1:sleeper:outlook:{season}:{sleeper_id}"
            weeks = await self._cache.get_or_fetch(key, OUTLOOK_TTL, fetch)
            return player.player_id, {
                int(week): standard + reception_points * (ppr - standard)
                for week, (standard, ppr) in weeks.items()
                if int(week) > current
            }

        return dict(await asyncio.gather(*(one(player) for player in players)))

    async def _get_identity(self) -> IdentityIndex:
        index = await self._cache.get_or_fetch(
            "v2:sleeper:players", PLAYERS_TTL, self._fetch_player_index
        )
        if self._identity is None or index is not self._index:
            self._index = index
            self._identity = IdentityIndex.build(index)
        return self._identity

    async def _fetch_state(self) -> dict[str, Any]:
        data = await self._get_json("/state/nfl")
        return {"season": data.get("season"), "week": data.get("week")}

    async def _fetch_player_index(self) -> dict[str, dict[str, str]]:
        raw = await self._get_json("/players/nfl")
        return {
            player_id: {
                "name": info.get("full_name")
                or f"{info.get('first_name', '')} {info.get('last_name', '')}".strip(),
                "pos": info.get("position") or "",
                "team": info.get("team") or "",
                "injury": _injury_detail(info),
            }
            for player_id, info in raw.items()
            if info.get("position")
        }

    async def _get_json(self, path: str) -> Any:
        return await self._get_json_url(f"{BASE_URL}{path}")

    async def _get_json_url(self, url: str) -> Any:
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise UpstreamUnavailable("Sleeper") from exc
        return response.json()

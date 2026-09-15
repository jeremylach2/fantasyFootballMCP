"""Sleeper market-signal adapter. No auth. ``httpx2`` only, since plain ``httpx`` is not
installed and ``respx`` does not intercept it.

The full player dump is ~14.6 MB and is fetched at most daily, then immediately reduced to a
narrow ``id -> {name, pos, team}`` index. The raw parsed document is never retained past the
reduction (docs/architecture.md §3).
"""

from __future__ import annotations

from typing import Any

import httpx2

from ffmcp.domain.models import MarketSignal
from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache
from ffmcp.providers.identity import IdentityIndex

BASE_URL = "https://api.sleeper.app/v1"

STATE_TTL = 3_600
PLAYERS_TTL = 86_400
TRENDING_TTL = 900


class SleeperMarketProvider:
    """Live ``MarketProvider`` backed by the public Sleeper API."""

    def __init__(self, client: httpx2.AsyncClient, cache: Cache) -> None:
        self._client = client
        self._cache = cache
        self._identity: IdentityIndex | None = None

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
        return MarketSignal(trending_adds=trending.get(sleeper_id))

    async def _get_identity(self) -> IdentityIndex:
        if self._identity is None:
            index = await self._cache.get_or_fetch(
                "v1:sleeper:players", PLAYERS_TTL, self._fetch_player_index
            )
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
            }
            for player_id, info in raw.items()
            if info.get("position")
        }

    async def _get_json(self, path: str) -> Any:
        try:
            response = await self._client.get(f"{BASE_URL}{path}")
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise UpstreamUnavailable("Sleeper") from exc
        return response.json()

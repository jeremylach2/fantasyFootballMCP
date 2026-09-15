"""Structural interfaces that ``providers/espn.py`` and ``providers/sleeper.py`` implement,
and that ``providers/demo.py`` swaps in behind unchanged when ``FFMCP_MODE=demo``
(docs/architecture.md §6). No branching on mode belongs anywhere outside ``lifespan``.
"""

from __future__ import annotations

from typing import Protocol

from ffmcp.domain.models import LeagueSettings, MarketSignal, Matchup, Player, Team


class LeagueProvider(Protocol):
    async def get_current_week(self) -> int: ...

    async def get_settings(self) -> LeagueSettings: ...

    async def get_teams(self) -> list[Team]: ...

    async def get_matchups(self, week: int) -> list[Matchup]: ...

    async def get_free_agents(
        self, week: int, *, size: int = 50, position: str | None = None
    ) -> list[Player]: ...


class MarketProvider(Protocol):
    async def get_current_season(self) -> int: ...

    async def get_current_week(self) -> int: ...

    async def get_trending_adds(
        self, *, lookback_hours: int = 24, limit: int = 25
    ) -> dict[str, int]: ...

    async def get_market_signal(
        self, *, espn_player_id: int, name: str, position: str, pro_team: str
    ) -> MarketSignal | None: ...

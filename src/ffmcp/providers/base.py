"""Structural interfaces that ``providers/espn.py``, ``providers/sleeper.py``,
``providers/nflverse.py`` and ``providers/odds.py`` implement, and that ``providers/demo.py``
swaps in behind unchanged when ``FFMCP_MODE=demo`` (docs/architecture.md §6). No branching on
mode belongs anywhere outside ``lifespan``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ffmcp.domain.models import (
    GameLine,
    LeagueSettings,
    MarketSignal,
    Matchup,
    Player,
    PlayerWeek,
    Team,
    UsageWeek,
)


class LeagueProvider(Protocol):
    async def get_current_week(self) -> int: ...

    async def get_settings(self) -> LeagueSettings: ...

    async def get_teams(self) -> list[Team]: ...

    async def get_matchups(self, week: int) -> list[Matchup]: ...

    async def get_free_agents(
        self, week: int, *, size: int = 50, position: str | None = None
    ) -> list[Player]: ...

    async def get_player_history(self) -> list[PlayerWeek]:
        """Every rostered player-week of this season's completed weeks."""
        ...


class MarketProvider(Protocol):
    async def get_current_season(self) -> int: ...

    async def get_current_week(self) -> int: ...

    async def get_trending_adds(
        self, *, lookback_hours: int = 24, limit: int = 25
    ) -> dict[str, int]: ...

    async def get_market_signal(
        self, *, espn_player_id: int, name: str, position: str, pro_team: str
    ) -> MarketSignal | None: ...

    async def get_alt_projections(
        self, week: int, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, float]:
        """A second, independent projection for each of ``players`` that the source covers,
        keyed by ESPN player id."""
        ...


class UsageProvider(Protocol):
    async def get_usage(self, *, reception_points: float = 1.0) -> list[UsageWeek]:
        """Every offensive player-game of the current season, keyed to ESPN player ids."""
        ...


class OddsProvider(Protocol):
    async def get_game_lines(self) -> dict[str, GameLine]:
        """Each NFL team's next game line, keyed by ESPN pro-team abbreviation. Empty when the
        source is not configured."""
        ...

"""Structural interfaces that ``providers/espn.py``, ``providers/sleeper.py``,
``providers/nflverse.py``, ``providers/odds.py`` and ``providers/fantasycalc.py`` implement, and
that ``providers/demo.py`` swaps in behind unchanged when ``FFMCP_MODE=demo``
(docs/architecture.md §6). No branching on mode belongs anywhere outside ``lifespan``.
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
    Transaction,
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

    async def get_recent_activity(self, *, size: int = 25) -> list[Transaction]:
        """The league's latest transactions, newest first."""
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

    async def get_outlook(
        self, players: Sequence[Player], *, reception_points: float = 1.0
    ) -> dict[int, dict[int, float]]:
        """Each player's projection for every remaining week the source covers, as
        ``{espn_id: {week: points}}``. A week with no game (a bye) is absent; a week he is
        expected to miss is ``0.0``. Meant for a handful of players, not a whole league."""
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

    async def get_upcoming_lines(self) -> dict[str, list[GameLine]]:
        """Every posted game line for each team, soonest first."""
        ...


class TradeValueProvider(Protocol):
    async def get_trade_values(
        self, *, team_count: int, quarterbacks: int, reception_points: float
    ) -> dict[int, float]:
        """What the trade market thinks each player is worth in a league of this shape, keyed
        by ESPN player id, on the source's own arbitrary scale. Players it does not list (most
        kickers and defenses, deep bench) are absent."""
        ...

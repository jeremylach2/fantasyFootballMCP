"""The Odds API adapter: each NFL team's next game line, as an implied team total.

A sportsbook's spread and total together say how many points the market expects each team to
score, which is the best single public read on a game's scoring environment:

    implied = total / 2 - spread / 2        (spread negative for the favourite)

so a 5.5-point favourite in a 43.5-point game is implied for 24.5. Lines are the median across
the configured books, which is steadier than any one book.

Budget: the free plan is 500 requests a month and one call here costs 2 (two markets, one
region). ``LINES_TTL`` of 12 hours holds a single server to about 120 a month, leaving room for
a second instance or a local copy. Without a key, ``get_game_lines`` returns ``{}`` and the
rest of the server carries on without the context; this is enrichment, never a dependency.

The key travels as a query parameter (the API accepts nothing else), which is why
``errors.redact`` scrubs ``apiKey=`` from every message.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import httpx2

from ffmcp.domain.models import GameLine
from ffmcp.errors import UpstreamUnavailable, redact
from ffmcp.providers.cache import Cache

logger = logging.getLogger("ffmcp.odds")

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
LINES_TTL = 12 * 3_600
_BOOKMAKERS = "draftkings,fanduel,betmgm,caesars"

TEAM_ABBREVIATIONS: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH",
}
"""The Odds API's full team names -> ESPN's pro-team abbreviations."""


class OddsApiProvider:
    """Live ``OddsProvider``. Constructed with ``api_key=None`` it is a silent no-op."""

    def __init__(self, client: httpx2.AsyncClient, cache: Cache, api_key: str | None) -> None:
        self._client = client
        self._cache = cache
        self._api_key = api_key

    async def get_game_lines(self) -> dict[str, GameLine]:
        if not self._api_key:
            return {}
        events = await self._cache.get_or_fetch("v1:odds:nfl", LINES_TTL, self._fetch)
        return game_lines_from_events(events)

    async def _fetch(self) -> list[dict[str, Any]]:
        params = {
            "apiKey": self._api_key or "",
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
            "bookmakers": _BOOKMAKERS,
        }
        try:
            response = await self._client.get(ODDS_URL, params=params)
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            logger.warning("odds fetch failed: %s", redact(str(exc)))
            raise UpstreamUnavailable("The Odds API") from None
        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info("The Odds API requests remaining this month: %s", remaining)
        data: list[dict[str, Any]] = response.json()
        return data


def game_lines_from_events(events: list[dict[str, Any]]) -> dict[str, GameLine]:
    """Each team's *earliest* listed game, as a ``GameLine``. Pure, so it is tested without the
    network against a recorded payload."""
    lines: dict[str, GameLine] = {}
    for event in sorted(events, key=lambda e: e.get("commence_time", "")):
        home = TEAM_ABBREVIATIONS.get(event.get("home_team", ""))
        away = TEAM_ABBREVIATIONS.get(event.get("away_team", ""))
        if home is None or away is None or home in lines or away in lines:
            continue
        spreads: list[float] = []
        totals: list[float] = []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    point = outcome.get("point")
                    if point is None:
                        continue
                    if market.get("key") == "spreads" and outcome.get("name") == event["home_team"]:
                        spreads.append(float(point))
                    elif market.get("key") == "totals" and outcome.get("name") == "Over":
                        totals.append(float(point))
        if not spreads or not totals:
            continue
        home_spread = statistics.median(spreads)
        total = statistics.median(totals)
        kickoff = event.get("commence_time")
        lines[home] = GameLine(
            pro_team=home,
            opponent=away,
            implied_total=total / 2 - home_spread / 2,
            spread=home_spread,
            total=total,
            kickoff=kickoff,
        )
        lines[away] = GameLine(
            pro_team=away,
            opponent=home,
            implied_total=total / 2 + home_spread / 2,
            spread=-home_spread,
            total=total,
            kickoff=kickoff,
        )
    return lines

"""nflverse usage adapter: snap counts, targets, carries and air yards, keyed to ESPN ids.

nflverse (github.com/nflverse) publishes play-by-play-derived NFL stats as public CSV release
assets: no key, no rate limit beyond GitHub's own, refreshed nightly in season. Three files are
used, each reduced to the handful of columns this server reads as soon as it is downloaded, and
the reduction (not the multi-megabyte CSV) is what gets cached:

* ``players.csv``: the crosswalk. Carries every player's ESPN id beside nflverse's own (GSIS)
  and Pro Football Reference's, which is what makes this an exact join rather than a fuzzy name
  match. Measured against a live 16-team league: 239 of 239 rostered skill players resolved.
* ``stats_player_week_{season}.csv``: per-game volume and fantasy points.
* ``snap_counts_{season}.csv``: per-game offensive snap share, keyed by PFR id.

Two abbreviations differ from ESPN's and are translated at the boundary (``_TEAM_ALIASES``), so
nothing downstream ever compares ``"LA"`` with ``"LAR"``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2

from ffmcp.domain.models import UsageWeek
from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache

RELEASES_URL = "https://github.com/nflverse/nflverse-data/releases/download"

CROSSWALK_TTL = 86_400
WEEKLY_TTL = 6 * 3_600
"""nflverse rebuilds weekly stats overnight; a few hours' staleness costs nothing."""

_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
_TEAM_ALIASES = {"LA": "LAR", "WAS": "WSH"}
"""nflverse spelling -> ESPN spelling, for the two teams the sources disagree on."""


class NflverseUsageProvider:
    """Live ``UsageProvider`` backed by nflverse's public release files."""

    def __init__(
        self,
        client: httpx2.AsyncClient,
        cache: Cache,
        resolve_season: Callable[[], Awaitable[int]],
    ) -> None:
        self._client = client
        self._cache = cache
        self._resolve_season = resolve_season

    async def get_usage(self, *, reception_points: float = 1.0) -> list[UsageWeek]:
        season = await self._resolve_season()
        crosswalk = await self._cache.get_or_fetch(
            "v1:nflverse:crosswalk", CROSSWALK_TTL, self._fetch_crosswalk
        )
        weekly = await self._cache.get_or_fetch(
            f"v1:nflverse:weekly:{season}", WEEKLY_TTL, lambda: self._fetch_weekly(season)
        )
        snaps = await self._cache.get_or_fetch(
            f"v1:nflverse:snaps:{season}", WEEKLY_TTL, lambda: self._fetch_snaps(season)
        )
        by_gsis: dict[str, int] = crosswalk["gsis"]
        by_pfr: dict[str, int] = crosswalk["pfr"]
        snap_by_espn = {(by_pfr[pfr], int(week)): pct for pfr, week, pct in snaps if pfr in by_pfr}

        result = []
        for row in weekly:
            espn_id = by_gsis.get(row["gsis"])
            if espn_id is None:
                continue
            week = int(row["week"])
            standard, ppr = row["points"]
            result.append(
                UsageWeek(
                    season=season,
                    week=week,
                    player_id=espn_id,
                    name=row["name"],
                    position=row["position"],
                    pro_team=_TEAM_ALIASES.get(row["team"], row["team"]),
                    snap_pct=snap_by_espn.get((espn_id, week)),
                    attempts=row["attempts"],
                    passing_air_yards=row["passing_air_yards"],
                    carries=row["carries"],
                    targets=row["targets"],
                    receiving_air_yards=row["receiving_air_yards"],
                    target_share=row["target_share"],
                    air_yards_share=row["air_yards_share"],
                    fantasy_points=standard + reception_points * (ppr - standard),
                )
            )
        return result

    async def _fetch_crosswalk(self) -> dict[str, dict[str, int]]:
        gsis: dict[str, int] = {}
        pfr: dict[str, int] = {}
        for row in await self._csv("players/players.csv"):
            espn = _int_or_none(row.get("espn_id"))
            if espn is None:
                continue
            if row.get("gsis_id"):
                gsis[row["gsis_id"]] = espn
            if row.get("pfr_id"):
                pfr[row["pfr_id"]] = espn
        return {"gsis": gsis, "pfr": pfr}

    async def _fetch_weekly(self, season: int) -> list[dict[str, Any]]:
        rows = await self._csv(f"stats_player/stats_player_week_{season}.csv")
        return [
            {
                "gsis": row["player_id"],
                "name": row["player_display_name"],
                "position": row["position"],
                "team": row["team"],
                "week": int(row["week"]),
                "attempts": _float(row.get("attempts")),
                "passing_air_yards": _float(row.get("passing_air_yards")),
                "carries": _float(row.get("carries")),
                "targets": _float(row.get("targets")),
                "receiving_air_yards": _float(row.get("receiving_air_yards")),
                "target_share": _float_or_none(row.get("target_share")),
                "air_yards_share": _float_or_none(row.get("air_yards_share")),
                "points": [
                    _float(row.get("fantasy_points")),
                    _float(row.get("fantasy_points_ppr")),
                ],
            }
            for row in rows
            if row.get("season_type") == "REG" and row.get("position") in _POSITIONS
        ]

    async def _fetch_snaps(self, season: int) -> list[tuple[str, int, float]]:
        rows = await self._csv(f"snap_counts/snap_counts_{season}.csv")
        return [
            (row["pfr_player_id"], int(row["week"]), _float(row.get("offense_pct")))
            for row in rows
            if row.get("game_type") == "REG" and row.get("position") in _POSITIONS
        ]

    async def _csv(self, path: str) -> list[dict[str, str]]:
        try:
            response = await self._client.get(f"{RELEASES_URL}/{path}", follow_redirects=True)
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise UpstreamUnavailable("nflverse") from exc
        return list(csv.DictReader(io.StringIO(response.text)))


def _float(value: str | None) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: str | None) -> float | None:
    if value is None or value in ("", "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    parsed = _float_or_none(value)
    return None if parsed is None else int(parsed)

"""FantasyCalc trade-market values: what managers actually trade players for.

Every trade suggestion in this server is priced by what it does to each roster's optimal lineup,
which is what a trade is *worth*. Whether it gets *accepted* is a different question: managers
anchor on names and market value, not on lineup arithmetic. FantasyCalc publishes redraft values
derived from real trades, for a league of a given size, quarterback count and reception scoring,
and that is the closest public read on "would a person say yes". ``domain.trades`` uses it only
as a plausibility filter, never as a value.

Each listed player carries his ESPN id, so there is no identity matching to do. Kickers and
defenses are not listed, and neither is anyone outside roughly the top 200.

Unofficial API with no published terms or compatibility promise. One request a day per league
shape; any failure means "market values unavailable", never a failed tool call.
"""

from __future__ import annotations

from typing import Any

import httpx2

from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache

VALUES_URL = "https://api.fantasycalc.com/values/current"
VALUES_TTL = 86_400


def ppr_parameter(reception_points: float) -> float:
    """FantasyCalc offers standard, half and full PPR; a league in between gets the nearest."""
    return min((0.0, 0.5, 1.0), key=lambda option: abs(option - reception_points))


class FantasyCalcValueProvider:
    """Live ``TradeValueProvider`` backed by FantasyCalc's public values endpoint."""

    def __init__(self, client: httpx2.AsyncClient, cache: Cache) -> None:
        self._client = client
        self._cache = cache

    async def get_trade_values(
        self, *, team_count: int, quarterbacks: int, reception_points: float
    ) -> dict[int, float]:
        ppr = ppr_parameter(reception_points)
        params = f"isDynasty=false&numQbs={quarterbacks}&numTeams={team_count}&ppr={ppr:g}"

        async def fetch() -> dict[str, float]:
            try:
                response = await self._client.get(f"{VALUES_URL}?{params}")
                response.raise_for_status()
            except httpx2.HTTPError as exc:
                raise UpstreamUnavailable("FantasyCalc") from exc
            return _reduce(response.json())

        key = f"v1:fantasycalc:{team_count}:{quarterbacks}:{ppr:g}"
        values = await self._cache.get_or_fetch(key, VALUES_TTL, fetch)
        return {int(espn_id): value for espn_id, value in values.items()}


def _reduce(rows: Any) -> dict[str, float]:
    """``{espn_id: value}``: the one id and the one number this server uses out of each
    ~800-byte row."""
    values: dict[str, float] = {}
    for row in rows or []:
        espn_id = (row.get("player") or {}).get("espnId")
        value = row.get("value")
        if espn_id and value is not None:
            values[str(espn_id)] = float(value)
    return values

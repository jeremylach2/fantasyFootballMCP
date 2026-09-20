"""``enrich_with_market`` merges the league provider's own market reading (ESPN ownership,
attached by ``providers.espn._map_market_signal``) with the market provider's signal (Sleeper's
trending-adds), rather than letting one overwrite the other. No real server or protocol layer
needed: a bare stand-in for ``Context`` exercises the function directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ffmcp.domain.models import MarketSignal, Player
from ffmcp.mcp._shared import enrich_with_market


def _player(player_id: int, *, market: MarketSignal | None = None) -> Player:
    return Player(
        player_id=player_id,
        name=f"Player {player_id}",
        position="RB",
        eligible_slots=("RB",),
        pro_team="BUF",
        market=market,
    )


class _FakeMarketProvider:
    def __init__(self, signal: MarketSignal | None) -> None:
        self._signal = signal

    async def get_market_signal(
        self, *, espn_player_id: int, name: str, position: str, pro_team: str
    ) -> MarketSignal | None:
        return self._signal


@dataclass
class _FakeAppContext:
    market: _FakeMarketProvider


@dataclass
class _FakeRequestContext:
    lifespan_context: _FakeAppContext


@dataclass
class _FakeCtx:
    request_context: _FakeRequestContext


def _ctx(signal: MarketSignal | None) -> _FakeCtx:
    return _FakeCtx(_FakeRequestContext(_FakeAppContext(_FakeMarketProvider(signal))))


async def test_sleeper_trending_adds_is_merged_onto_existing_espn_ownership() -> None:
    espn_only = _player(1, market=MarketSignal(percent_owned=87.3, percent_started=71.2))
    sleeper_signal = MarketSignal(trending_adds=12_000)

    [enriched] = await enrich_with_market(_ctx(sleeper_signal), [espn_only])

    assert enriched.market is not None
    assert enriched.market.percent_owned == 87.3
    assert enriched.market.percent_started == 71.2
    assert enriched.market.trending_adds == 12_000


async def test_no_sleeper_signal_leaves_espn_ownership_untouched() -> None:
    espn_only = _player(1, market=MarketSignal(percent_owned=45.0))

    [enriched] = await enrich_with_market(_ctx(None), [espn_only])

    assert enriched.market is not None
    assert enriched.market.percent_owned == 45.0


async def test_no_prior_market_signal_uses_sleepers_as_is() -> None:
    bare = _player(1)
    sleeper_signal = MarketSignal(trending_adds=500)

    [enriched] = await enrich_with_market(_ctx(sleeper_signal), [bare])

    assert enriched.market == sleeper_signal

"""Server construction and process entry point.

Tools, resources and prompts are registered here in a fixed order. Ordering is deliberate:
the MCP specification asks servers to return ``tools/list`` deterministically so clients can
cache it and so host prompt caches hit more often.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2
from mcp.server.caching import CacheableMethod, CacheHint
from mcp.server.mcpserver import MCPServer

from ffmcp import __version__
from ffmcp.config import ConfigurationError, Settings, load_settings
from ffmcp.errors import FFMCPError
from ffmcp.providers.base import LeagueProvider, MarketProvider
from ffmcp.providers.cache import Cache
from ffmcp.providers.demo import DemoLeagueProvider, DemoMarketProvider
from ffmcp.providers.espn import EspnLeagueProvider
from ffmcp.providers.sleeper import SleeperMarketProvider

logger = logging.getLogger("ffmcp")

# Ships in the context of every session, so every clause has to earn its place.
INSTRUCTIONS = (
    "Analyzes the user's ESPN fantasy football league. All tools are read-only; the user "
    "must act on recommendations in the ESPN app. Projections are estimates — state "
    "uncertainty rather than implying precision. Prefer optimize_lineup over reasoning "
    "about rosters yourself."
)

# Freshness hints let clients cache instead of re-fetching. Anything derived from the
# owner's league is "private": it must not sit in a shared intermediary cache.
CACHE_HINTS: Mapping[CacheableMethod, CacheHint] = {
    "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
    "prompts/list": CacheHint(ttl_ms=3_600_000, scope="public"),
    "resources/list": CacheHint(ttl_ms=3_600_000, scope="public"),
    "resources/read": CacheHint(ttl_ms=300_000, scope="private"),
}


@dataclass
class AppContext:
    """What ``mcp/tools_*.py`` reads off ``ctx.request_context.lifespan_context``."""

    settings: Settings
    cache: Cache
    league: LeagueProvider
    market: MarketProvider


_current_app_context: AppContext | None = None
"""The one running server's ``AppContext``, for the handful of *static* resources that need it.
The installed SDK's static resource functions take no parameters at all, not even ``Context``
(only resource *templates* and tools get that injected), so there is no per-call path to the
lifespan context for them. A module-level singleton is safe here because a server process runs
exactly one lifespan for its whole life on every transport this project supports (stdio or
stateless HTTP, never multiple concurrent lifespans in one process)."""


def get_app_context() -> AppContext:
    """Accessor for static resources. Anything with a ``Context`` should prefer
    ``ctx.request_context.lifespan_context`` instead (``ffmcp.mcp._shared.app_context``)."""
    if _current_app_context is None:
        raise RuntimeError("AppContext is not available outside of a running server lifespan")
    return _current_app_context


@asynccontextmanager
async def app_lifespan(_server: MCPServer) -> AsyncGenerator[AppContext]:
    """Select live vs. demo providers once, here, never inside a tool (docs/architecture.md §6).
    Must not touch the network before the first tool call (docs/architecture.md §3): a
    ``League`` handle and the Sleeper client are both built lazily, so nothing below actually
    fetches anything yet.
    """
    global _current_app_context
    settings = load_settings()
    cache = Cache(settings.cache_dir)

    league: LeagueProvider
    market: MarketProvider
    http_client: httpx2.AsyncClient | None = None

    if settings.mode == "demo":
        league = DemoLeagueProvider()
        market = DemoMarketProvider()
    else:
        assert settings.league_id is not None  # guaranteed by load_settings() in live mode
        http_client = httpx2.AsyncClient(timeout=httpx2.Timeout(15.0, connect=5.0))
        market = SleeperMarketProvider(http_client, cache)
        # FFMCP_SEASON defaults to "current", derived from Sleeper (docs/architecture.md §5),
        # resolved lazily on first real use rather than here. Startup must not touch the
        # network (docs/architecture.md §3).
        league = EspnLeagueProvider(
            league_id=settings.league_id,
            season=settings.season,
            espn_s2=settings.espn_s2.get_secret_value() if settings.espn_s2 else None,
            swid=settings.swid.get_secret_value() if settings.swid else None,
            resolve_season=market.get_current_season,
        )

    app_context = AppContext(settings=settings, cache=cache, league=league, market=market)
    _current_app_context = app_context
    try:
        yield app_context
    finally:
        _current_app_context = None
        if http_client is not None:
            await http_client.aclose()


def build_server() -> MCPServer:
    """Construct the server. Registration happens here so tests can build it in-process.

    The ``ffmcp.mcp.*`` imports are deferred to inside this function: those modules import
    ``AppContext`` from this one, so importing them at module scope would be a cycle. Tools are
    registered one call per tool, in a fixed documented order, since the spec asks for a
    deterministic ``tools/list`` order for prompt-cache stability, and grouping registration by
    file (as ``mcp/tools_*.py`` are organized by perspective, not by surface order) would not
    produce it.
    """
    from ffmcp.mcp import prompts, resources, tools_league, tools_lineup, tools_roster, tools_trade

    mcp = MCPServer(
        name="fantasy-football",
        title="ESPN Fantasy Football",
        version=__version__,
        instructions=INSTRUCTIONS,
        cache_hints=CACHE_HINTS,
        lifespan=app_lifespan,
    )

    tools_roster.register_get_my_team(mcp)
    tools_lineup.register_optimize_lineup(mcp)
    tools_lineup.register_analyze_matchup(mcp)
    tools_trade.register_find_trades(mcp)
    tools_trade.register_evaluate_trade(mcp)
    tools_roster.register_find_waiver_targets(mcp)
    tools_league.register_simulate_season(mcp)
    tools_league.register_league_standings(mcp)
    tools_roster.register_player_report(mcp)
    tools_roster.register_compare_players(mcp)

    resources.register(mcp)
    prompts.register(mcp)

    return mcp


def _configure_logging() -> None:
    """Log to stderr only.

    On stdio transport, stdout carries the protocol. Writing anything else to it corrupts
    the session, which is the most common way to break an MCP server.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="ffmcp", description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio for local clients (default); streamable-http for deployment",
    )
    parser.add_argument("--port", type=int, default=8000, help="port for streamable-http")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address for streamable-http (use 0.0.0.0 in a container)",
    )
    args = parser.parse_args()

    _configure_logging()

    # Validate configuration *before* a transport starts. `app_lifespan` loads settings too,
    # but by then we are inside anyio's task group, and a raise there reaches the operator as a
    # forty-line ExceptionGroup traceback rather than the single actionable line the error was
    # written to be. Fail fast, on stderr, with a non-zero exit.
    try:
        load_settings()
    except (ConfigurationError, FFMCPError) as exc:
        print(f"ffmcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    mcp = build_server()

    if args.transport == "streamable-http":
        # Stateless matches the 2026-07-28 core and load-balances without sticky sessions.
        mcp.run(transport="streamable-http", host=args.host, port=args.port, stateless_http=True)
    else:
        mcp.run()


if __name__ == "__main__":
    main()

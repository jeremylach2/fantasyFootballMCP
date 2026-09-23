"""Adapter-shared plumbing: the glue every ``mcp/tools_*.py`` module needs, so none of them
duplicate it. Nothing here is business logic: it is entirely decode/dispatch, reading the
lifespan context off ``ctx``, assembling a ``LeagueState`` from the provider protocol, resolving
"my team", and looking a player up by name. The arithmetic lives in ``domain/``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from ffmcp.domain.models import (
    GameLine,
    LeagueSettings,
    LeagueState,
    Player,
    PlayerWeek,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.domain.schedule import WeekWeight, remaining_week_weights
from ffmcp.domain.usage import UsageProfile, usage_profiles
from ffmcp.domain.variance import player_sds
from ffmcp.errors import FFMCPError, PlayerNotFound, TeamNotConfigured, WeekOutOfRange
from ffmcp.server import AppContext

logger = logging.getLogger("ffmcp.tools")

OPTIONAL_FAILURES = (FFMCPError, ValueError, KeyError)
"""What an *optional* signal (history, a second projection source, usage, betting lines) may
fail with and still leave the tool standing. Each of these sources only refines an answer the
server can already give from ESPN alone, so its absence becomes a one-line caveat rather than
an error: a tool that refuses to optimize a lineup because GitHub is slow would be worse than
one that optimizes it with position-level variance and says so."""

Detail = Literal["compact", "standard", "full"]
"""Shared by every list-shaped tool. Default is always ``"compact"``."""


@contextmanager
def adapt_errors(wrap: type[Exception] = ToolError) -> Iterator[None]:
    """Wrap a tool (or resource) body's logic in ``with adapt_errors():``.

    Every ``FFMCPError`` in this project is written to be read by the model: short, actionable,
    no stack trace (docs/architecture.md §4 rule 1). But the installed SDK
    only forwards a ``ToolError``'s (or ``ResourceError``'s) text to the client. Any other
    exception is treated as a crash and its message is logged server-side, never sent
    (``mcp/server/mcpserver/tools/base.py``). This bridges the two at the one place every tool
    and resource already has to sit anyway. Pass ``wrap=ResourceError`` inside a resource.
    """
    try:
        yield
    except FFMCPError as exc:
        raise wrap(str(exc)) from exc


_FREE_AGENT_POOL_SIZE = 300
"""Free agents pulled when searching by name or valuing the replacement pool. Generous enough
to cover a 12-team league's relevant universe without asking the provider for everything."""


def app_context(ctx: object) -> AppContext:
    """Pull the lifespan-built ``AppContext`` off a tool's injected ``Context``.

    Typed ``object`` rather than ``mcp.server.mcpserver.Context`` so this module, and every
    tool module importing it, need not import the SDK just for an accessor. The real type is
    enforced at the call site, where ``ctx`` is always a ``Context``.
    """
    return ctx.request_context.lifespan_context  # type: ignore[attr-defined,no-any-return]


async def resolve_week(ctx: object, week: int | None, settings: LeagueSettings) -> int:
    """The requested week, defaulting to the current one and validated against the season."""
    app = app_context(ctx)
    if week is None:
        return await app.league.get_current_week()
    if not 1 <= week <= settings.reg_season_weeks:
        raise WeekOutOfRange(week, valid_min=1, valid_max=settings.reg_season_weeks)
    return week


async def load_league_state(ctx: object) -> LeagueState:
    """Assemble the full snapshot ``domain.simulate`` needs: settings, teams, and every
    remaining week's matchups, fetched concurrently.

    Tools that only need the roster/settings shape (``get_my_team``, ``find_waiver_targets``)
    should call the provider directly instead: this pulls the whole remaining schedule and is
    for the tools that price a season-long outcome.
    """
    app = app_context(ctx)
    settings = await app.league.get_settings()
    current_week = await app.league.get_current_week()
    teams = await app.league.get_teams()
    weeks = range(current_week, settings.reg_season_weeks + 1)
    matchup_lists = await asyncio.gather(*(app.league.get_matchups(week) for week in weeks))
    matchups = tuple(matchup for week_matchups in matchup_lists for matchup in week_matchups)
    return LeagueState(
        settings=settings, teams=tuple(teams), current_week=current_week, matchups=matchups
    )


class _TeamChoice(BaseModel):
    team_id: int = Field(description="The numeric team id from the roster below.")


async def resolve_my_team_id(ctx: object, teams: Sequence[Team]) -> int:
    """Which team is "mine": the one elicited spot in the whole surface.

    ``FFMCP_TEAM_ID`` wins when set. Otherwise, a client that can elicit is asked once per
    session and the answer is cached onto ``Settings`` for the rest of it. A client that
    cannot elicit gets a two-line error naming the variable, never a guess.
    """
    app = app_context(ctx)
    if app.settings.team_id is not None:
        return app.settings.team_id

    if ctx.client_capabilities.elicitation is not None:  # type: ignore[attr-defined]
        roster = "; ".join(f"{team.team_id}={team.name}" for team in teams)
        result = await ctx.elicit(  # type: ignore[attr-defined]
            f"Which team is yours? {roster}", _TeamChoice
        )
        if result.action == "accept" and result.data is not None:
            chosen: int = result.data.team_id
            if any(team.team_id == chosen for team in teams):
                app.settings.team_id = chosen
                return chosen

    raise TeamNotConfigured()


def find_players_by_name(pool: Sequence[Player], query: str) -> list[Player]:
    """Case-insensitive lookup: exact matches win outright, otherwise every substring match.

    Returns an empty list on no match and more than one entry on ambiguity. The caller decides
    what to do with either: on ambiguity, stop and do not guess.
    """
    needle = query.strip().lower()
    exact = [player for player in pool if player.name.lower() == needle]
    if exact:
        return exact
    return [player for player in pool if needle in player.name.lower()]


async def player_pool(ctx: object, teams: Sequence[Team], week: int) -> list[Player]:
    """Every player in the league worth knowing about by name: rostered (including IR) plus a
    generous slice of free agents. Deduplicated by id, first occurrence wins."""
    app = app_context(ctx)
    free_agents = await app.league.get_free_agents(week, size=_FREE_AGENT_POOL_SIZE)
    seen: set[int] = set()
    pool: list[Player] = []
    for team in teams:
        for player in (*team.roster.players, *team.roster.ir):
            if player.player_id not in seen:
                seen.add(player.player_id)
                pool.append(player)
    for player in free_agents:
        if player.player_id not in seen:
            seen.add(player.player_id)
            pool.append(player)
    return pool


async def enrich_with_market(ctx: object, players: Sequence[Player]) -> list[Player]:
    """Attach a ``MarketSignal`` to each player, via the market provider's own identity
    resolution. Cross-provider, so it happens here rather than inside either provider: an ESPN
    roster fetch has no reason to know Sleeper exists (docs/architecture.md §3).

    ``player.market`` may already carry ``percent_owned``/``percent_started`` set by the league
    provider itself (``providers.espn._map_market_signal``: this league's own ESPN ownership
    stats, unrelated to Sleeper). The market provider's signal is merged on top rather than
    replacing it outright, so Sleeper's ``trending_adds`` doesn't wipe out ESPN's ownership
    reading, and vice versa.
    """
    app = app_context(ctx)

    async def _one(player: Player) -> Player:
        signal = await app.market.get_market_signal(
            espn_player_id=player.player_id,
            name=player.name,
            position=player.position,
            pro_team=player.pro_team,
        )
        if signal is None:
            return player
        if player.market is not None:
            signal = signal.model_copy(
                update={
                    "percent_owned": signal.percent_owned
                    if signal.percent_owned is not None
                    else player.market.percent_owned,
                    "percent_started": signal.percent_started
                    if signal.percent_started is not None
                    else player.market.percent_started,
                }
            )
        return player.model_copy(update={"market": signal})

    return list(await asyncio.gather(*(_one(player) for player in players)))


async def enrich_team_roster(ctx: object, team: Team) -> Team:
    """``team`` with a ``MarketSignal`` attached to every rostered player (slots, bench, IR).

    Shared by ``get_my_team`` and the ``team/{team_id}/roster`` resource template so the two
    don't diverge on how "standard"/"full" detail gets its market column.
    """
    slot_players = [slot.player for slot in team.roster.slots if slot.player is not None]
    all_players = [*slot_players, *team.roster.bench, *team.roster.ir]
    enriched = await enrich_with_market(ctx, all_players)
    by_id = {player.player_id: player for player in enriched}
    new_slots = tuple(
        RosterSlot(slot=slot.slot, player=by_id[slot.player.player_id] if slot.player else None)
        for slot in team.roster.slots
    )
    new_bench = tuple(by_id[player.player_id] for player in team.roster.bench)
    new_ir = tuple(by_id[player.player_id] for player in team.roster.ir)
    return team.model_copy(update={"roster": Roster(slots=new_slots, bench=new_bench, ir=new_ir)})


@dataclass
class Insights:
    """Everything the variance model knows this week, and what it had to do without."""

    sds: dict[int, float] = field(default_factory=dict)
    """``{player_id: weekly sd}``: see ``domain.variance``."""
    alternative: dict[int, float] = field(default_factory=dict)
    """Second-source projections for this week, keyed by ESPN id."""
    caveats: list[str] = field(default_factory=list)


async def load_history(ctx: object) -> tuple[list[PlayerWeek], str | None]:
    """This season's completed player-weeks, or ``[]`` and a caveat."""
    app = app_context(ctx)
    try:
        return await app.league.get_player_history(), None
    except OPTIONAL_FAILURES as exc:
        logger.warning("history unavailable: %s", exc)
        return [], "League history unavailable; using position-level variance."


async def load_insights(
    ctx: object, players: Sequence[Player], week: int, settings: LeagueSettings
) -> Insights:
    """The variance inputs for ``players`` in ``week``: each position's measured spread,
    widened where a second projection source disagrees with ESPN. Without the second source
    the position spread stands alone, with a caveat saying so."""
    app = app_context(ctx)
    caveats: list[str] = []
    try:
        alternative = await app.market.get_alt_projections(
            week, players, reception_points=settings.reception_points
        )
    except OPTIONAL_FAILURES as exc:
        logger.warning("second projection source unavailable: %s", exc)
        alternative = {}
        caveats.append("Second projection source unavailable; ranges use ESPN alone.")
    sds = player_sds(players, alternative=alternative)
    return Insights(sds=sds, alternative=alternative, caveats=caveats)


async def load_usage(
    ctx: object, settings: LeagueSettings
) -> tuple[dict[int, UsageProfile], str | None]:
    """Season-to-date usage profiles, or ``{}`` and a caveat."""
    app = app_context(ctx)
    try:
        weeks = await app.usage.get_usage(reception_points=settings.reception_points)
    except OPTIONAL_FAILURES as exc:
        logger.warning("usage data unavailable: %s", exc)
        return {}, "Usage data (nflverse) unavailable right now."
    return usage_profiles(weeks, settings.reception_points), None


async def load_game_lines(ctx: object) -> dict[str, GameLine]:
    """This week's betting lines by pro team, or ``{}``: purely contextual, so no caveat."""
    app = app_context(ctx)
    try:
        return await app.odds.get_game_lines()
    except OPTIONAL_FAILURES as exc:
        logger.warning("game lines unavailable: %s", exc)
        return {}


def week_weights_by_team(state: LeagueState) -> dict[int, tuple[WeekWeight, ...]]:
    """Each team's remaining-season calendar, playoff weeks weighted by that team's own
    playoff odds (ESPN's, already on ``Team``: no simulation needed to price a trade)."""
    return {
        team.team_id: remaining_week_weights(
            state.settings, state.current_week, playoff_probability=team.playoff_pct / 100.0
        )
        for team in state.teams
    }


def league_players(state: LeagueState) -> list[Player]:
    """Every startable rostered player in the league."""
    return [player for team in state.teams for player in team.roster.players]


def require_single_match(matches: Sequence[Player], query: str) -> Player:
    """The one player ``query`` names, or a ``PlayerNotFound`` naming up to 3 candidates."""
    if len(matches) != 1:
        candidates = [player.name for player in matches] if matches else None
        raise PlayerNotFound(query, candidates)
    return matches[0]

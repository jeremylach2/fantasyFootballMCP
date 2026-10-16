"""Adapter-shared plumbing: the glue every ``mcp/tools_*.py`` module needs, so none of them
duplicate it. Nothing here is business logic: it is entirely decode/dispatch, reading the
lifespan context off ``ctx``, assembling a ``LeagueState`` from the provider protocol, resolving
"my team", looking a player up by name, and caching the one simulation every tool reads its
playoff odds from. The arithmetic lives in ``domain/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from ffmcp.domain.models import (
    GameLine,
    LeagueSettings,
    LeagueState,
    Player,
    PlayerWeek,
    Projection,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.domain.names import match_players, suggestions
from ffmcp.domain.optimizer import is_ruled_out
from ffmcp.domain.schedule import WeekWeight, as_of_week, remaining_week_weights
from ffmcp.domain.simulate import TEAM_STRENGTH_SD, SeasonOutcome, simulate_rest_of_season
from ffmcp.domain.usage import UsageProfile, usage_profiles
from ffmcp.domain.variance import player_sds
from ffmcp.errors import FFMCPError, PlayerNotFound, TeamNotConfigured, WeekOutOfRange
from ffmcp.server import AppContext

logger = logging.getLogger("ffmcp.tools")

OPTIONAL_FAILURES = (FFMCPError, ValueError, KeyError)
"""What an *optional* signal (history, a second projection source, usage, betting lines, market
values) may fail with and still leave the tool standing. Each of these sources only refines an
answer the server can already give from ESPN alone, so its absence becomes a one-line caveat
rather than an error: a tool that refuses to optimize a lineup because GitHub is slow would be
worse than one that optimizes it with position-level variance and says so."""

Detail = Literal["compact", "standard", "full"]
"""Shared by every list-shaped tool. Default is always ``"compact"``."""

SIMULATION_TTL_SECONDS = 6 * 3_600
"""How long one simulation of the season is reused. Its key is a fingerprint of everything it
reads, so a roster move or a new projection is a new key, not a stale answer."""


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
to cover a 16-team league's relevant universe without asking the provider for everything."""


def app_context(ctx: object) -> AppContext:
    """Pull the lifespan-built ``AppContext`` off a tool's injected ``Context``.

    Typed ``object`` rather than ``mcp.server.mcpserver.Context`` so this module, and every
    tool module importing it, need not import the SDK just for an accessor. The real type is
    enforced at the call site, where ``ctx`` is always a ``Context``.
    """
    return ctx.request_context.lifespan_context  # type: ignore[attr-defined,no-any-return]


def now(ctx: object) -> datetime:
    """The current instant: the real clock live, the moment a fixture was recorded in demo
    mode, so a lineup lock or a trade deadline reads the same in a test as on the day."""
    return app_context(ctx).clock()


async def resolve_week(ctx: object, week: int | None, settings: LeagueSettings) -> int:
    """The requested week, defaulting to the current one and validated against the season."""
    app = app_context(ctx)
    if week is None:
        return await app.league.get_current_week()
    if not 1 <= week <= settings.reg_season_weeks:
        raise WeekOutOfRange(week, valid_min=1, valid_max=settings.reg_season_weeks)
    return week


def week_note(week: int, current_week: int) -> str | None:
    """The one-line caveat every view of a later week carries."""
    if week <= current_week:
        return None
    return (
        f"Week {week} uses season-average projections with everyone healthy; byes applied. "
        "Real matchups and injuries will move it."
    )


def team_as_of_week(team: Team, week: int, current_week: int) -> Team:
    """``team`` as a later week sees it (``domain.schedule.as_of_week``), lineup as set now."""
    if week == current_week:
        return team

    def view(player: Player) -> Player:
        return as_of_week(player, week, current_week)

    roster = team.roster
    return team.model_copy(
        update={
            "roster": Roster(
                slots=tuple(
                    RosterSlot(slot=s.slot, player=view(s.player) if s.player else None)
                    for s in roster.slots
                ),
                bench=tuple(view(p) for p in roster.bench),
                ir=tuple(view(p) for p in roster.ir),
            )
        }
    )


async def load_league_state(ctx: object) -> LeagueState:
    """Assemble the full snapshot ``domain.simulate`` needs: settings, teams, and every
    remaining week's matchups, fetched concurrently (the provider fetches the schedule once).

    Tools that only need the roster/settings shape (``get_my_team``) should call the provider
    directly instead: this is for the tools that price a season-long outcome.
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


def _fingerprint(state: LeagueState, n_sims: int) -> str:
    """Everything the simulation reads, hashed. Live scores and ownership are left out: they
    change by the minute on Sunday and the simulation does not read them."""

    def player(p: Player) -> list[object]:
        return [
            p.player_id,
            p.position,
            p.eligible_slots,
            p.projection.week if p.projection else None,
            p.projection.points if p.projection else None,
            p.injury_status,
            p.injured,
            p.bye_week,
            p.season_rate,
            [(o.week, o.points) for o in p.outlook],
        ]

    payload = {
        "v": 2,
        "sims": n_sims,
        "strength": TEAM_STRENGTH_SD,
        "settings": state.settings.model_dump(mode="json"),
        "week": state.current_week,
        "teams": [
            [
                t.team_id,
                t.wins,
                t.losses,
                t.ties,
                round(t.points_for, 2),
                t.division_id,
                [player(p) for p in t.roster.players],
            ]
            for t in state.teams
        ],
        "games": sorted(
            (m.week, m.home_team_id or -1, m.away_team_id or -1, m.is_playoff)
            for m in state.matchups
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


async def load_season_outcome(
    ctx: object,
    state: LeagueState,
    *,
    n_sims: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SeasonOutcome:
    """The season simulation, run once per league state and reused by every tool that shows or
    prices playoff odds. One source: before this, ``league_standings`` showed ESPN's number,
    ``simulate_season`` its own, and ``analyze_matchup`` a third, and a manager who sees three
    different odds for one team stops trusting all of them."""
    app = app_context(ctx)
    sims = n_sims if n_sims is not None else app.settings.sims
    key = (
        f"v2:sim:{state.settings.league_id}:{state.settings.season}:{state.current_week}:"
        f"{_fingerprint(state, sims)}"
    )

    async def run() -> dict[str, object]:
        outcome = await asyncio.to_thread(
            simulate_rest_of_season, state, sims, None, progress=progress
        )
        return outcome.model_dump(mode="json")

    return SeasonOutcome.model_validate(
        await app.cache.get_or_fetch(key, SIMULATION_TTL_SECONDS, run)
    )


async def playoff_probabilities(ctx: object, state: LeagueState) -> dict[int, float]:
    """``{team_id: probability in [0, 1]}``, from the shared simulation."""
    return (await load_season_outcome(ctx, state)).playoff_probabilities()


def week_weights_by_team(
    state: LeagueState, probabilities: dict[int, float]
) -> dict[int, tuple[WeekWeight, ...]]:
    """Each team's remaining-season calendar, its playoff weeks weighted by its own simulated
    playoff odds."""
    return {
        team.team_id: remaining_week_weights(
            state.settings,
            state.current_week,
            playoff_probability=probabilities.get(team.team_id, 0.5),
        )
        for team in state.teams
    }


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
    """Everyone ``query`` names, after normalizing case, punctuation and suffixes
    (``domain.names``): exact matches win outright, otherwise every substring match.

    Returns an empty list on no match and more than one entry on ambiguity. The caller decides
    what to do with either: on ambiguity, stop and do not guess.
    """
    return match_players(pool, query)


def require_single_match(
    matches: Sequence[Player], query: str, pool: Sequence[Player] = ()
) -> Player:
    """The one player ``query`` names, or a ``PlayerNotFound`` naming up to 3 candidates (on
    ambiguity) or the closest names in ``pool`` (on no match)."""
    if len(matches) != 1:
        candidates = [player.name for player in matches] if matches else None
        raise PlayerNotFound(
            query, candidates, suggestions=None if matches else suggestions(pool, query)
        )
    return matches[0]


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


def needs_outlook(player: Player) -> bool:
    """A player whose rest of season ESPN cannot price: ruled out, or with no season rate
    (ESPN publishes zero for anyone on injured reserve)."""
    return is_ruled_out(player) or player.season_rate is None


async def attach_outlooks(
    ctx: object, players: Sequence[Player], settings: LeagueSettings
) -> list[Player]:
    """``players`` with a week-by-week outlook attached to each injured one
    (``MarketProvider.get_outlook``), which is how an expected return reaches every season
    value. Everyone else, and everyone when the source is down, comes back unchanged."""
    app = app_context(ctx)
    injured = [player for player in players if needs_outlook(player)]
    if not injured:
        return list(players)
    try:
        outlooks = await app.market.get_outlook(injured, reception_points=settings.reception_points)
    except OPTIONAL_FAILURES as exc:
        logger.warning("injury outlook unavailable: %s", exc)
        return list(players)
    result = []
    for player in players:
        weeks = outlooks.get(player.player_id)
        if weeks:
            outlook = tuple(
                Projection(week=week, points=points, source="sleeper")
                for week, points in sorted(weeks.items())
            )
            player = player.model_copy(update={"outlook": outlook})
        result.append(player)
    return result


def quarterback_count(settings: LeagueSettings) -> int:
    """Quarterbacks a lineup can start: 2 with a superflex slot, else 1."""
    slots = settings.slot_counts
    return 2 if slots.get("OP", 0) or slots.get("SUPERFLEX", 0) or slots.get("QB", 0) > 1 else 1


async def load_trade_values(
    ctx: object, settings: LeagueSettings
) -> tuple[dict[int, float] | None, str | None]:
    """Market trade values for a league of this shape, or ``None`` and a caveat."""
    app = app_context(ctx)
    try:
        values = await app.values.get_trade_values(
            team_count=settings.team_count,
            quarterbacks=quarterback_count(settings),
            reception_points=settings.reception_points,
        )
    except OPTIONAL_FAILURES as exc:
        logger.warning("market values unavailable: %s", exc)
        return None, "Market values unavailable; offers are not checked for plausibility."
    return (values or None), None


def lineup_locks(
    roster_slots: Sequence[RosterSlot],
    bench: Sequence[Player],
    slots: Sequence[str],
    at: datetime,
) -> tuple[dict[int, int], set[int]]:
    """``(pinned, frozen)`` for ``domain.optimizer.optimize``: starters whose games have kicked
    off stay in their slots (by index into ``slots``), benched ones stay benched."""
    pinned: dict[int, int] = {}
    taken: set[int] = set()
    for roster_slot in roster_slots:
        player = roster_slot.player
        if player is None or player.kickoff is None or player.kickoff > at:
            continue
        index = next(
            (i for i, slot in enumerate(slots) if slot == roster_slot.slot and i not in taken),
            None,
        )
        if index is not None:
            taken.add(index)
            pinned[player.player_id] = index
    frozen = {p.player_id for p in bench if p.kickoff is not None and p.kickoff <= at}
    return pinned, frozen


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


async def load_upcoming_lines(ctx: object) -> dict[str, list[GameLine]]:
    """Every posted game line per pro team, or ``{}``: context only, so no caveat."""
    app = app_context(ctx)
    try:
        return await app.odds.get_upcoming_lines()
    except OPTIONAL_FAILURES as exc:
        logger.warning("game lines unavailable: %s", exc)
        return {}


def league_players(state: LeagueState) -> list[Player]:
    """Every startable rostered player in the league."""
    return [player for team in state.teams for player in team.roster.players]

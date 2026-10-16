"""League-wide tools: the season simulation, the standings table, power rankings and recent
activity. Adapters only: decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.league_intel import manager_reports
from ffmcp.domain.models import LeagueSettings
from ffmcp.mcp._shared import (
    Detail,
    adapt_errors,
    app_context,
    load_history,
    load_league_state,
    load_season_outcome,
)
from ffmcp.render.tables import (
    render_activity,
    render_power_rankings,
    render_season_outcome,
    render_standings,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
"""``idempotent_hint=True`` because the default seed is fixed: identical arguments reproduce
the identical table."""

MIN_SIMS = 500
MAX_SIMS = 50_000
"""``n_sims`` is clamped to this range. Below it the odds wobble by several points; above it
a confused caller could tie up the server for minutes for no visible gain."""

_MODELLED_TIE_RULES = frozenset({"TOTAL_POINTS_SCORED"})
_MAX_ACTIVITY = 25


def format_caveats(settings: LeagueSettings) -> list[str]:
    """What the simulation approximates about this league's format, said once."""
    caveats = []
    rule = settings.playoff_seed_tie_rule
    if rule and rule not in _MODELLED_TIE_RULES:
        caveats.append(
            f"Tied records are seeded by total points here; this league uses "
            f"{rule.replace('_', ' ').lower()}, so close seeds are approximate."
        )
    return caveats


def register_simulate_season(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Simulate Season",
        description=(
            "Simulate the rest of the season for playoff odds and seeding. Each team's every "
            "remaining week is projected separately (byes, the league's format). n_sims is "
            f"clamped to {MIN_SIMS:,}-{MAX_SIMS:,}."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def simulate_season(
        ctx: Context, n_sims: int | None = None, team_id: int | None = None
    ) -> str:
        with adapt_errors():
            app = app_context(ctx)
            state = await load_league_state(ctx)
            sims = min(max(n_sims if n_sims is not None else app.settings.sims, MIN_SIMS), MAX_SIMS)
            loop = asyncio.get_running_loop()

            def report(done: int, total: int) -> None:
                # Called from the worker thread the simulation runs in (it is CPU-bound numpy,
                # docs/architecture.md §7). `ctx.report_progress` is a coroutine, so it has to
                # be handed back to the event loop rather than awaited here.
                asyncio.run_coroutine_threadsafe(ctx.report_progress(done, total), loop)

            outcome = await load_season_outcome(ctx, state, n_sims=sims, progress=report)
            return render_season_outcome(
                outcome, highlight_team_id=team_id, caveats=format_caveats(state.settings)
            )


def register_league_standings(mcp: MCPServer) -> None:
    @mcp.tool(
        title="League Standings",
        description=(
            "Show standings with simulated playoff odds (the same simulation as simulate_season) "
            "and strength of schedule. The ID column is each team's team_id, needed by "
            "evaluate_trade's partner_team_id — do not guess it from rank position, which "
            "reflects record, not id."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def league_standings(ctx: Context, detail: Detail = "compact") -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            outcome = await load_season_outcome(ctx, state)
            odds = {team.team_id: team.playoff_odds for team in outcome.teams}
            return render_standings(state, detail, odds)


def register_power_rankings(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Power Rankings",
        description=(
            "Rank teams by all-play record (every team vs every team, every week), with "
            "each team's luck and how well its manager sets lineups."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def power_rankings(ctx: Context) -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            history, caveat = await load_history(ctx)
            if not history:
                return caveat or "No completed weeks yet: power rankings start after week 1."
            weeks = len({row.week for row in history})
            return render_power_rankings(manager_reports(state, history), weeks)


def register_league_activity(mcp: MCPServer) -> None:
    @mcp.tool(
        title="League Activity",
        description=(
            "The league's latest adds, drops, waiver claims and trades, newest first: who just "
            "got dropped, and who is active."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def league_activity(ctx: Context, limit: int = 10) -> str:
        with adapt_errors():
            app = app_context(ctx)
            capped = min(max(1, limit), _MAX_ACTIVITY)
            teams = await app.league.get_teams()
            transactions = await app.league.get_recent_activity(size=_MAX_ACTIVITY)
            names = {team.team_id: team.name for team in teams}
            return render_activity(transactions, names, limit=capped)

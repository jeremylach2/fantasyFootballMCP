"""League-wide tools: the season simulation and the standings table. Adapters only: decode
args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.simulate import simulate_rest_of_season
from ffmcp.mcp._shared import Detail, adapt_errors, app_context, load_league_state
from ffmcp.render.tables import render_season_outcome, render_standings

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
"""``idempotent_hint=True`` because the default seed is fixed: identical arguments reproduce
the identical table."""


def register_simulate_season(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Simulate Season",
        description="Simulate the rest of the season for playoff odds and seeding.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def simulate_season(
        ctx: Context, n_sims: int | None = None, team_id: int | None = None
    ) -> str:
        with adapt_errors():
            app = app_context(ctx)
            state = await load_league_state(ctx)
            sims = n_sims if n_sims is not None else app.settings.sims
            loop = asyncio.get_running_loop()

            def report(done: int, total: int) -> None:
                # Called from the worker thread `simulate_rest_of_season` runs in (it is
                # CPU-bound numpy, docs/architecture.md §7). `ctx.report_progress` is a
                # coroutine, so it has to be handed back to the event loop rather than
                # awaited here.
                asyncio.run_coroutine_threadsafe(ctx.report_progress(done, total), loop)

            outcome = await asyncio.to_thread(
                simulate_rest_of_season, state, sims, None, progress=report
            )
            return render_season_outcome(outcome, highlight_team_id=team_id)


def register_league_standings(mcp: MCPServer) -> None:
    @mcp.tool(
        title="League Standings",
        description="Show standings with playoff odds and strength of schedule.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def league_standings(ctx: Context, detail: Detail = "compact") -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            return render_standings(state, detail)

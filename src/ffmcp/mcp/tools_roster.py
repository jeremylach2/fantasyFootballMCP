"""Roster-perspective tools: my team, waiver targets, and single/comparative player lookups.
Adapters only: decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

from collections.abc import Sequence

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.models import Player
from ffmcp.domain.optimizer import projected_points
from ffmcp.domain.trades import apply_offer, marginal_value
from ffmcp.domain.valuation import value_over_replacement
from ffmcp.mcp._shared import (
    Detail,
    adapt_errors,
    app_context,
    enrich_team_roster,
    enrich_with_market,
    find_players_by_name,
    player_pool,
    require_single_match,
    resolve_my_team_id,
    resolve_week,
)
from ffmcp.render.tables import (
    WaiverTarget,
    render_compare_players,
    render_player_report,
    render_roster,
    render_waiver_targets,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
_MAX_WAIVER_LIMIT = 25
"""Hard cap regardless of the caller's ``limit``."""
_FREE_AGENT_CANDIDATES = 100
"""Free agents pulled from the provider before ranking by marginal value. Generous relative to
any realistic ``limit`` so a good pickup outside ESPN's own default ordering isn't missed."""
_NEAR_MISS_COUNT = 3
"""How many closest-but-negative candidates to show when nothing clears the bar. Enough to
answer "how close was it" without turning a bare-shelf week into a wall of numbers."""


def register_get_my_team(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Get My Team",
        description="Show my roster with projections, injuries and bye weeks.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def get_my_team(ctx: Context, week: int | None = None, detail: Detail = "compact") -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            resolved_week = await resolve_week(ctx, week, settings)
            teams = await app.league.get_teams()
            team_id = await resolve_my_team_id(ctx, teams)
            team = next(t for t in teams if t.team_id == team_id)
            if detail != "compact":
                team = await enrich_team_roster(ctx, team)
            return render_roster(team, resolved_week, detail)


def register_find_waiver_targets(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Find Waiver Targets",
        description="Rank available free agents by how much they'd help my team.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def find_waiver_targets(
        ctx: Context,
        position: str | None = None,
        limit: int = 10,
        detail: Detail = "compact",
    ) -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            week = await app.league.get_current_week()
            teams = await app.league.get_teams()
            team_id = await resolve_my_team_id(ctx, teams)
            team = next(t for t in teams if t.team_id == team_id)
            slots = settings.starting_slots
            roster_players = list(team.roster.players)

            candidates = await app.league.get_free_agents(
                week, size=_FREE_AGENT_CANDIDATES, position=position
            )
            gains = (
                (candidate, marginal_value(roster_players, candidate, slots))
                for candidate in candidates
            )
            ranked = sorted(gains, key=lambda pair: (-pair[1], pair[0].player_id))
            positive = [pair for pair in ranked if pair[1] > 0.0]

            def to_targets(
                pairs: Sequence[tuple[Player, float]], *, compute_drop: bool = True
            ) -> list[WaiverTarget]:
                result = []
                for player, gain in pairs:
                    drop: Player | None = None
                    if compute_drop:
                        _, drops = apply_offer(roster_players, [], [player], slots)
                        drop = drops[0] if drops else None
                    result.append(WaiverTarget(player, gain, drop))
                return result

            capped_limit = min(max(1, limit), _MAX_WAIVER_LIMIT)
            # Nothing clearing the bar is ambiguous between "the wire is empty" and "everything
            # scored -0.1", neither of which is visible from a bare "no targets" line, so the
            # near-misses stand in for the targets list when it would otherwise be empty.
            targets = to_targets(positive[:capped_limit])
            # A near-miss candidate is, by definition, worse than everything already on the
            # roster, so `apply_offer` would always name the candidate itself as the "drop",
            # reading as "pick up X, then drop X". There is no real drop decision here, so it is
            # never computed rather than shown as nonsense advice.
            near_misses = (
                [] if positive else to_targets(ranked[:_NEAR_MISS_COUNT], compute_drop=False)
            )

            if detail != "compact":
                display = targets or near_misses
                enriched = await enrich_with_market(ctx, [t.player for t in display])
                for target, player in zip(display, enriched, strict=True):
                    target.player = player
            return render_waiver_targets(
                targets,
                detail,
                limit=capped_limit,
                total_considered=len(positive),
                near_misses=near_misses,
            )


def register_player_report(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Player Report",
        description="Detailed outlook for one player: projection, value over replacement, market.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def player_report(ctx: Context, name: str, week: int | None = None) -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            resolved_week = await resolve_week(ctx, week, settings)
            current_week = await app.league.get_current_week()
            teams = await app.league.get_teams()
            pool = await player_pool(ctx, teams, resolved_week)

            player = require_single_match(find_players_by_name(pool, name), name)
            enriched = (await enrich_with_market(ctx, [player]))[0]
            weeks_remaining = max(1, settings.reg_season_weeks - current_week + 1)
            vor = value_over_replacement(enriched, settings, pool, weeks_remaining=1)
            return render_player_report(
                enriched,
                week=resolved_week,
                value_over_replacement=vor,
                weeks_remaining=weeks_remaining,
            )


def register_compare_players(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Compare Players",
        description="Compare players head-to-head for a start/sit decision.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def compare_players(ctx: Context, names: list[str], week: int | None = None) -> str:
        with adapt_errors():
            wanted = names[:4]
            if len(wanted) < 2:
                return "compare_players needs at least 2 player names."

            app = app_context(ctx)
            settings = await app.league.get_settings()
            resolved_week = await resolve_week(ctx, week, settings)
            teams = await app.league.get_teams()
            pool = await player_pool(ctx, teams, resolved_week)

            matched = [
                require_single_match(find_players_by_name(pool, name), name) for name in wanted
            ]
            enriched = await enrich_with_market(ctx, matched)

            rows = [
                (
                    player,
                    projected_points(player) or 0.0,
                    value_over_replacement(player, settings, pool, weeks_remaining=1),
                )
                for player in enriched
            ]
            rows.sort(key=lambda row: -row[2])
            best = rows[0][0]
            recommendation = f"Start {best.name} ({best.position} {best.pro_team})."
            return render_compare_players(recommendation, rows)

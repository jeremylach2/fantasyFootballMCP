"""Roster-perspective tools: my team, waiver targets, and single/comparative player lookups.
Adapters only: decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

from collections.abc import Sequence

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.models import Player, Team
from ffmcp.domain.optimizer import optimize, projected_points
from ffmcp.domain.schedule import (
    effective_weeks,
    remaining_week_weights,
    season_lineup_points,
    weekly_rate,
)
from ffmcp.domain.trades import apply_offer, marginal_value
from ffmcp.domain.usage import UsageProfile, backup_running_back
from ffmcp.domain.valuation import value_over_replacement
from ffmcp.mcp._shared import (
    Detail,
    adapt_errors,
    app_context,
    enrich_team_roster,
    enrich_with_market,
    find_players_by_name,
    load_game_lines,
    load_insights,
    load_usage,
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
    short_name,
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
_HANDCUFF_MIN_POINTS = 8.0
"""A starting running back projected below this is not worth insuring with a roster spot."""


def _handcuff_notes(
    team: Team,
    teams: Sequence[Team],
    slots: Sequence[str],
    profiles: dict[int, UsageProfile],
    free_agent_ids: set[int],
) -> list[str]:
    """For each running back this roster starts, who inherits his work if he goes down, and
    whether that player can actually be had. Worded as facts, not orders: whether a handcuff
    is worth a bench spot depends on what else that spot could hold, which this cannot see."""
    owner = {p.player_id: t for t in teams for p in (*t.roster.players, *t.roster.ir)}
    notes = []
    for starter in optimize(team.roster.players, slots).started:
        if starter.position != "RB" or (projected_points(starter) or 0.0) < _HANDCUFF_MIN_POINTS:
            continue
        backup = backup_running_back(starter.player_id, starter.pro_team, profiles)
        if backup is None:
            continue
        holder = owner.get(backup.player_id)
        if backup.player_id in free_agent_ids:
            where = "free agent"
        elif holder is not None and holder.team_id == team.team_id:
            where = "on your roster"
        elif holder is not None:
            where = f"on {holder.name}"
        else:
            where = "unrostered"
        snaps = f", {backup.snap_pct:.0f}% snaps" if backup.snap_pct is not None else ""
        notes.append(f"{short_name(starter.name)} -> {short_name(backup.name)} ({where}{snaps})")
    return notes


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
        description=(
            "Rank available free agents by how much they'd help my team this week (VAL) and "
            "over the rest of the season (ROS), and show who backs up my starting running backs."
        ),
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
            weights = remaining_week_weights(
                settings, week, playoff_probability=team.playoff_pct / 100.0
            )
            season_now = season_lineup_points(team.roster.players, settings.starting_slots, weights)
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
                    season_value: float | None = None
                    if compute_drop:
                        after, drops = apply_offer(roster_players, [], [player], slots)
                        drop = drops[0] if drops else None
                        # The whole move, add *and* drop, over the rest of the calendar: the
                        # roster after the claim against the roster as it stands.
                        season_value = season_lineup_points(after, slots, weights) - season_now
                    result.append(WaiverTarget(player, gain, drop, season_value))
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

            profiles, _ = await load_usage(ctx, settings)
            handcuffs: list[str] = []
            if profiles and position in (None, "RB"):
                available_rbs = await app.league.get_free_agents(
                    week, size=_FREE_AGENT_CANDIDATES, position="RB"
                )
                handcuffs = _handcuff_notes(
                    team, teams, slots, profiles, {p.player_id for p in available_rbs}
                )
            return render_waiver_targets(
                targets,
                detail,
                limit=capped_limit,
                total_considered=len(positive),
                near_misses=near_misses,
                handcuffs=handcuffs,
            )


def register_player_report(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Player Report",
        description=(
            "Detailed outlook for one player: projection and range, a second projection "
            "source, value over replacement, workload and luck, and his game's betting line."
        ),
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
            # Valued at his rest-of-season rate, so a bye this week is not a zero for the year.
            rates = {p.player_id: r for p in pool if (r := weekly_rate(p)) is not None}
            vor = value_over_replacement(
                enriched, settings, pool, weeks_remaining=1, projections=rates
            )
            my_team_id = app.settings.team_id
            my_team = next((t for t in teams if t.team_id == my_team_id), None)
            playoff_probability = my_team.playoff_pct / 100.0 if my_team else 0.5
            weights = remaining_week_weights(
                settings, current_week, playoff_probability=playoff_probability
            )
            insights = await load_insights(ctx, [player], resolved_week, settings)
            profiles, _ = await load_usage(ctx, settings)
            lines = await load_game_lines(ctx) if resolved_week == current_week else {}
            return render_player_report(
                enriched,
                week=resolved_week,
                value_over_replacement=vor,
                effective_weeks=effective_weeks(player, weights),
                sd=insights.sds.get(player.player_id),
                alternative=insights.alternative.get(player.player_id),
                usage=profiles.get(player.player_id),
                game_line=lines.get(player.pro_team),
            )


def register_compare_players(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Compare Players",
        description=(
            "Compare players head-to-head for a start/sit decision, with each one's floor, "
            "ceiling and team implied total."
        ),
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
            insights = await load_insights(ctx, matched, resolved_week, settings)
            current_week = await app.league.get_current_week()
            lines = await load_game_lines(ctx) if resolved_week == current_week else {}
            return render_compare_players(recommendation, rows, insights.sds, lines)

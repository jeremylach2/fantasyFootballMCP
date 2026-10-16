"""Roster-perspective tools: my team, waiver targets, the roster planner, and single and
comparative player lookups. Adapters only: decode args, call domain, render, return
(docs/architecture.md §2).
"""

from __future__ import annotations

from collections.abc import Sequence

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.models import FREE_AGENT_TEAM, Team
from ffmcp.domain.optimizer import is_ruled_out, optimize, projected_points
from ffmcp.domain.roster import (
    STASH_STATUSES,
    bye_weeks,
    drop_candidates,
    rank_pickups,
    rank_stashes,
)
from ffmcp.domain.schedule import (
    as_of_week,
    average_rate,
    effective_weeks,
    remaining_week_weights,
    weekly_rate,
)
from ffmcp.domain.usage import UsageProfile, backup_running_back
from ffmcp.domain.valuation import value_over_replacement
from ffmcp.errors import FFMCPError
from ffmcp.mcp._shared import (
    Detail,
    Insights,
    adapt_errors,
    app_context,
    attach_outlooks,
    enrich_team_roster,
    enrich_with_market,
    find_players_by_name,
    load_game_lines,
    load_history,
    load_insights,
    load_league_state,
    load_trade_values,
    load_upcoming_lines,
    load_usage,
    now,
    player_pool,
    playoff_probabilities,
    require_single_match,
    resolve_my_team_id,
    resolve_week,
    team_as_of_week,
    week_note,
)
from ffmcp.render.tables import (
    defense_matchups,
    render_compare_players,
    render_player_report,
    render_roster,
    render_roster_plan,
    render_waiver_targets,
    render_week_recap,
    short_name,
    start_sit_sentence,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
_MAX_WAIVER_LIMIT = 25
"""Hard cap regardless of the caller's ``limit``."""
_FREE_AGENT_CANDIDATES = 100
"""Free agents pulled from the provider before ranking. Generous relative to any realistic
``limit`` so a good pickup outside ESPN's own default ordering isn't missed."""
_STASH_COUNT = 3
_DROP_COUNT = 5
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
        description=(
            "Show my roster with projections, injuries, byes and lineup locks. A past week "
            "shows what I started and what each player scored; a future week uses season "
            "averages with that week's byes."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def get_my_team(ctx: Context, week: int | None = None, detail: Detail = "compact") -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            current_week = await app.league.get_current_week()
            resolved_week = await resolve_week(ctx, week, settings)
            teams = await app.league.get_teams()
            team_id = await resolve_my_team_id(ctx, teams)
            team = next(t for t in teams if t.team_id == team_id)
            if resolved_week < current_week:
                return await _week_recap(ctx, team, resolved_week, settings.starting_slots)
            if detail != "compact":
                team = await enrich_team_roster(ctx, team)
            at = now(ctx)
            locked = {
                p.player_id
                for p in (*team.roster.players, *team.roster.ir)
                if p.kickoff is not None and p.kickoff <= at
            }
            return render_roster(
                team_as_of_week(team, resolved_week, current_week),
                resolved_week,
                detail,
                note=week_note(resolved_week, current_week),
                locked=locked if resolved_week == current_week else (),
            )


async def _week_recap(ctx: Context, team: Team, week: int, slots: Sequence[str]) -> str:
    history, caveat = await load_history(ctx)
    rows = [r for r in history if r.week == week and r.fantasy_team_id == team.team_id]
    if not rows:
        raise FFMCPError(caveat or f"No lineup is recorded for week {week}.")
    eligible = [r for r in rows if r.slot != "IR"]
    return render_week_recap(
        team,
        week,
        rows,
        best_projected=optimize(
            [r.as_player(hindsight=False) for r in eligible], slots
        ).projected_points,
        best_actual=optimize([r.as_player() for r in eligible], slots).projected_points,
    )


def register_find_waiver_targets(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Find Waiver Targets",
        description=(
            "Rank free-agent moves (add plus the drop that costs least) by what they add to my "
            "rest of season (ROS), with this week's part (VAL) beside it; injured players worth "
            "stashing on IR; who backs up my starting running backs; and, for "
            "position='D/ST', each defense's next opponents' implied points."
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
            state = await load_league_state(ctx)
            settings, week = state.settings, state.current_week
            team_id = await resolve_my_team_id(ctx, state.teams)
            team = state.team(team_id)
            slots = settings.starting_slots
            roster = list(team.roster.players)
            wanted = position.upper() if position else None
            probabilities = await playoff_probabilities(ctx, state)
            weights = remaining_week_weights(
                settings, week, playoff_probability=probabilities.get(team_id, 0.5)
            )

            free_agents = await app.league.get_free_agents(
                week, size=_FREE_AGENT_CANDIDATES, position=wanted
            )
            # A player no NFL team employs is not a pickup, whatever his projection says.
            employed = [p for p in free_agents if p.pro_team != FREE_AGENT_TEAM]
            candidates = await attach_outlooks(ctx, employed, settings)
            stash_pool = [p for p in candidates if (p.injury_status or "") in STASH_STATUSES]
            movable = [p for p in candidates if (p.injury_status or "") not in STASH_STATUSES]

            capped_limit = min(max(1, limit), _MAX_WAIVER_LIMIT)
            pickups, near_misses = rank_pickups(roster, movable, slots, weights, limit=capped_limit)
            stashes = rank_stashes(roster, stash_pool, slots, weights, limit=_STASH_COUNT)
            open_ir = max(0, settings.slot_counts.get("IR", 0) - len(team.roster.ir))

            if detail != "compact":
                shown = [*pickups, *near_misses]
                enriched = await enrich_with_market(ctx, [p.player for p in shown])
                by_id = {p.player_id: p for p in enriched}
                pickups = [
                    p.model_copy(update={"player": by_id[p.player.player_id]}) for p in pickups
                ]
                near_misses = [
                    p.model_copy(update={"player": by_id[p.player.player_id]}) for p in near_misses
                ]

            profiles, _ = await load_usage(ctx, settings)
            handcuffs: list[str] = []
            if profiles and wanted in (None, "RB"):
                available_rbs = (
                    free_agents
                    if wanted == "RB"
                    else await app.league.get_free_agents(
                        week, size=_FREE_AGENT_CANDIDATES, position="RB"
                    )
                )
                handcuffs = _handcuff_notes(
                    team, state.teams, slots, profiles, {p.player_id for p in available_rbs}
                )
            matchups = None
            if wanted == "D/ST":
                defenses = [
                    *(p for p in roster if p.position == "D/ST"),
                    *(p.player for p in [*pickups, *near_misses]),
                ]
                matchups = defense_matchups(defenses, await load_upcoming_lines(ctx))
            return render_waiver_targets(
                pickups,
                detail,
                limit=capped_limit,
                near_misses=near_misses,
                stashes=stashes,
                open_ir_slots=open_ir,
                handcuffs=handcuffs,
                matchups=matchups,
            )


def register_roster_planner(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Roster Planner",
        description=(
            "Plan the rest of my season: which weeks byes take regular starters out of my "
            "lineup (and what it costs), and which bench players I could drop at least cost."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def roster_planner(ctx: Context) -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            settings, week = state.settings, state.current_week
            team_id = await resolve_my_team_id(ctx, state.teams)
            team = state.team(team_id)
            slots = settings.starting_slots
            roster = await attach_outlooks(ctx, list(team.roster.players), settings)
            probabilities = await playoff_probabilities(ctx, state)
            weights = remaining_week_weights(
                settings, week, playoff_probability=probabilities.get(team_id, 0.5)
            )
            byes = bye_weeks(roster, slots, range(week, settings.reg_season_weeks + 1), week)
            drops = drop_candidates(roster, slots, weights, week, limit=_DROP_COUNT)
            return render_roster_plan(byes, drops, typical=byes[0].typical if byes else None)


def register_player_report(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Player Report",
        description=(
            "Detailed outlook for one player: projection and range, a second projection "
            "source, injury return, value over replacement, workload and luck, his game's "
            "betting line, and his trade-market value."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def player_report(ctx: Context, name: str, week: int | None = None) -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            settings, current_week = state.settings, state.current_week
            resolved_week = await resolve_week(ctx, week, settings)
            if resolved_week < current_week:
                raise FFMCPError(
                    f"player_report looks ahead; for week {resolved_week}, use "
                    f"get_my_team(week={resolved_week})."
                )
            pool = await player_pool(ctx, state.teams, current_week)
            found = require_single_match(find_players_by_name(pool, name), name, pool)
            [found] = await attach_outlooks(ctx, [found], settings)
            enriched = (await enrich_with_market(ctx, [found]))[0]
            player = as_of_week(enriched, resolved_week, current_week)

            weeks = range(current_week, settings.reg_season_weeks + 1)
            # Valued at his rest-of-season rate, so a bye this week is not a zero for the year
            # and an injured player's expected return counts.
            rates = {p.player_id: r for p in pool if (r := weekly_rate(p)) is not None}
            own_rate = average_rate(found, weeks)
            if own_rate is not None:
                rates[found.player_id] = own_rate
            vor = value_over_replacement(
                player, settings, pool, weeks_remaining=1, projections=rates
            )
            my_team_id = app_context(ctx).settings.team_id
            probabilities = await playoff_probabilities(ctx, state)
            weights = remaining_week_weights(
                settings,
                current_week,
                playoff_probability=probabilities.get(my_team_id or -1, 0.5),
            )
            is_current = resolved_week == current_week
            insights = (
                await load_insights(ctx, [found], resolved_week, settings)
                if is_current
                else Insights()
            )
            profiles, _ = await load_usage(ctx, settings)
            lines = await load_game_lines(ctx) if is_current else {}
            values, _ = await load_trade_values(ctx, settings)
            return render_player_report(
                player,
                week=resolved_week,
                value_over_replacement=vor,
                effective_weeks=effective_weeks(found, weights),
                sd=insights.sds.get(found.player_id),
                alternative=insights.alternative.get(found.player_id),
                usage=profiles.get(found.player_id),
                game_line=lines.get(found.pro_team),
                trade_value=values.get(found.player_id) if values else None,
                note=week_note(resolved_week, current_week),
            )


def register_compare_players(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Compare Players",
        description=(
            "Compare players for a start/sit decision: which of them my optimal lineup starts "
            "(and in which slot) and which sit, with each one's floor, ceiling and team "
            "implied total."
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
            current_week = await app.league.get_current_week()
            resolved_week = await resolve_week(ctx, week, settings)
            if resolved_week < current_week:
                raise FFMCPError(
                    "compare_players looks ahead; use get_my_team(week=N) for a past week."
                )
            teams = await app.league.get_teams()
            pool = await player_pool(ctx, teams, current_week)
            matched = [
                require_single_match(find_players_by_name(pool, name), name, pool)
                for name in wanted
            ]
            enriched = await enrich_with_market(ctx, matched)
            compared = [as_of_week(p, resolved_week, current_week) for p in enriched]
            compared_ids = {p.player_id for p in compared}

            # The question is "which of these do I start", and the answer depends on everyone
            # else on the roster: two backs and a tight end compared for RB, RB and FLEX all
            # start, while a lone FLEX decision starts one. So the optimizer answers it.
            my_team_id = await resolve_my_team_id(ctx, teams)
            my_team = next(t for t in teams if t.team_id == my_team_id)
            others = [
                as_of_week(p, resolved_week, current_week)
                for p in my_team.roster.players
                if p.player_id not in compared_ids
            ]
            lineup = optimize([*others, *compared], settings.starting_slots)
            starts = [
                (slot.player, slot.slot)
                for slot in lineup.slots
                if slot.player is not None and slot.player.player_id in compared_ids
            ]
            started_ids = {p.player_id for p, _ in starts}
            sits = [p for p in compared if p.player_id not in started_ids]

            rows = [
                (
                    player,
                    projected_points(player) or 0.0,
                    value_over_replacement(player, settings, pool, weeks_remaining=1),
                )
                for player in compared
            ]
            rows.sort(key=lambda row: (row[0].player_id not in started_ids, -row[1]))
            is_current = resolved_week == current_week
            insights = (
                await load_insights(ctx, compared, resolved_week, settings)
                if is_current
                else Insights()
            )
            lines = await load_game_lines(ctx) if is_current else {}
            out_notes = [
                f"{p.name} is {(p.injury_status or 'out').replace('_', ' ').title()}."
                for p in compared
                if is_ruled_out(p)
            ]
            note = " ".join(filter(None, [week_note(resolved_week, current_week), *out_notes]))
            return render_compare_players(
                start_sit_sentence(starts, sits), rows, insights.sds, lines, note or None
            )

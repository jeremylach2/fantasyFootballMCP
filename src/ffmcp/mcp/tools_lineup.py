"""Lineup tools: the headline optimizer and the weekly matchup breakdown. Adapters only:
decode args, call domain, render, return (docs/architecture.md §2). No arithmetic lives here
beyond picking which of the domain's own numbers to show.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.models import LeagueState, Lineup, Player, RosterSlot
from ffmcp.domain.models import Swap as DomainSwap
from ffmcp.domain.optimizer import diff, optimize, projected_points
from ffmcp.domain.risk import best_lineup_for_matchup
from ffmcp.domain.simulate import (
    DEFAULT_DELTA_SIMS,
    LineupScenario,
    PlayerSds,
    lineup_score_model,
    matchup_win_probability,
    player_score_sd,
    week_stakes,
)
from ffmcp.domain.simulate import win_prob_delta as compute_win_prob_delta
from ffmcp.mcp._shared import (
    Detail,
    adapt_errors,
    app_context,
    league_players,
    load_insights,
    load_league_state,
    resolve_my_team_id,
    resolve_week,
)
from ffmcp.render.detail import LineupAdvice, build_lineup_advice, strategy_note
from ffmcp.render.tables import render_matchup

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)


def _current_lineup(
    team_roster_slots: Sequence[RosterSlot],
    bench: Sequence[Player],
    projections: Mapping[int, float] | None = None,
) -> Lineup:
    points = sum(
        (projected_points(slot.player, projections) or 0.0)
        for slot in team_roster_slots
        if slot.player is not None
    )
    return Lineup(
        slots=tuple(team_roster_slots), projected_points=round(points, 4), bench=tuple(bench)
    )


def _live_projections(players: Sequence[Player]) -> dict[int, float] | None:
    """A ``{player_id: points}`` map built from what has actually happened this week, or
    ``None`` if nobody's game has kicked off yet (the pregame section already covers that case).

    A player whose game is in progress or final gets ``live_points``; anyone whose game hasn't
    started yet (e.g. Monday-night players read Sunday morning) falls back to the pregame
    projection, so the recap never has to guess at a score that doesn't exist.
    """
    if not any(player.live_points is not None for player in players):
        return None
    projections: dict[int, float] = {}
    for player in players:
        if player.live_points is not None:
            projections[player.player_id] = player.live_points
        elif player.projection is not None:
            projections[player.player_id] = player.projection.points
    return projections


_STAKES_SIMS = 2 * DEFAULT_DELTA_SIMS
"""Enough seasons to pin a win/loss playoff-odds gap to about a point, in well under a second."""


def _opponent_id(state: LeagueState, team_id: int, week: int) -> int | None:
    for matchup in state.matchups:
        if matchup.week != week or matchup.is_playoff:
            continue
        if matchup.home_team_id == team_id:
            return matchup.away_team_id
        if matchup.away_team_id == team_id:
            return matchup.home_team_id
    return None


def _positional_edges(mine: Lineup, theirs: Lineup) -> list[tuple[str, float]]:
    """Total projected points per position, my optimal lineup minus theirs. Sorted by the size
    of the edge, largest first, for ``analyze_matchup`` to show the top 3."""

    def totals(lineup: Lineup) -> dict[str, float]:
        result: dict[str, float] = {}
        for player in lineup.started:
            result[player.position] = result.get(player.position, 0.0) + (
                projected_points(player) or 0.0
            )
        return result

    mine_totals, their_totals = totals(mine), totals(theirs)
    positions = {*mine_totals, *their_totals}
    deltas = [(pos, mine_totals.get(pos, 0.0) - their_totals.get(pos, 0.0)) for pos in positions]
    deltas.sort(key=lambda pair: -abs(pair[1]))
    return deltas


def _swing_player(
    mine: Lineup, theirs: Lineup, sds: PlayerSds | None = None
) -> tuple[Player | None, float]:
    """The single highest-variance starter across both lineups: the player whose outcome
    swings this specific matchup the most, under the same sd model ``domain.simulate`` uses
    for the season simulation."""
    best_player: Player | None = None
    best_sd = -1.0
    for player in (*mine.started, *theirs.started):
        points = projected_points(player) or 0.0
        sd = player_score_sd(player, points, sds)
        if sd > best_sd:
            best_player, best_sd = player, sd
    return best_player, max(best_sd, 0.0)


def register_optimize_lineup(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Optimize Lineup",
        description=(
            "Find the lineup most likely to win this week's matchup (or, with "
            "objective='points', the highest-projected one) and the swaps to get there."
        ),
        annotations=_READ_ONLY,
    )
    async def optimize_lineup(
        ctx: Context,
        week: int | None = None,
        objective: Literal["win_probability", "points"] = "win_probability",
    ) -> LineupAdvice:
        with adapt_errors():
            state = await load_league_state(ctx)
            resolved_week = await resolve_week(ctx, week, state.settings)
            team_id = await resolve_my_team_id(ctx, state.teams)
            team = state.team(team_id)
            slots = state.settings.starting_slots
            insights = await load_insights(
                ctx, league_players(state), resolved_week, state.settings
            )

            current = _current_lineup(team.roster.slots, team.roster.bench)
            optimal = optimize(team.roster.players, slots)
            win_probability: float | None = None
            strategy: str | None = None
            tilt = 0.0
            opponent_id = _opponent_id(state, team_id, resolved_week)
            if opponent_id is not None:
                opponent_lineup = optimize(state.team(opponent_id).roster.players, slots)
                opponent = lineup_score_model(opponent_id, opponent_lineup, sds=insights.sds)
                choice = best_lineup_for_matchup(
                    team_id, team.roster.players, slots, opponent, sds=insights.sds
                )
                win_probability = choice.points_win_probability
                if objective == "win_probability":
                    optimal, win_probability, tilt = (
                        choice.lineup,
                        choice.win_probability,
                        choice.tilt,
                    )
                    strategy = strategy_note(
                        my_projected=choice.points_lineup.projected_points,
                        opponent_projected=opponent.mean,
                        tilt=choice.tilt,
                        points_win_probability=choice.points_win_probability,
                        win_probability=choice.win_probability,
                        points_cost=choice.points_lineup.projected_points
                        - choice.lineup.projected_points,
                    )
            swaps = diff(current, optimal)

            delta = 0.0
            if resolved_week in _remaining_weeks(state):
                delta = compute_win_prob_delta(
                    state,
                    LineupScenario(team_id=team_id, week=resolved_week, lineup=current),
                    LineupScenario(team_id=team_id, week=resolved_week, lineup=optimal),
                    player_sds=insights.sds,
                )

            actual_current: Lineup | None = None
            actual_optimal: Lineup | None = None
            actual_swaps: list[DomainSwap] = []
            # The roster's ``live_points`` always reflect the provider's current week, so the
            # recap only makes sense when the caller is asking about the week in progress.
            if resolved_week == state.current_week:
                live_map = _live_projections(team.roster.players)
                if live_map is not None:
                    actual_current = _current_lineup(team.roster.slots, team.roster.bench, live_map)
                    actual_optimal = optimize(team.roster.players, slots, live_map)
                    actual_swaps = diff(actual_current, actual_optimal, live_map)

            return build_lineup_advice(
                week=resolved_week,
                current=current,
                optimal=optimal,
                playoff_odds_delta=delta,
                swaps=swaps,
                actual_current=actual_current,
                actual_optimal=actual_optimal,
                actual_swaps=actual_swaps,
                win_probability=win_probability,
                strategy=strategy,
                tilt=tilt,
                extra_caveats=insights.caveats,
            )


def register_analyze_matchup(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Analyze Matchup",
        description="Break down this week's head-to-head matchup and win probability.",
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def analyze_matchup(
        ctx: Context, week: int | None = None, detail: Detail = "compact"
    ) -> str:
        with adapt_errors():
            app = app_context(ctx)
            state = await load_league_state(ctx)
            resolved_week = await resolve_week(ctx, week, state.settings)
            team_id = await resolve_my_team_id(ctx, state.teams)

            matchups = await app.league.get_matchups(resolved_week)
            my_matchup = next(
                (m for m in matchups if team_id in (m.home_team_id, m.away_team_id)), None
            )
            my_team = state.team(team_id)
            if my_matchup is None:
                return f"No matchup found for {my_team.name} in week {resolved_week}."
            i_am_home = my_matchup.home_team_id == team_id
            opponent_id = my_matchup.away_team_id if i_am_home else my_matchup.home_team_id
            if opponent_id is None:
                return f"Week {resolved_week} is a bye for {my_team.name}."
            my_live_score = my_matchup.home_score if i_am_home else my_matchup.away_score
            opp_live_score = my_matchup.away_score if i_am_home else my_matchup.home_score

            opp_team = state.team(opponent_id)
            slots = state.settings.starting_slots
            insights = await load_insights(
                ctx, league_players(state), resolved_week, state.settings
            )
            my_lineup = optimize(my_team.roster.players, slots)
            opp_lineup = optimize(opp_team.roster.players, slots)
            my_model = lineup_score_model(team_id, my_lineup, sds=insights.sds)
            opp_model = lineup_score_model(opponent_id, opp_lineup, sds=insights.sds)
            win_probability = matchup_win_probability(my_model, opp_model)
            all_edges = _positional_edges(my_lineup, opp_lineup)
            edges = all_edges if detail != "compact" else all_edges[:3]
            swing_player, swing_sd = _swing_player(my_lineup, opp_lineup, insights.sds)
            stakes = await asyncio.to_thread(
                week_stakes,
                state,
                team_id,
                resolved_week,
                _STAKES_SIMS,
                player_sds=insights.sds,
            )

            return render_matchup(
                week=resolved_week,
                my_team=my_team,
                opp_team=opp_team,
                my_projected=my_lineup.projected_points,
                opp_projected=opp_lineup.projected_points,
                my_live_score=my_live_score,
                opp_live_score=opp_live_score,
                win_probability=win_probability,
                edges=edges,
                swing_player=swing_player,
                swing_sd=swing_sd,
                stakes=stakes,
            )


def _remaining_weeks(state: LeagueState) -> frozenset[int]:
    return frozenset(
        m.week
        for m in state.matchups
        if state.current_week <= m.week <= state.settings.reg_season_weeks
    )

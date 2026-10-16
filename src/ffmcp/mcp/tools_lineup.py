"""Lineup tools: the headline optimizer and the weekly matchup breakdown. Adapters only:
decode args, call domain, render, return (docs/architecture.md §2). No arithmetic lives here
beyond picking which of the domain's own numbers to show.

Both honour ``week`` in all three directions. The current week is the live roster. A later week
is the same roster at season-average projections with that week's byes (``domain.schedule``),
said so in a caveat. A completed week is the league's own record of it: what was started, what
the projections of the day recommended, and what would have been best in hindsight.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from typing import Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.live import any_kicked_off, live_score_model
from ffmcp.domain.models import LeagueState, Lineup, Player, PlayerWeek, RosterSlot, Team
from ffmcp.domain.models import Swap as DomainSwap
from ffmcp.domain.optimizer import (
    diff,
    is_game_time_decision,
    optimize,
    projected_points,
    slot_accepts,
)
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
from ffmcp.domain.variance import expected_out_by_primary
from ffmcp.errors import FFMCPError
from ffmcp.mcp._shared import (
    Detail,
    Insights,
    adapt_errors,
    app_context,
    enrich_with_market,
    league_players,
    lineup_locks,
    load_history,
    load_insights,
    load_league_state,
    now,
    resolve_my_team_id,
    resolve_week,
    team_as_of_week,
    week_note,
)
from ffmcp.render.detail import LineupAdvice, build_lineup_advice, strategy_note
from ffmcp.render.tables import render_matchup, render_matchup_result, short_name, slot_label

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
        elif (points := projected_points(player)) is not None:
            projections[player.player_id] = points
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


def _zeroed(players: Sequence[Player], alternative: Mapping[int, float]) -> list[Player]:
    """Players ESPN has projected for zero while the second source expects them to play: a
    game-time decision the optimizer, which uses ESPN, would otherwise bench in silence."""
    return [
        player
        for player in players
        if is_game_time_decision(player)
        and expected_out_by_primary(projected_points(player), alternative.get(player.player_id))
    ]


def _zeroed_caveats(players: Sequence[Player], alternative: Mapping[int, float]) -> list[str]:
    """One line for each of ``_zeroed``'s players, with the injury where it is known."""
    caveats = []
    for player in players:
        other = alternative.get(player.player_id)
        if other is not None:
            status = (player.injury_status or "no status").replace("_", " ").title()
            detail = player.market.injury_detail if player.market else None
            if detail:
                status += f", {detail.lower()}"
            caveats.append(
                f"{player.name} ({status}): ESPN projects 0, Sleeper {other:.1f}; if he is "
                "active, re-run this. Inactives post 90 min before kickoff."
            )
    return caveats


def _late_swap_note(lineup: Lineup, locked: Collection[int]) -> str | None:
    """Advice for where to *slot* the recommended starters, not who: the flex should hold the
    latest kickoff among the players who could fill it, so that if he ends up inactive, a
    replacement can still be swapped in. Returns a note when the optimal lineup has an
    earlier game in the flex than in a position slot it could trade places with."""
    slots = list(lineup.slots)
    for flex_index, flex in enumerate(slots):
        early = flex.player
        if early is None or early.kickoff is None or slot_label(flex.slot) == flex.slot:
            continue
        if early.player_id in locked:
            continue
        for other in slots:
            late = other.player
            if (
                late is None
                or other is flex
                or late.kickoff is None
                or late.player_id in locked
                or late.kickoff <= early.kickoff
                or not slot_accepts(other.slot, early)
                or not slot_accepts(slots[flex_index].slot, late)
            ):
                continue
            return (
                f"Slot {short_name(late.name)} at FLEX and {short_name(early.name)} at "
                f"{slot_label(other.slot)}: the later game in the flex keeps a late swap open."
            )
    return None


def _past_week_advice(
    team_id: int, week: int, history: Sequence[PlayerWeek], slots: Sequence[str]
) -> LineupAdvice:
    """A completed week, from the league's record of it."""
    rows = [
        r for r in history if r.week == week and r.fantasy_team_id == team_id and r.slot != "IR"
    ]
    if not rows:
        raise FFMCPError(f"No lineup is recorded for week {week}.")
    then = [r.as_player(hindsight=False) for r in rows]
    actual = [r.as_player(hindsight=True) for r in rows]

    def started(players: Sequence[Player]) -> Lineup:
        by_id = {p.player_id: p for p in players}
        chosen = [r for r in rows if r.started]
        return Lineup(
            slots=tuple(RosterSlot(slot=r.slot or "", player=by_id[r.player_id]) for r in chosen),
            projected_points=round(
                sum(projected_points(by_id[r.player_id]) or 0.0 for r in chosen), 4
            ),
        )

    current, actual_current = started(then), started(actual)
    optimal = optimize(then, slots)
    actual_optimal = optimize(actual, slots)
    return build_lineup_advice(
        week=week,
        current=current,
        optimal=optimal,
        playoff_odds_delta=0.0,
        swaps=diff(current, optimal),
        actual_current=actual_current,
        actual_optimal=actual_optimal,
        actual_swaps=diff(actual_current, actual_optimal),
        extra_caveats=[
            f"Week {week} is complete: swaps are what that week's projections called for; "
            "'actual' compares what was started with the best lineup in hindsight."
        ],
    )


def _week_team(state: LeagueState, team_id: int, week: int) -> Team:
    return team_as_of_week(state.team(team_id), week, state.current_week)


def register_optimize_lineup(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Optimize Lineup",
        description=(
            "Find the lineup most likely to win this week's matchup (or, with "
            "objective='points', the highest-projected one) and the swaps to get there. Players "
            "whose games have started stay put. A past week returns a hindsight recap; a future "
            "week uses season averages with byes."
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
            slots = state.settings.starting_slots
            if resolved_week < state.current_week:
                history, caveat = await load_history(ctx)
                if not history:
                    raise FFMCPError(caveat or f"No history is available for week {resolved_week}.")
                return _past_week_advice(team_id, resolved_week, history, slots)

            is_current = resolved_week == state.current_week
            team = _week_team(state, team_id, resolved_week)
            insights = (
                await load_insights(ctx, league_players(state), resolved_week, state.settings)
                if is_current
                else Insights()
            )
            pinned, frozen = (
                lineup_locks(team.roster.slots, team.roster.bench, slots, now(ctx))
                if is_current
                else ({}, set())
            )
            locked = {*pinned, *frozen}

            current = _current_lineup(team.roster.slots, team.roster.bench)
            optimal = optimize(team.roster.players, slots, pinned=pinned, frozen=frozen)
            win_probability: float | None = None
            strategy: str | None = None
            tilt = 0.0
            opponent_id = _opponent_id(state, team_id, resolved_week)
            if opponent_id is not None:
                opponent_team = _week_team(state, opponent_id, resolved_week)
                opponent_lineup = optimize(opponent_team.roster.players, slots)
                opponent = lineup_score_model(opponent_id, opponent_lineup, sds=insights.sds)
                choice = best_lineup_for_matchup(
                    team_id,
                    team.roster.players,
                    slots,
                    opponent,
                    sds=insights.sds,
                    pinned=pinned,
                    frozen=frozen,
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
                delta = await asyncio.to_thread(
                    compute_win_prob_delta,
                    state,
                    LineupScenario(team_id=team_id, week=resolved_week, lineup=current),
                    LineupScenario(team_id=team_id, week=resolved_week, lineup=optimal),
                    player_sds=insights.sds,
                )

            actual_current: Lineup | None = None
            actual_optimal: Lineup | None = None
            actual_swaps: list[DomainSwap] = []
            caveats: list[str] = []
            if is_current:
                live_map = _live_projections(team.roster.players)
                if live_map is not None:
                    actual_current = _current_lineup(team.roster.slots, team.roster.bench, live_map)
                    actual_optimal = optimize(team.roster.players, slots, live_map)
                    actual_swaps = diff(actual_current, actual_optimal, live_map)
                zeroed = await enrich_with_market(
                    ctx, _zeroed(team.roster.players, insights.alternative)
                )
                caveats.extend(_zeroed_caveats(zeroed, insights.alternative))
                if locked:
                    caveats.append(
                        f"{len(locked)} of your players' games have started; they stay where "
                        "they are."
                    )
                late_swap = _late_swap_note(optimal, locked)
                if late_swap:
                    caveats.append(late_swap)
            else:
                caveats.append(week_note(resolved_week, state.current_week) or "")
            caveats.extend(insights.caveats)

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
                extra_caveats=[c for c in caveats if c],
            )


def register_analyze_matchup(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Analyze Matchup",
        description=(
            "Break down a week's head-to-head matchup: win probability, positional edges, the "
            "swing player, and what the game is worth to my playoff odds. A past week returns "
            "the final."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def analyze_matchup(
        ctx: Context, week: int | None = None, detail: Detail = "compact"
    ) -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            resolved_week = await resolve_week(ctx, week, state.settings)
            team_id = await resolve_my_team_id(ctx, state.teams)
            my_team = state.team(team_id)
            if resolved_week < state.current_week:
                return await _past_matchup(ctx, state, team_id, resolved_week)

            opponent_id = _opponent_id(state, team_id, resolved_week)
            my_matchup = next(
                (
                    m
                    for m in state.matchups
                    if m.week == resolved_week and team_id in (m.home_team_id, m.away_team_id)
                ),
                None,
            )
            if my_matchup is None:
                return f"No matchup found for {my_team.name} in week {resolved_week}."
            if opponent_id is None:
                return f"Week {resolved_week} is a bye for {my_team.name}."
            i_am_home = my_matchup.home_team_id == team_id
            my_live_score = my_matchup.home_score if i_am_home else my_matchup.away_score
            opp_live_score = my_matchup.away_score if i_am_home else my_matchup.home_score

            is_current = resolved_week == state.current_week
            mine_view = _week_team(state, team_id, resolved_week)
            theirs_view = _week_team(state, opponent_id, resolved_week)
            slots = state.settings.starting_slots
            insights = (
                await load_insights(ctx, league_players(state), resolved_week, state.settings)
                if is_current
                else Insights()
            )
            my_lineup = optimize(mine_view.roster.players, slots)
            opp_lineup = optimize(theirs_view.roster.players, slots)
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
            live: tuple[float, float, float] | None = None
            at = now(ctx)
            mine_set = [s.player for s in my_team.roster.slots if s.player is not None]
            theirs_set = [
                s.player for s in state.team(opponent_id).roster.slots if s.player is not None
            ]
            if is_current and any_kicked_off([*mine_set, *theirs_set], at):
                # Once games start, it is the lineups as set that are playing, and the score so
                # far that counts: the pregame number above stops being the question.
                my_live, my_left = live_score_model(team_id, mine_set, at, insights.sds)
                opp_live, opp_left = live_score_model(opponent_id, theirs_set, at, insights.sds)
                live = (matchup_win_probability(my_live, opp_live), my_left, opp_left)

            return render_matchup(
                week=resolved_week,
                my_team=my_team,
                opp_team=state.team(opponent_id),
                my_projected=my_lineup.projected_points,
                opp_projected=opp_lineup.projected_points,
                my_live_score=my_live_score,
                opp_live_score=opp_live_score,
                win_probability=win_probability,
                edges=edges,
                swing_player=swing_player,
                swing_sd=swing_sd,
                stakes=stakes,
                note=week_note(resolved_week, state.current_week),
                live=live,
            )


async def _past_matchup(ctx: Context, state: LeagueState, team_id: int, week: int) -> str:
    matchups = await app_context(ctx).league.get_matchups(week)
    mine = next((m for m in matchups if team_id in (m.home_team_id, m.away_team_id)), None)
    my_team = state.team(team_id)
    if mine is None or None in (mine.home_team_id, mine.away_team_id):
        return f"{my_team.name} had no game in week {week}."
    home = mine.home_team_id == team_id
    opponent_id = mine.away_team_id if home else mine.home_team_id
    assert opponent_id is not None
    history, _ = await load_history(ctx)

    def projected(tid: int) -> float | None:
        rows = [r for r in history if r.week == week and r.fantasy_team_id == tid and r.started]
        return round(sum(r.projected or 0.0 for r in rows), 1) if rows else None

    return render_matchup_result(
        week=week,
        my_team=my_team,
        opp_team=state.team(opponent_id),
        my_score=mine.home_score if home else mine.away_score,
        opp_score=mine.away_score if home else mine.home_score,
        my_projected=projected(team_id),
        opp_projected=projected(opponent_id),
    )


def _remaining_weeks(state: LeagueState) -> frozenset[int]:
    return frozenset(
        m.week
        for m in state.matchups
        if state.current_week <= m.week <= state.settings.reg_season_weeks
    )

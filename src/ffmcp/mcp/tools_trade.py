"""Trade tools: search across the whole league and price one named offer from both sides.
Adapters only: decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.league_intel import manager_reports
from ffmcp.domain.models import LeagueState, Player, TradeOffer
from ffmcp.domain.trades import evaluate, positional_impact, search, with_playoff_odds
from ffmcp.errors import TradePartnerAmbiguous
from ffmcp.mcp._shared import (
    adapt_errors,
    find_players_by_name,
    load_history,
    load_league_state,
    load_trade_values,
    now,
    playoff_probabilities,
    require_single_match,
    resolve_my_team_id,
    week_weights_by_team,
)
from ffmcp.render.detail import TradeVerdict, build_trade_verdict, render_positional_impact
from ffmcp.render.tables import deadline_note, render_trade_line, render_trades, team_label

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
_MAX_RESULTS = 10
"""Hard cap regardless of the caller's ``max_results``."""


def _resolve_named(roster_players: Sequence[Player], names: Sequence[str]) -> list[Player]:
    return [
        require_single_match(find_players_by_name(roster_players, name), name, roster_players)
        for name in names
    ]


def _infer_partner(state: LeagueState, my_team_id: int, get_names: Sequence[str]) -> int:
    candidates = [
        team.team_id
        for team in state.teams
        if team.team_id != my_team_id
        and all(len(find_players_by_name(team.roster.players, name)) == 1 for name in get_names)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise TradePartnerAmbiguous()


def _partner_label(name: str, tags: Sequence[str]) -> str:
    """A partner's name with what the league's history says about how they manage. An
    unlucky or inattentive manager's record understates their roster, which is exactly the
    manager most likely to accept a deal that looks like a fix."""
    label = team_label(name, 20)
    return f"{label} [{', '.join(tags)}]" if tags else label


def register_find_trades(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Find Trades",
        description=(
            "Suggest trades that help my roster over the rest of the season (byes and playoff "
            "weeks included) and that the other manager would plausibly accept: both lineups "
            "gain, and the market (FantasyCalc) calls it fair. Kickers and defenses only when "
            "named in positions_wanted."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def find_trades(
        ctx: Context,
        max_results: int = 5,
        partner_team_id: int | None = None,
        positions_wanted: list[str] | None = None,
    ) -> str:
        with adapt_errors():
            state = await load_league_state(ctx)
            my_team_id = await resolve_my_team_id(ctx, state.teams)
            capped = min(max(1, max_results), _MAX_RESULTS)
            slots = state.settings.starting_slots
            my_roster = list(state.team(my_team_id).roster.players)
            weights = week_weights_by_team(state, await playoff_probabilities(ctx, state))
            values, values_caveat = await load_trade_values(ctx, state.settings)
            history, _ = await load_history(ctx)
            tags = {r.team_id: r.tags for r in manager_reports(state, history)} if history else {}
            loop = asyncio.get_running_loop()

            def report(done: int, total: int) -> None:
                # Called from the worker thread `search` runs in. `ctx.report_progress` is a
                # coroutine, so it is handed back to the event loop rather than awaited here.
                asyncio.run_coroutine_threadsafe(ctx.report_progress(done, total), loop)

            # The search runs the optimizer thousands of times and then simulates its
            # finalists: CPU-bound, and long enough that holding the event loop would stall
            # every other request on the session (docs/architecture.md §7).
            evaluations = await asyncio.to_thread(
                lambda: search(
                    state,
                    my_team_id,
                    capped,
                    partner_team_id=partner_team_id,
                    positions_wanted=positions_wanted,
                    progress=report,
                    week_weights=weights,
                    market_values=values,
                )
            )
            my_weights = weights.get(my_team_id)
            lines = []
            for evaluation in evaluations:
                offer = evaluation.offer
                market = (
                    (evaluation.market_given, evaluation.market_received)
                    if evaluation.market_given is not None
                    and evaluation.market_received is not None
                    else None
                )
                impact = positional_impact(my_roster, offer.give, offer.get, slots, my_weights)
                lines.append(
                    render_trade_line(
                        _partner_label(
                            state.team(offer.partner_team_id).name,
                            tags.get(offer.partner_team_id, ()),
                        ),
                        offer.partner_team_id,
                        offer.give,
                        offer.get,
                        evaluation.my_value_delta,
                        evaluation.partner_value_delta,
                        evaluation.my_playoff_odds_delta,
                        render_positional_impact(impact, ros=True),
                        market,
                    )
                )
            notes = [
                note
                for note in (deadline_note(state.settings.trade_deadline, now(ctx)), values_caveat)
                if note
            ]
            return render_trades(lines, notes)


def register_evaluate_trade(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Evaluate Trade",
        description=(
            "Evaluate a specific proposed trade from both sides: what it does to my playoff "
            "odds, and whether the other manager would plausibly accept. partner_team_id is the "
            "numeric team_id shown in league_standings' ID column or find_trades' '(id N)' "
            "suffix — never guess it from standings rank or roster position."
        ),
        annotations=_READ_ONLY,
    )
    async def evaluate_trade(
        ctx: Context,
        give: list[str],
        get: list[str],
        partner_team_id: int | None = None,
    ) -> TradeVerdict:
        with adapt_errors():
            state = await load_league_state(ctx)
            my_team_id = await resolve_my_team_id(ctx, state.teams)
            my_team = state.team(my_team_id)
            resolved_partner_id = (
                partner_team_id
                if partner_team_id is not None
                else _infer_partner(state, my_team_id, get)
            )
            partner_team = state.team(resolved_partner_id)

            give_players = _resolve_named(my_team.roster.players, give)
            get_players = _resolve_named(partner_team.roster.players, get)
            offer = TradeOffer(
                partner_team_id=resolved_partner_id,
                give=tuple(give_players),
                get=tuple(get_players),
            )

            slots = state.settings.starting_slots
            weights = week_weights_by_team(state, await playoff_probabilities(ctx, state))
            values, _ = await load_trade_values(ctx, state.settings)
            evaluation = evaluate(
                offer,
                my_team.roster,
                partner_team.roster,
                slots,
                weeks_remaining=max(1, state.weeks_remaining),
                my_weeks=weights.get(my_team_id),
                their_weeks=weights.get(resolved_partner_id),
                market_values=values,
            )
            evaluation = await asyncio.to_thread(with_playoff_odds, state, my_team_id, evaluation)
            my_roster = list(my_team.roster.players)
            return build_trade_verdict(
                evaluation,
                evaluation.my_drops,
                positional_impact(my_roster, offer.give, offer.get, slots, weights.get(my_team_id)),
                ros=True,
                deadline=deadline_note(state.settings.trade_deadline, now(ctx)),
            )

"""Trade tools: search across the whole league and price one named offer from both sides.
Adapters only: decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.models import LeagueState, Player, TradeOffer
from ffmcp.domain.trades import (
    apply_offer,
    evaluate,
    positional_impact,
    search,
    with_playoff_odds,
)
from ffmcp.errors import TradePartnerAmbiguous
from ffmcp.mcp._shared import (
    adapt_errors,
    find_players_by_name,
    load_league_state,
    require_single_match,
    resolve_my_team_id,
)
from ffmcp.render.detail import TradeVerdict, build_trade_verdict, render_positional_impact
from ffmcp.render.tables import render_trade_line, render_trades

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
_MAX_RESULTS = 10
"""Hard cap regardless of the caller's ``max_results``."""


def _resolve_named(roster_players: Sequence[Player], names: Sequence[str]) -> list[Player]:
    return [
        require_single_match(find_players_by_name(roster_players, name), name) for name in names
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


def register_find_trades(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Find Trades",
        description="Suggest trades that help my roster and are plausibly accepted.",
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
                )
            )
            lines = [
                render_trade_line(
                    state.team(evaluation.offer.partner_team_id).name,
                    evaluation.offer.partner_team_id,
                    evaluation.offer.give,
                    evaluation.offer.get,
                    evaluation.my_value_delta,
                    evaluation.partner_value_delta,
                    evaluation.my_playoff_odds_delta,
                    render_positional_impact(
                        positional_impact(
                            my_roster, evaluation.offer.give, evaluation.offer.get, slots
                        )
                    ),
                )
                for evaluation in evaluations
            ]
            return render_trades(lines)


def register_evaluate_trade(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Evaluate Trade",
        description=(
            "Evaluate a specific proposed trade from both sides. partner_team_id is the "
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
            weeks_remaining = max(1, state.weeks_remaining)
            evaluation = evaluate(
                offer, my_team.roster, partner_team.roster, slots, weeks_remaining=weeks_remaining
            )
            evaluation = with_playoff_odds(state, my_team_id, evaluation, slots)
            my_roster = list(my_team.roster.players)
            _, drops = apply_offer(my_roster, offer.give, offer.get, slots)
            return build_trade_verdict(
                evaluation,
                drops,
                positional_impact(my_roster, offer.give, offer.get, slots),
            )

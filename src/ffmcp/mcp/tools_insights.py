"""Insight tools: who is due to regress, and how far to trust the projections. Adapters only:
decode args, call domain, render, return (docs/architecture.md §2).
"""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from ffmcp.domain.calibration import source_accuracy
from ffmcp.domain.models import Player, PlayerWeek
from ffmcp.domain.optimizer import is_ruled_out, projected_points
from ffmcp.domain.trades import marginal_value, season_rates
from ffmcp.domain.usage import MIN_GAMES, UsageProfile
from ffmcp.domain.valuation import replacement_levels
from ffmcp.domain.variance import disagreement
from ffmcp.mcp._shared import (
    OPTIONAL_FAILURES,
    adapt_errors,
    app_context,
    load_history,
    load_usage,
    player_pool,
    resolve_my_team_id,
)
from ffmcp.render.tables import (
    render_buy_low_sell_high,
    render_projection_accuracy,
    short_name,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)
_MAX_PER_SECTION = 8
_DISPUTE_THRESHOLD = 0.25
"""Relative ESPN/Sleeper gap worth pointing out: in the 2025 calibration data, projections
disputed by more than about a quarter missed by roughly 40% more than typical ones."""
_MAX_DISPUTES = 4


def register_buy_low_sell_high(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Buy Low / Sell High",
        description=(
            "Find players whose scoring is running ahead of or behind their workload "
            "(snaps, targets, carries): sell-high candidates on my roster, buy-low targets on "
            "other rosters and waivers."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def buy_low_sell_high(ctx: Context, max_per_section: int = 5) -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            week = await app.league.get_current_week()
            teams = await app.league.get_teams()
            my_team_id = await resolve_my_team_id(ctx, teams)
            profiles, caveat = await load_usage(ctx, settings)
            if not profiles:
                return caveat or "No usage data yet this season."

            owner = {
                p.player_id: team for team in teams for p in (*team.roster.players, *team.roster.ir)
            }
            pool = await player_pool(ctx, teams, week)
            by_id = {p.player_id: p for p in pool}
            labelled = any(p.games >= MIN_GAMES for p in profiles.values())
            limit = min(max(1, max_per_section), _MAX_PER_SECTION)

            # A buy-low is only worth naming if he would start for somebody: above replacement
            # at his position, or an upgrade on my own lineup. Rated at the better of his season
            # projection and his volume (xFP), since the claim is that his scoring will catch
            # up with the volume. In a one-quarterback league, four free-agent quarterbacks
            # running below their volume are four players nobody will start.
            slots = settings.starting_slots
            mine = list(next(t for t in teams if t.team_id == my_team_id).roster.players)
            rates = season_rates(pool)
            levels = replacement_levels(settings, pool, rates)
            injured_out: list[str] = []

            def relevant(profile: UsageProfile) -> bool:
                player = by_id.get(profile.player_id)
                if player is None:
                    return False
                if is_ruled_out(player):
                    status = (player.injury_status or "out").replace("_", " ").title()
                    injured_out.append(f"{short_name(player.name)} ({status})")
                    return False
                rate = max(rates.get(player.player_id, 0.0), profile.expected_ppg)
                if rate > levels.get(player.position, 0.0):
                    return True
                as_rated = {**rates, player.player_id: rate}
                return marginal_value(mine, player, slots, as_rated) > 0.0

            def gap(profile: UsageProfile, direction: str) -> bool:
                if labelled:
                    return profile.signal == direction
                return profile.gap_direction == direction

            def pick(direction: str, where: str) -> list[tuple[UsageProfile, str]]:
                rows = []
                for profile in profiles.values():
                    team = owner.get(profile.player_id)
                    if where == "mine" and (team is None or team.team_id != my_team_id):
                        continue
                    if where == "others" and (team is None or team.team_id == my_team_id):
                        continue
                    if where == "free" and (team is not None or profile.player_id not in by_id):
                        continue
                    if not gap(profile, direction):
                        continue
                    if where != "mine" and not relevant(profile):
                        continue
                    rows.append((profile, team.name if team is not None else "FA"))
                sign = -1.0 if direction == "sell_high" else 1.0
                rows.sort(key=lambda row: (sign * row[0].luck_ppg, row[0].player_id))
                return rows[:limit]

            text = render_buy_low_sell_high(
                [
                    ("Sell high (yours, scoring above their workload)", pick("sell_high", "mine")),
                    ("Hold, do not sell low (yours, due to improve)", pick("buy_low", "mine")),
                    ("Buy low (other rosters)", pick("buy_low", "others")),
                    ("Buy low (free agents)", pick("buy_low", "free")),
                ],
                labelled=labelled,
                min_games=MIN_GAMES,
            )
            if injured_out:
                text += "\nInjured, so left out of the buy-low lists: " + ", ".join(
                    sorted(set(injured_out))
                )
            return text if caveat is None else f"{text}\n{caveat}"


def register_projection_accuracy(mcp: MCPServer) -> None:
    @mcp.tool(
        title="Projection Accuracy",
        description=(
            "How accurate ESPN's projections have been in this league this season, by "
            "position, against a second source (Sleeper) and their average; plus where the "
            "two disagree about my roster this week."
        ),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def projection_accuracy(ctx: Context) -> str:
        with adapt_errors():
            app = app_context(ctx)
            settings = await app.league.get_settings()
            week = await app.league.get_current_week()
            teams = await app.league.get_teams()
            my_team_id = await resolve_my_team_id(ctx, teams)
            history, caveat = await load_history(ctx)
            if not history:
                return caveat or "No completed weeks yet: nothing to score projections against."

            by_week: dict[int, list[PlayerWeek]] = {}
            for row in history:
                by_week.setdefault(row.week, []).append(row)

            async def alternative_for(week_rows: list[PlayerWeek]) -> dict[int, float]:
                players = [row.as_player(hindsight=False) for row in week_rows]
                try:
                    return await app.market.get_alt_projections(
                        week_rows[0].week, players, reception_points=settings.reception_points
                    )
                except OPTIONAL_FAILURES:
                    return {}  # the comparison is optional: ESPN alone still reports

            weeks = sorted(by_week)
            alternatives = await asyncio.gather(*(alternative_for(by_week[w]) for w in weeks))
            secondary = {
                (w, player_id): points
                for w, alt in zip(weeks, alternatives, strict=True)
                for player_id, points in alt.items()
            }

            my_team = next(team for team in teams if team.team_id == my_team_id)
            mine: list[Player] = list(my_team.roster.players)
            try:
                current_alt = await app.market.get_alt_projections(
                    week, mine, reception_points=settings.reception_points
                )
            except OPTIONAL_FAILURES:
                current_alt = {}
            disputes = []
            for player in mine:
                espn = projected_points(player)
                other = current_alt.get(player.player_id)
                gap = disagreement(espn, other)
                if (
                    espn is not None
                    and other is not None
                    and gap is not None
                    and gap >= _DISPUTE_THRESHOLD
                    and max(espn, other) >= 5.0
                ):
                    disputes.append((gap, f"{short_name(player.name)} {espn:.1f} vs {other:.1f}"))
            disputes.sort(key=lambda item: -item[0])

            return render_projection_accuracy(
                source_accuracy(history, secondary),
                max(weeks),
                [text for _, text in disputes[:_MAX_DISPUTES]],
            )

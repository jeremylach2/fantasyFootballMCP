"""Prompts: the demo. Each returns a single instruction message that tells the connected model
which tools to call, in what order, and what to hand back. The messages themselves are the
whole implementation. The orchestration happens in the host model's own tool-calling loop, not
on this server.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer


def register(mcp: MCPServer) -> None:
    @mcp.prompt(
        name="weekly_checkin",
        title="Weekly Check-in",
        description="Full Sunday-morning review: lineup, matchup, waivers.",
    )
    def weekly_checkin(week: int | None = None) -> str:
        target = f" for week {week}" if week is not None else ""
        return (
            f"Run my weekly check-in{target}. Call optimize_lineup, then analyze_matchup, then "
            "find_waiver_targets, then buy_low_sell_high. Summarize the results as a single "
            "prioritized action list of at most five items, most impactful first."
        )

    @mcp.prompt(
        name="trade_workshop",
        title="Trade Workshop",
        description="Explore trades to fix my roster's weakest position.",
    )
    def trade_workshop(position: str | None = None, partner_team_id: int | None = None) -> str:
        focus = f" at {position}" if position else " at my weakest position"
        partner = f" with team {partner_team_id}" if partner_team_id is not None else ""
        return (
            f"Help me fix my roster{focus}{partner}. Call power_rankings to see who is really "
            "strong and who manages carelessly, buy_low_sell_high for targets whose value is "
            "about to change, then find_trades to look for options"
            + (f" (position {position})" if position else "")
            + (f" (partner_team_id={partner_team_id})" if partner_team_id is not None else "")
            + ". Present the two best trades, and for each, draft the message I would send the "
            "other manager to propose it."
        )

    @mcp.prompt(
        name="playoff_push",
        title="Playoff Push",
        description="What do I need to do to make the playoffs?",
    )
    def playoff_push() -> str:
        return (
            "Call simulate_season, then find_waiver_targets. Identify the highest-leverage "
            "changes I could make to improve my playoff odds, ranked by their odds delta, and "
            "explain why each one matters."
        )

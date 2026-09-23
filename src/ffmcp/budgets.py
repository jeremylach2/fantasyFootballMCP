"""Token budgets: the single source of truth for the per-tool response cost limits.

Imported by both ``tests/test_token_budgets.py`` and ``scripts/bench_tokens.py`` so the two can
never drift apart. The budget a tool is held to in CI is the exact number printed in the README.
Keyed ``"tool_name"`` for a tool with one shape, or ``"tool_name:detail"`` for one that varies by
``Detail``.
"""

from __future__ import annotations

TOKEN_BUDGETS: dict[str, int] = {
    "get_my_team:compact": 450,
    "get_my_team:standard": 900,
    "get_my_team:full": 1800,
    "optimize_lineup": 500,
    "analyze_matchup:compact": 500,
    "analyze_matchup:standard": 1000,
    "find_trades": 800,
    "evaluate_trade": 350,
    "find_waiver_targets:compact": 600,
    "find_waiver_targets:standard": 1200,
    "find_waiver_targets:full": 2000,
    "simulate_season": 700,
    "league_standings:compact": 400,
    "power_rankings": 700,
    "buy_low_sell_high": 700,
    "player_report": 600,
    "compare_players": 500,
    "projection_accuracy": 450,
}

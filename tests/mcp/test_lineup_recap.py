"""``optimize_lineup`` on the current week must show a "what actually
happened" recap once live scores exist, distinct from the pregame section, and must not show
one at all for a week with no live data (the future, or before this milestone existed).
"""

from __future__ import annotations

from conftest import demo_client


async def test_optimize_lineup_shows_live_recap_for_the_current_week() -> None:
    async with demo_client() as client:
        result = await client.call_tool("optimize_lineup", {})

    assert not result.is_error
    advice = result.structured_content
    assert advice is not None
    actual = advice["actual"]
    assert actual is not None
    # Team Alpha's fixture-placed Monday recap (scripts/make_demo_fixture.py): the started WR
    # goes cold live while a bench WR explodes, so the live-optimal lineup swaps them.
    assert actual["points_scored"] < actual["optimal_live_total"]
    assert actual["points_left_on_bench"] > 0
    # Elijah Villanueva exploded live off the bench; Damon Rios, who started, went cold. The
    # live-optimal lineup promotes one and benches the other (not necessarily paired directly,
    # since both compete with the rest of the WR corps for the same slots).
    benched = {swap["bench"] for swap in actual["swaps"]}
    started = {swap["starter"] for swap in actual["swaps"]}
    assert "Elijah Villanueva" in started
    assert "Damon Rios" in benched


async def test_optimize_lineup_has_no_live_recap_for_a_future_week() -> None:
    async with demo_client() as client:
        result = await client.call_tool("optimize_lineup", {"week": 9})

    assert not result.is_error
    advice = result.structured_content
    assert advice is not None
    assert advice["actual"] is None

"""Progress is reported from ``simulate_season`` and ``find_trades``.

Both tools do long CPU-bound work in a worker thread, so the interesting part is not that the
domain layer emits progress (``tests/domain/test_simulate.py`` covers that) but that the
adapter gets those callbacks, which arrive on the worker thread, back onto the event loop and
out as real ``notifications/progress`` messages. That only shows up over the protocol.
"""

from __future__ import annotations

import pytest
from conftest import demo_client


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("simulate_season", {"n_sims": 500}),
        ("find_trades", {"max_results": 3}),
    ],
)
async def test_long_running_tools_report_progress(tool: str, arguments: dict[str, object]) -> None:
    seen: list[tuple[float, float | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total))

    async with demo_client() as client:
        result = await client.call_tool(tool, arguments, progress_callback=on_progress)

    assert not result.is_error, result.content
    assert seen, f"{tool} reported no progress"
    assert seen == sorted(seen), f"{tool} reported progress out of order: {seen}"
    assert all(total is not None and 0 < done <= total for done, total in seen)

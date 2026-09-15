"""Every tool has ``read_only_hint=True``. This server never writes to ESPN, and says so
structurally, not just in prose.
"""

from __future__ import annotations

from conftest import demo_client


async def test_every_tool_is_read_only() -> None:
    async with demo_client() as client:
        result = await client.list_tools()

    assert result.tools, "expected at least one tool"
    for tool in result.tools:
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.read_only_hint is True, f"{tool.name} is not read_only_hint=True"


async def test_every_tool_is_idempotent() -> None:
    """``simulate_season`` is stochastic unless seeded, but the default seed is fixed for
    reproducibility specifically so it can be ``idempotent_hint=True`` too. This deployment
    always seeds it, so every tool ends up idempotent here."""
    async with demo_client() as client:
        result = await client.list_tools()

    for tool in result.tools:
        assert tool.annotations is not None
        assert tool.annotations.idempotent_hint is True, f"{tool.name} is not idempotent_hint=True"

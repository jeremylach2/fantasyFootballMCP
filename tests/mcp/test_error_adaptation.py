"""An ``FFMCPError`` must reach the client as a short, readable message, not a stack trace and
not silently swallowed. The installed SDK only forwards a ``ToolError``'s/``ResourceError``'s
text; everything else is logged server-side only (see ``ffmcp.mcp._shared.adapt_errors``). This
file is the regression test for that bridge.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import demo_client
from mcp.shared.exceptions import MCPError


async def test_ambiguous_player_name_is_a_clean_tool_error() -> None:
    async with demo_client() as client:
        result = await client.call_tool("player_report", {"name": "WR Starter4"})

    assert result.is_error
    text = result.content[0].text  # type: ignore[union-attr]
    assert "WR Starter4" in text
    assert "Traceback" not in text


async def test_unknown_team_id_is_a_clean_resource_error() -> None:
    """A resource has no ``isError`` content channel: reading one that fails surfaces as a
    JSON-RPC error at the client-library level, not a ``CallToolResult``. Still just the short
    message, though: never a stack trace."""
    async with demo_client() as client:
        with pytest.raises(MCPError) as exc_info:
            await client.read_resource("ffmcp://team/999/roster")

    message = str(exc_info.value)
    assert "999" in message
    assert "Traceback" not in message


async def test_missing_team_id_without_elicitation_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one elicited parameter in the whole
    surface has a working non-interactive fallback (an error naming ``FFMCP_TEAM_ID``) for a
    client (like this test's default in-process one) that does not declare elicitation.

    ``monkeypatch.delenv`` only clears the process environment variable. ``Settings``' own
    ``env_file=".env"`` support (``config.py``) reads a real ``.env`` in the repo root directly,
    so a contributor's own gitignored file (e.g. a real ``FFMCP_TEAM_ID`` for their live league)
    would otherwise leak into this test and hide the fallback path it exists to check. Changing
    to an empty ``tmp_path`` makes the relative ``".env"`` resolve to a file that doesn't exist.
    """
    monkeypatch.delenv("FFMCP_TEAM_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    async with demo_client() as client:
        result = await client.call_tool("get_my_team", {})

    assert result.is_error
    text = result.content[0].text  # type: ignore[union-attr]
    assert "FFMCP_TEAM_ID" in text

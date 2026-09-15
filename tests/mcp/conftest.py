"""Shared fixtures for tests/mcp/*: an in-process client against the demo fixture league,
exercised through the real MCP protocol layer (``mcp.Client`` connected in-process to the
``MCPServer``, no network, no stdio, but the actual request/response path a real client uses,
including the lifespan). Every tool is callable end-to-end in demo mode.

``demo_client()`` is a plain async-context-manager factory rather than a fixture that itself
spans ``async with`` across a ``yield``: anyio's cancel scopes must be entered and exited in the
same asyncio Task, and a generator fixture's teardown can run in a different Task than its
setup under pytest-asyncio. Entering and exiting entirely inside one test avoids that.
"""

from __future__ import annotations

import pytest

from ffmcp.server import build_server
from mcp import Client


@pytest.fixture(autouse=True)
def _demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMCP_MODE", "demo")
    monkeypatch.setenv("FFMCP_TEAM_ID", "1")


def demo_client() -> Client:
    """``async with demo_client() as client:``: a fresh server and connection per use."""
    return Client(build_server())

"""BearerAuthMiddleware: the one thing standing between a public streamable-HTTP deployment
and anyone on the internet reading a live league through it."""

from __future__ import annotations

import httpx2
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from ffmcp.server import BearerAuthMiddleware


async def _ok(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _wrapped_app(token: str) -> BearerAuthMiddleware:
    inner = Starlette(routes=[Route("/mcp", _ok)])
    return BearerAuthMiddleware(inner, token)


@pytest.fixture
def client() -> httpx2.AsyncClient:
    transport = httpx2.ASGITransport(app=_wrapped_app("secret-token"))
    return httpx2.AsyncClient(transport=transport, base_url="http://test")


async def test_missing_authorization_header_is_rejected(client: httpx2.AsyncClient) -> None:
    response = await client.get("/mcp")
    assert response.status_code == 401


async def test_wrong_token_is_rejected(client: httpx2.AsyncClient) -> None:
    response = await client.get("/mcp", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


async def test_correct_token_is_admitted(client: httpx2.AsyncClient) -> None:
    response = await client.get("/mcp", headers={"Authorization": "Bearer secret-token"})
    assert response.status_code == 200
    assert response.text == "ok"

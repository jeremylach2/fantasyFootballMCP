# syntax=docker/dockerfile:1
#
# Streamable-HTTP deployment image. stdio clients (Claude Desktop/Code) run the server locally
# via `uvx ffmcp` or `uv run ffmcp` instead, per the README's Quickstart section. This image is
# for the HTTP transport only.
#
# Build:  docker build -t ffmcp .
# Run:    docker run -p 8000:8000 --env-file .env ffmcp
# Try it with no credentials at all: docker run -p 8000:8000 -e FFMCP_MODE=demo ffmcp
#
# Live mode (a real league) over HTTP additionally requires FFMCP_AUTH_TOKEN — the server
# refuses to start without one, since an open endpoint would let anyone who finds the URL read
# the league. Demo mode needs none: it only ever serves synthetic fixtures.

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer, before the source changes underneath them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY README.md ./
COPY tests/fixtures ./tests/fixtures
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.13-slim

RUN groupadd --system ffmcp && useradd --system --gid ffmcp --create-home ffmcp

WORKDIR /app
COPY --from=builder --chown=ffmcp:ffmcp /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER ffmcp

EXPOSE 8000

# Stateless streamable HTTP (server.py: stateless_http=True), so no sticky sessions are needed
# and this scales behind a plain load balancer. --host 0.0.0.0 because the default 127.0.0.1 is
# unreachable through Docker's port mapping, which forwards to the container's external interface.
#
# Routed through `sh -c` so ${PORT:-8000} expands: most PaaS hosts (Render, Railway, Cloud
# Run, Heroku-style buildpacks) inject their own PORT and route traffic to whatever the app
# binds, ignoring EXPOSE — hardcoding 8000 here would silently mismatch it on any of them. A
# plain `docker run -p 8000:8000` still works since PORT is unset there. `exec` replaces the
# shell with ffmcp as PID 1, so it receives SIGTERM directly instead of the shell swallowing it
# and forcing a SIGKILL after the platform's shutdown grace period.
ENTRYPOINT ["/bin/sh", "-c", "exec ffmcp --transport streamable-http --host 0.0.0.0 --port ${PORT:-8000}"]

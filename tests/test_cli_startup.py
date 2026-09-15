"""A misconfigured server fails with one readable line, not a traceback (docs/architecture.md
§5).

`tests/test_config.py` already asserts that `load_settings()` raises the right message. That is
a different claim: a message only reaches an operator well if it is raised somewhere that
prints it well. Loading settings inside `app_lifespan` puts the raise inside anyio's task
group, where it surfaces as a forty-line `ExceptionGroup` with the one useful line at the
bottom. That is how this was found, by a user running the server with no `.env` present. So
`main()` validates before starting a transport, and this test covers that path end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_missing_league_id_in_live_mode_prints_one_line_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    # A subprocess, not a call to `main()`: the behaviour under test is what the *process*
    # writes to stderr and exits with. Run from an empty directory with every FFMCP_* variable
    # stripped, so neither the ambient environment nor a stray `.env` can satisfy the setting.
    env = {k: v for k, v in os.environ.items() if not k.startswith("FFMCP_")}
    result = subprocess.run(
        [sys.executable, "-m", "ffmcp.server"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**env, "FFMCP_MODE": "live"},
        stdin=subprocess.DEVNULL,
        cwd=str(tmp_path),
    )

    assert result.returncode != 0
    stderr = result.stderr.strip()

    assert "FFMCP_LEAGUE_ID" in stderr, stderr
    assert ".env.example" in stderr, stderr

    # The actual regression: no traceback, and short enough that the message is the message.
    assert "Traceback" not in stderr, stderr
    assert "ExceptionGroup" not in stderr, stderr
    assert len(stderr.splitlines()) == 1, f"expected one line, got:\n{stderr}"

    # stdout carries the protocol on stdio and must stay clean even on the failure path.
    assert result.stdout == "", result.stdout


def test_live_streamable_http_without_auth_token_prints_one_line_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    # Same shape as the missing-league-id case above: a live-mode HTTP server with no bearer
    # token would let anyone who finds the URL read the owner's real league, so this must fail
    # before uvicorn ever binds a socket, not silently serve.
    env = {k: v for k, v in os.environ.items() if not k.startswith("FFMCP_")}
    result = subprocess.run(
        [sys.executable, "-m", "ffmcp.server", "--transport", "streamable-http"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**env, "FFMCP_MODE": "live", "FFMCP_LEAGUE_ID": "1234567"},
        stdin=subprocess.DEVNULL,
        cwd=str(tmp_path),
    )

    assert result.returncode != 0
    stderr = result.stderr.strip()

    assert "FFMCP_AUTH_TOKEN" in stderr, stderr
    assert "Traceback" not in stderr, stderr
    assert len(stderr.splitlines()) == 1, f"expected one line, got:\n{stderr}"
    assert result.stdout == "", result.stdout

"""One readable line, naming the variable, on misconfiguration."""

from __future__ import annotations

from pathlib import Path

import pytest

from ffmcp.config import ConfigurationError, load_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("FFMCP_MODE", "FFMCP_LEAGUE_ID", "FFMCP_SEASON", "FFMCP_TEAM_ID"):
        monkeypatch.delenv(var, raising=False)
    # Isolate from a real .env in the working directory.
    monkeypatch.chdir(tmp_path)


def test_live_mode_without_league_id_raises_one_line_error() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings()

    message = str(exc_info.value)
    assert "\n" not in message
    assert "FFMCP_LEAGUE_ID" in message


def test_demo_mode_needs_no_league_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMCP_MODE", "demo")
    settings = load_settings()
    assert settings.mode == "demo"
    assert settings.league_id is None


def test_live_mode_with_league_id_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMCP_LEAGUE_ID", "1234567")
    settings = load_settings()
    assert settings.league_id == 1234567

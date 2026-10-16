"""Fixtures carry no real league id and no cookie-shaped strings."""

from __future__ import annotations

import json
from pathlib import Path

from ffmcp.errors import redact

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_fixtures_contain_no_cookie_shaped_strings() -> None:
    for path in FIXTURES_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert redact(text) == text, f"{path.name} contains a cookie-shaped string"


def test_league_fixtures_use_the_documented_placeholder_id() -> None:
    leagues = sorted(FIXTURES_DIR.glob("*_league.json"))
    assert {path.name for path in leagues} >= {"demo_league.json", "snapshot_league.json"}
    for path in leagues:
        league = json.loads(path.read_text(encoding="utf-8"))
        assert league["settings"]["league_id"] == 1234567, path.name

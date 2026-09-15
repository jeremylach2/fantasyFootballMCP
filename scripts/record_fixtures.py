"""Pull a real league via ``providers/espn.py`` and ``providers/sleeper.py``, anonymize it, and
write ``tests/fixtures/demo_league.json`` / ``demo_market.json`` in the exact shape
``providers/demo.py`` reads back (docs/architecture.md §6).

Anonymization: team names are replaced with ``Team Alpha``, ``Team Bravo``, ... in standing
order, and the league id is replaced with ``1234567``. Player names stay real: the analysis
is meaningless without them and they are public facts (docs/architecture.md §6).

Requires live credentials (``FFMCP_MODE=live`` and the usual ``.env``); run it once whenever
the fixture needs refreshing against real data shapes, then commit the result. Never run this
against a league whose owners have not agreed to have (anonymized) roster data committed to a
public repo.

    uv run python scripts/record_fixtures.py
"""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path
from typing import Any

import httpx2

from ffmcp.config import load_settings
from ffmcp.providers.cache import Cache
from ffmcp.providers.espn import EspnLeagueProvider
from ffmcp.providers.sleeper import SleeperMarketProvider

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ANONYMOUS_LEAGUE_ID = 1234567


def _team_alias(index: int) -> str:
    letter = string.ascii_uppercase[index % 26]
    names = {
        "A": "Alpha",
        "B": "Bravo",
        "C": "Charlie",
        "D": "Delta",
        "E": "Echo",
        "F": "Foxtrot",
        "G": "Golf",
        "H": "Hotel",
        "I": "India",
        "J": "Juliet",
        "K": "Kilo",
        "L": "Lima",
    }
    return f"Team {names.get(letter, letter)}"


async def _record() -> None:
    settings = load_settings()
    if settings.mode != "live" or settings.league_id is None or settings.season is None:
        raise SystemExit(
            "record_fixtures.py needs FFMCP_MODE=live, FFMCP_LEAGUE_ID and FFMCP_SEASON set."
        )

    league = EspnLeagueProvider(
        league_id=settings.league_id,
        season=settings.season,
        espn_s2=settings.espn_s2.get_secret_value() if settings.espn_s2 else None,
        swid=settings.swid.get_secret_value() if settings.swid else None,
    )

    espn_settings = await league.get_settings()
    teams = await league.get_teams()
    current_week = await league.get_current_week()
    matchups = await league.get_matchups(current_week)
    free_agents = await league.get_free_agents(current_week, size=10)

    alias_by_team_id = {team.team_id: _team_alias(i) for i, team in enumerate(teams)}

    def anonymize_team(team: Any) -> dict[str, Any]:
        dumped = team.model_dump(mode="json")
        dumped["name"] = alias_by_team_id[team.team_id]
        dumped["abbrev"] = alias_by_team_id[team.team_id].split()[1][:3].upper()
        return dumped

    league_state = {
        "settings": {**espn_settings.model_dump(mode="json"), "league_id": ANONYMOUS_LEAGUE_ID},
        "current_week": current_week,
        "teams": [anonymize_team(t) for t in teams],
        "matchups": {str(current_week): [m.model_dump(mode="json") for m in matchups]},
        "free_agents": {str(current_week): [p.model_dump(mode="json") for p in free_agents]},
    }

    market_by_espn_id: dict[str, Any] = {}
    all_players = [slot.player for t in teams for slot in t.roster.slots if slot.player is not None]
    all_players += [p for t in teams for p in t.roster.bench]
    all_players += free_agents

    cache = Cache(settings.cache_dir)
    async with httpx2.AsyncClient(timeout=httpx2.Timeout(connect=5.0, read=15.0)) as client:
        sleeper = SleeperMarketProvider(client, cache)
        sleeper_week = await sleeper.get_current_week()
        trending = await sleeper.get_trending_adds()
        for player in all_players:
            signal = await sleeper.get_market_signal(
                espn_player_id=player.player_id,
                name=player.name,
                position=player.position,
                pro_team=player.pro_team,
            )
            if signal is not None:
                market_by_espn_id[str(player.player_id)] = signal.model_dump(mode="json")

    market = {
        "current_week": sleeper_week,
        "trending_adds": trending,
        "market_by_espn_id": market_by_espn_id,
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "demo_league.json").write_text(
        json.dumps(league_state, indent=2) + "\n", encoding="utf-8"
    )
    (FIXTURES_DIR / "demo_market.json").write_text(
        json.dumps(market, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURES_DIR / 'demo_league.json'}")
    print(f"wrote {FIXTURES_DIR / 'demo_market.json'}")


def main() -> None:
    asyncio.run(_record())


if __name__ == "__main__":
    main()

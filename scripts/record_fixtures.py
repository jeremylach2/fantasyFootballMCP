"""Record a real league as a committed, anonymized fixture set that ``FFMCP_MODE=demo`` serves
(``FFMCP_FIXTURE=<name>``, docs/architecture.md §6).

    uv run python scripts/record_fixtures.py [--name snapshot]

Writes ``tests/fixtures/{name}_league.json``, ``{name}_market.json`` and ``{name}_insights.json``
in the shapes ``providers/demo.py`` reads back: settings, every roster, the whole season's
schedule, a deep free-agent pool, league history, usage, second-opinion projections, injured
players' week-by-week outlooks, market trade values, game lines and recent activity. Everything
a tool reads, so a test against the fixture exercises the same code paths a live call does.

Anonymization: team names become ``Team Alpha``, ``Team Bravo``, ... in team-id order, and the
league id becomes ``1234567``. Team ids are kept (they are small integers, not identifiers).
Player names stay real: the analysis is meaningless without them and they are public facts.
There are no owner names anywhere in the domain models to strip.

Requires live credentials (``.env``, as for ``FFMCP_MODE=live``). Never run this against a
league whose managers have not agreed to have anonymized rosters committed to a public repo.
It refuses ``--name demo``: that set is the synthetic league the README is built from
(``scripts/make_demo_fixture.py``), and overwriting it would silently rewrite every transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2

from ffmcp.config import load_settings
from ffmcp.domain.models import Player, Team
from ffmcp.domain.optimizer import is_ruled_out
from ffmcp.providers.cache import Cache
from ffmcp.providers.espn import EspnLeagueProvider
from ffmcp.providers.fantasycalc import FantasyCalcValueProvider
from ffmcp.providers.nflverse import NflverseUsageProvider
from ffmcp.providers.odds import OddsApiProvider
from ffmcp.providers.sleeper import SleeperMarketProvider

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ANONYMOUS_LEAGUE_ID = 1234567
FREE_AGENT_POOL = 300
FREE_AGENTS_PER_POSITION = 60
POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")

# fmt: off
ALIASES = (
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
    "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
    "Sierra", "Tango",
)
# fmt: on


def _dump(path: Path, data: Any) -> None:
    """Compact, unlike the hand-readable demo set: this one is regenerated, never edited."""
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


def _quarterbacks(slot_counts: dict[str, int]) -> int:
    return 2 if slot_counts.get("OP", 0) or slot_counts.get("SUPERFLEX", 0) else 1


async def _record(name: str) -> None:
    settings = load_settings()
    if settings.mode != "live" or settings.league_id is None:
        raise SystemExit("record_fixtures.py needs FFMCP_MODE=live and FFMCP_LEAGUE_ID set.")

    cache = Cache(settings.cache_dir)
    async with httpx2.AsyncClient(timeout=httpx2.Timeout(15.0, connect=5.0)) as client:
        sleeper = SleeperMarketProvider(client, cache)
        league = EspnLeagueProvider(
            league_id=settings.league_id,
            season=settings.season,
            espn_s2=settings.espn_s2.get_secret_value() if settings.espn_s2 else None,
            swid=settings.swid.get_secret_value() if settings.swid else None,
            resolve_season=sleeper.get_current_season,
            cache=cache,
        )
        usage_provider = NflverseUsageProvider(client, cache, sleeper.get_current_season)
        odds_provider = OddsApiProvider(
            client,
            cache,
            settings.odds_api_key.get_secret_value() if settings.odds_api_key else None,
        )
        values_provider = FantasyCalcValueProvider(client, cache)

        league_settings = await league.get_settings()
        week = await league.get_current_week()
        teams = await league.get_teams()
        matchups = {
            str(w): [
                m.model_dump(mode="json", exclude_defaults=True)
                for m in await league.get_matchups(w)
            ]
            for w in range(1, league_settings.reg_season_weeks + 1)
        }

        pool: dict[int, Player] = {
            p.player_id: p for p in await league.get_free_agents(week, size=FREE_AGENT_POOL)
        }
        for position in POSITIONS:
            for p in await league.get_free_agents(
                week, size=FREE_AGENTS_PER_POSITION, position=position
            ):
                pool.setdefault(p.player_id, p)
        free_agents = sorted(
            pool.values(),
            key=lambda p: (-(p.market.percent_owned or 0.0) if p.market else 0.0, p.player_id),
        )

        rostered = [p for t in teams for p in (*t.roster.players, *t.roster.ir)]
        everyone = rostered + free_agents
        history = await league.get_player_history()
        activity = await league.get_recent_activity(size=25)

        trending = await sleeper.get_trending_adds()
        market_by_espn_id: dict[str, Any] = {}
        for player in everyone:
            signal = await sleeper.get_market_signal(
                espn_player_id=player.player_id,
                name=player.name,
                position=player.position,
                pro_team=player.pro_team,
            )
            if signal is not None and (signal.trending_adds or signal.injury_detail):
                market_by_espn_id[str(player.player_id)] = signal.model_dump(
                    mode="json", exclude_defaults=True
                )

        rp = league_settings.reception_points
        alt_projections: dict[str, dict[str, float]] = {}
        for w in range(1, week + 1):
            players = everyone if w == week else [r.as_player() for r in history if r.week == w]
            alt = await sleeper.get_alt_projections(w, players, reception_points=rp)
            alt_projections[str(w)] = {str(pid): round(pts, 2) for pid, pts in alt.items()}

        needs_outlook = [p for p in everyone if is_ruled_out(p) or p.season_rate is None]
        outlook = await sleeper.get_outlook(needs_outlook, reception_points=rp)

        values = await values_provider.get_trade_values(
            team_count=league_settings.team_count,
            quarterbacks=_quarterbacks(league_settings.slot_counts),
            reception_points=rp,
        )
        ids = {p.player_id for p in everyone}
        usage = [
            row.model_dump(mode="json", exclude_defaults=True)
            for row in await usage_provider.get_usage(reception_points=rp)
            if row.player_id in ids
        ]
        lines = await odds_provider.get_game_lines()

    alias = {team.team_id: f"Team {ALIASES[i]}" for i, team in enumerate(teams)}

    def anonymize(team: Team) -> dict[str, Any]:
        dumped = team.model_dump(mode="json", exclude_defaults=True)
        dumped["name"] = alias[team.team_id]
        dumped["abbrev"] = alias[team.team_id].split()[1][:4].upper()
        return dumped

    games = []
    for team, line in sorted(lines.items()):
        if team < line.opponent:
            games.append(
                {
                    "home": team,
                    "away": line.opponent,
                    "total": line.total,
                    "home_spread": line.spread,
                }
            )

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _dump(
        FIXTURES_DIR / f"{name}_league.json",
        {
            # The moment this was taken: demo mode replays it as "now", so lineup locks and
            # deadlines read the same in a test next year as they did on the day.
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                **league_settings.model_dump(mode="json", exclude_defaults=True),
                "league_id": ANONYMOUS_LEAGUE_ID,
            },
            "current_week": week,
            "teams": [anonymize(t) for t in teams],
            "matchups": matchups,
            "free_agents": {
                str(week): [p.model_dump(mode="json", exclude_defaults=True) for p in free_agents]
            },
        },
    )
    _dump(
        FIXTURES_DIR / f"{name}_market.json",
        {
            "current_season": league_settings.season,
            "current_week": week,
            "trending_adds": trending,
            "market_by_espn_id": market_by_espn_id,
        },
    )
    _dump(
        FIXTURES_DIR / f"{name}_insights.json",
        {
            "history": [row.model_dump(mode="json", exclude_defaults=True) for row in history],
            "usage": usage,
            "alt_projections": alt_projections,
            "outlook": {
                str(pid): {str(w): round(pts, 2) for w, pts in weeks.items()}
                for pid, weeks in outlook.items()
                if weeks
            },
            "trade_values": {str(pid): value for pid, value in values.items()},
            "game_lines": games,
            "activity": [t.model_dump(mode="json", exclude_defaults=True) for t in activity],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="snapshot", help="fixture set name (not 'demo')")
    args = parser.parse_args()
    if args.name == "demo":
        raise SystemExit("refusing to overwrite the synthetic demo fixture; pick another --name")
    asyncio.run(_record(args.name))


if __name__ == "__main__":
    main()

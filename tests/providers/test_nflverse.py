"""NflverseUsageProvider against httpx2.MockTransport: the crosswalk turns nflverse's ids into
ESPN's, snap counts join on PFR ids, and team spellings are translated to ESPN's."""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest

from ffmcp.errors import UpstreamUnavailable
from ffmcp.providers.cache import Cache
from ffmcp.providers.nflverse import NflverseUsageProvider

PLAYERS_CSV = (
    "gsis_id,display_name,pfr_id,espn_id\n"
    "00-001,Puka Nacua,NacuPu00,4426515\n"
    "00-002,Nobody Mapped,NobodyXX,NA\n"
)
WEEKLY_CSV = (
    "player_id,player_display_name,position,team,week,season_type,attempts,passing_air_yards,"
    "carries,targets,receiving_air_yards,target_share,air_yards_share,fantasy_points,"
    "fantasy_points_ppr\n"
    "00-001,Puka Nacua,WR,LA,1,REG,0,0,1,12,110,0.31,0.4,15.5,24.5\n"
    "00-001,Puka Nacua,WR,LA,2,POST,0,0,0,9,80,0.25,0.3,9.0,15.0\n"
    "00-002,Nobody Mapped,WR,LA,1,REG,0,0,0,2,20,0.05,0.1,1.0,2.0\n"
)
SNAPS_CSV = (
    "game_id,season,game_type,week,player,pfr_player_id,position,team,offense_pct\n"
    "g1,2026,REG,1,Puka Nacua,NacuPu00,WR,LA,0.93\n"
)


def _handler(request: httpx2.Request) -> httpx2.Response:
    path = request.url.path
    if path.endswith("/players/players.csv"):
        return httpx2.Response(200, text=PLAYERS_CSV)
    if path.endswith("/stats_player/stats_player_week_2026.csv"):
        return httpx2.Response(200, text=WEEKLY_CSV)
    if path.endswith("/snap_counts/snap_counts_2026.csv"):
        return httpx2.Response(200, text=SNAPS_CSV)
    return httpx2.Response(404)


async def _season() -> int:
    return 2026


async def test_usage_is_keyed_to_espn_ids_with_snaps_and_espn_team_names(tmp_path: Path) -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)) as client:
        provider = NflverseUsageProvider(client, Cache(tmp_path), _season)
        rows = await provider.get_usage(reception_points=0.5)

    assert len(rows) == 1  # the playoff game and the unmapped player are both dropped
    row = rows[0]
    assert row.player_id == 4426515
    assert row.pro_team == "LAR"
    assert row.snap_pct == 0.93
    assert row.targets == 12.0
    assert row.fantasy_points == pytest.approx(20.0)  # half-PPR, between 15.5 and 24.5


async def test_a_failed_download_is_an_upstream_error(tmp_path: Path) -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(fail)) as client:
        provider = NflverseUsageProvider(client, Cache(tmp_path), _season)
        with pytest.raises(UpstreamUnavailable):
            await provider.get_usage()

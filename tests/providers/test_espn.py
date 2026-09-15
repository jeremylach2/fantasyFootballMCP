"""EspnLeagueProvider against a stand-in League: zero network.

``espn_api.football.League`` does its own synchronous network I/O via ``requests``, so there is
no httpx2.MockTransport equivalent for it. Instead this stubs the ``League`` symbol that
``providers/espn.py`` imports with a fake exposing exactly the attributes verified against the
installed ``espn-api==0.46.0``, which exercises the real mapping code without touching the
network.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague, ESPNUnknownError

import ffmcp.providers.espn as espn_module
from ffmcp.errors import CredentialsMissing, LeagueNotAccessible, UpstreamUnavailable
from ffmcp.providers.espn import EspnLeagueProvider


class FakePlayer:
    def __init__(
        self,
        player_id: int,
        name: str,
        position: str,
        eligible_slots: list[str],
        pro_team: str,
        *,
        lineup_slot: str = "",
        injured: bool = False,
        injury_status: str | None = None,
        stats: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.playerId = player_id
        self.name = name
        self.position = position
        self.eligibleSlots = eligible_slots
        self.proTeam = pro_team
        self.lineupSlot = lineup_slot
        self.injured = injured
        self.injuryStatus = injury_status
        self.stats = stats or {}


class FakeTeam:
    def __init__(self, team_id: int, roster: list[FakePlayer]) -> None:
        self.team_id = team_id
        self.team_name = f"Team {team_id}"
        self.team_abbrev = f"T{team_id}"
        self.wins = 2
        self.losses = 1
        self.points_for = 250.5
        self.points_against = 210.1
        self.standing = team_id
        self.playoff_pct = 65.0
        self.roster = roster


class FakeSettings:
    team_count = 2
    playoff_team_count = 2
    reg_season_count = 14
    position_slot_counts: ClassVar[dict[str, int]] = {"QB": 1, "BE": 2}
    scoring_type = "PPR"


class FakeBoxScore:
    def __init__(self, home_team_id: int | None, away_team_id: int | None) -> None:
        self.home_team = home_team_id
        self.away_team = away_team_id
        self.home_score = 100.0
        self.away_score = 90.0
        self.home_projected = 105.0
        self.away_projected = 95.0
        self.is_playoff = False


QB = FakePlayer(
    1, "QB One", "QB", ["QB"], "BUF", lineup_slot="QB", stats={3: {"projected_points": 20.5}}
)
BENCH_RB = FakePlayer(2, "RB One", "RB", ["RB", "RB/WR/TE"], "SF", lineup_slot="BE")
IR_WR = FakePlayer(
    3, "WR One", "WR", ["WR"], "DAL", lineup_slot="IR", injured=True, injury_status="OUT"
)


class FakeLeague:
    def __init__(
        self, league_id: int, year: int, espn_s2: str | None = None, swid: str | None = None
    ) -> None:
        self.current_week = 3
        self.settings = FakeSettings()
        self.teams = [FakeTeam(1, [QB, BENCH_RB, IR_WR])]

    def box_scores(self, week: int) -> list[FakeBoxScore]:
        return [FakeBoxScore(1, 2)]

    def free_agents(self, week: int, size: int, position: str | None) -> list[FakePlayer]:
        agents = [QB]
        if position is not None:
            agents = [p for p in agents if p.position == position]
        return agents[:size]


@pytest.fixture(autouse=True)
def _patch_league(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(espn_module, "League", FakeLeague)


def _provider() -> EspnLeagueProvider:
    return EspnLeagueProvider(league_id=1234567, season=2026, espn_s2=None, swid=None)


async def test_get_current_week() -> None:
    assert await _provider().get_current_week() == 3


async def test_get_settings_maps_slot_counts_and_scoring_type() -> None:
    settings = await _provider().get_settings()
    assert settings.league_id == 1234567
    assert settings.team_count == 2
    assert settings.slot_counts == {"QB": 1, "BE": 2}
    assert settings.scoring_type == "PPR"


async def test_get_teams_splits_starters_bench_and_ir() -> None:
    teams = await _provider().get_teams()
    assert len(teams) == 1
    roster = teams[0].roster
    assert [s.slot for s in roster.slots] == ["QB"]
    assert roster.slots[0].player is not None
    assert roster.slots[0].player.projection is not None
    assert roster.slots[0].player.projection.points == 20.5
    assert [p.name for p in roster.bench] == ["RB One"]
    assert [p.name for p in roster.ir] == ["WR One"]
    assert roster.ir[0].injured is True


async def test_get_matchups_maps_team_ids_not_team_objects() -> None:
    matchups = await _provider().get_matchups(3)
    assert matchups[0].home_team_id == 1
    assert matchups[0].away_team_id == 2


async def test_missing_season_is_resolved_lazily_on_first_use() -> None:
    calls = 0

    async def resolve_season() -> int:
        nonlocal calls
        calls += 1
        return 2026

    provider = EspnLeagueProvider(
        league_id=1234567, season=None, espn_s2=None, swid=None, resolve_season=resolve_season
    )
    assert calls == 0  # not resolved at construction

    settings = await provider.get_settings()
    assert settings.season == 2026
    assert calls == 1

    await provider.get_current_week()
    assert calls == 1  # resolved once, then reused


async def test_get_free_agents_filters_by_position() -> None:
    agents = await _provider().get_free_agents(3, position="RB")
    assert agents == []
    agents = await _provider().get_free_agents(3, position="QB")
    assert [p.name for p in agents] == ["QB One"]


async def test_access_denied_without_credentials_raises_credentials_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_denied(*args: Any, **kwargs: Any) -> FakeLeague:
        raise ESPNAccessDenied("espn_s2 and swid are required")

    monkeypatch.setattr(espn_module, "League", raise_denied)
    with pytest.raises(CredentialsMissing):
        await _provider().get_current_week()


async def test_access_denied_with_credentials_raises_league_not_accessible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_denied(*args: Any, **kwargs: Any) -> FakeLeague:
        raise ESPNAccessDenied("rejected")

    monkeypatch.setattr(espn_module, "League", raise_denied)
    provider = EspnLeagueProvider(league_id=1, season=2026, espn_s2="s2", swid="swid")
    with pytest.raises(LeagueNotAccessible):
        await provider.get_current_week()


async def test_invalid_league_raises_league_not_accessible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_invalid(*args: Any, **kwargs: Any) -> FakeLeague:
        raise ESPNInvalidLeague("does not exist")

    monkeypatch.setattr(espn_module, "League", raise_invalid)
    with pytest.raises(LeagueNotAccessible):
        await _provider().get_current_week()


async def test_unknown_error_raises_upstream_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unknown(*args: Any, **kwargs: Any) -> FakeLeague:
        raise ESPNUnknownError("HTTP 500")

    monkeypatch.setattr(espn_module, "League", raise_unknown)
    with pytest.raises(UpstreamUnavailable):
        await _provider().get_current_week()

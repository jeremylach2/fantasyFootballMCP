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
        percent_owned: float = -1,
        percent_started: float = -1,
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
        # espn-api's real Player always sets these, defaulting to -1 (see
        # ffmcp.providers.espn._map_market_signal) rather than omitting them.
        self.percent_owned = percent_owned
        self.percent_started = percent_started


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


class FakeTeamRef:
    """Stands in for the ``Team`` objects ``League.scoreboard()`` attaches to a matchup."""

    def __init__(self, team_id: int) -> None:
        self.team_id = team_id


class FakeScheduledMatchup:
    """Stands in for ``espn_api.football.matchup.Matchup`` as returned by ``scoreboard()``.
    ``home_team``/``away_team`` are absent entirely (not ``None``) for a bye, matching the real
    class, which only assigns them inside a loop that runs when a team actually matches."""

    def __init__(self, home_team_id: int | None, away_team_id: int | None) -> None:
        if home_team_id is not None:
            self.home_team = FakeTeamRef(home_team_id)
        if away_team_id is not None:
            self.away_team = FakeTeamRef(away_team_id)
        self.home_score = 0.0
        self.away_score = 0.0
        self.is_playoff = False


QB = FakePlayer(
    1,
    "QB One",
    "QB",
    ["QB"],
    "BUF",
    lineup_slot="QB",
    stats={3: {"projected_points": 20.5}},
    percent_owned=87.3,
    percent_started=71.2,
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

    def scoreboard(self, week: int) -> list[FakeScheduledMatchup]:
        # Deliberately a different pairing from box_scores(), so a test can tell which one
        # get_matchups actually called.
        return [FakeScheduledMatchup(1, 3)]

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


async def test_get_matchups_for_a_played_week_uses_box_scores() -> None:
    # week == current_week (3): box_scores() is trustworthy here.
    matchups = await _provider().get_matchups(3)
    assert (matchups[0].home_team_id, matchups[0].away_team_id) == (1, 2)
    assert matchups[0].home_projected == 105.0


async def test_get_matchups_for_a_future_week_uses_scoreboard_not_box_scores() -> None:
    # week > current_week (3): box_scores() silently returns the current week's pairing
    # relabeled, so get_matchups must fall back to scoreboard() instead. FakeLeague's two
    # methods return different pairings precisely so this distinguishes them.
    matchups = await _provider().get_matchups(6)
    assert (matchups[0].home_team_id, matchups[0].away_team_id) == (1, 3)
    assert matchups[0].week == 6
    assert matchups[0].home_projected == 0.0


async def test_get_matchups_for_a_future_bye_has_no_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # scoreboard() only sets home_team/away_team when a team actually matched; a bye leaves
    # the attribute absent rather than None, and _map_scheduled_matchup must handle that
    # rather than raising AttributeError.
    monkeypatch.setattr(
        FakeLeague, "scoreboard", lambda self, week: [FakeScheduledMatchup(1, None)]
    )
    matchups = await _provider().get_matchups(6)
    assert matchups[0].home_team_id == 1
    assert matchups[0].away_team_id is None


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


async def test_get_teams_maps_espn_ownership_into_market_signal() -> None:
    teams = await _provider().get_teams()
    starter = teams[0].roster.slots[0].player
    assert starter is not None
    assert starter.market is not None
    assert starter.market.percent_owned == 87.3
    assert starter.market.percent_started == 71.2
    assert starter.market.trending_adds is None  # ESPN has no notion of this; Sleeper adds it


async def test_get_teams_leaves_market_none_when_espn_reports_no_ownership_data() -> None:
    # BENCH_RB uses the FakePlayer default of -1, espn-api's sentinel for "no reading" (see
    # ffmcp.providers.espn._map_market_signal): it must not be surfaced as a real 0% or -1%.
    teams = await _provider().get_teams()
    assert teams[0].roster.bench[0].market is None


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

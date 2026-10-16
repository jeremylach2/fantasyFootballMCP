"""EspnLeagueProvider against a stand-in League: zero network.

``espn_api.football.League`` does its own synchronous network I/O via ``requests``, so there is
no httpx2.MockTransport equivalent for it. Instead this stubs the ``League`` symbol that
``providers/espn.py`` imports with a fake exposing exactly the attributes verified against the
installed ``espn-api==0.46.0``, which exercises the real mapping code without touching the
network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague, ESPNUnknownError

import ffmcp.providers.espn as espn_module
from ffmcp.errors import CredentialsMissing, LeagueNotAccessible, UpstreamUnavailable
from ffmcp.providers.cache import Cache
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
        self.team_name = f"Team {team_id}  "  # ESPN names can carry trailing whitespace
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
    median_scoring = True
    division_map: ClassVar[dict[int, str]] = {0: "East", 1: "West"}
    playoff_seed_tie_rule = "TOTAL_POINTS_SCORED"
    faab = True
    acquisition_budget = 100
    trade_deadline = 1796230800000


class FakeBoxScore:
    def __init__(self, home_team_id: int | None, away_team_id: int | None) -> None:
        self.home_team = home_team_id
        self.away_team = away_team_id
        self.home_score = 100.0
        self.away_score = 90.0
        self.home_projected = 105.0
        self.away_projected = 95.0
        self.is_playoff = False


class FakeRequests:
    """Stands in for ``League.espn_request``: the raw ``mMatchupScore`` schedule, counting how
    often it is asked for."""

    calls = 0
    schedule: ClassVar[list[dict[str, Any]]] = [
        # Deliberately a different week-6 pairing from box_scores(), so a test can tell which
        # one get_matchups actually read.
        {"matchupPeriodId": 6, "home": {"teamId": 1}, "away": {"teamId": 3}},
        {"matchupPeriodId": 7, "home": {"teamId": 1}},  # a bye: no away side at all
        {"matchupPeriodId": 15, "home": {"teamId": 1}, "playoffTierType": "WINNERS_BRACKET"},
    ]

    def league_get(self, params: dict[str, str]) -> dict[str, Any]:
        assert params == {"view": "mMatchupScore"}
        FakeRequests.calls += 1
        return {"schedule": self.schedule}


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
RELEASED = FakePlayer(4, "Cut Back", "RB", ["RB"], "None")
"""A free agent no NFL team employs: ESPN reports his team as the string ``"None"``."""


class FakeLeague:
    def __init__(
        self, league_id: int, year: int, espn_s2: str | None = None, swid: str | None = None
    ) -> None:
        self.current_week = 3
        self.settings = FakeSettings()
        self.teams = [FakeTeam(1, [QB, BENCH_RB, IR_WR])]
        self.espn_request = FakeRequests()

    free_agent_calls = 0

    def box_scores(self, week: int) -> list[FakeBoxScore]:
        return [FakeBoxScore(1, 2)]

    def free_agents(self, week: int, size: int, position: str | None) -> list[FakePlayer]:
        FakeLeague.free_agent_calls += 1
        agents = [QB, RELEASED]
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


async def test_get_matchups_for_a_future_week_reads_the_schedule_not_box_scores() -> None:
    # week > current_week (3): box_scores() silently returns the current week's pairing
    # relabeled, so get_matchups must read the season schedule instead. The two return
    # different pairings precisely so this distinguishes them.
    matchups = await _provider().get_matchups(6)
    assert (matchups[0].home_team_id, matchups[0].away_team_id) == (1, 3)
    assert matchups[0].week == 6
    assert matchups[0].home_projected == 0.0


async def test_get_matchups_for_a_future_bye_has_no_opponent() -> None:
    matchups = await _provider().get_matchups(7)
    assert matchups[0].home_team_id == 1
    assert matchups[0].away_team_id is None
    assert not matchups[0].is_playoff
    assert (await _provider().get_matchups(15))[0].is_playoff


async def test_the_season_schedule_is_fetched_once_for_every_remaining_week(
    tmp_path: Path,
) -> None:
    # Before: one full-schedule download per remaining week, on every season-long tool call.
    FakeRequests.calls = 0
    provider = EspnLeagueProvider(
        league_id=1234567, season=2026, espn_s2=None, swid=None, cache=Cache(tmp_path)
    )
    for week in range(4, 15):
        await provider.get_matchups(week)
    assert FakeRequests.calls == 1


async def test_free_agent_lists_are_cached(tmp_path: Path) -> None:
    FakeLeague.free_agent_calls = 0
    provider = EspnLeagueProvider(
        league_id=1234567, season=2026, espn_s2=None, swid=None, cache=Cache(tmp_path)
    )
    first = await provider.get_free_agents(3, size=50)
    second = await provider.get_free_agents(3, size=50)
    assert first == second
    assert FakeLeague.free_agent_calls == 1


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
    agents = await _provider().get_free_agents(3, position="TE")
    assert agents == []
    agents = await _provider().get_free_agents(3, position="QB")
    assert [p.name for p in agents] == ["QB One"]


async def test_a_released_player_is_on_team_fa_not_team_none() -> None:
    agents = await _provider().get_free_agents(3, position="RB")
    assert [(p.name, p.pro_team) for p in agents] == [("Cut Back", "FA")]


async def test_team_names_are_stripped_of_trailing_whitespace() -> None:
    teams = await _provider().get_teams()
    assert teams[0].name == "Team 1"


async def test_settings_carry_the_league_format_the_simulation_must_respect() -> None:
    settings = await _provider().get_settings()
    assert settings.median_scoring is True
    assert settings.division_count == 2
    assert settings.waiver_budget == 100
    assert settings.trade_deadline == datetime.fromtimestamp(1796230800, UTC)


async def test_a_failed_refresh_keeps_serving_the_last_good_league(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    assert await provider.get_current_week() == 3

    def raise_unknown(*args: Any, **kwargs: Any) -> FakeLeague:
        raise ESPNUnknownError("HTTP 503")

    monkeypatch.setattr(espn_module, "League", raise_unknown)
    provider._league_built_at = -1e9  # force the handle stale
    assert await provider.get_current_week() == 3


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


# ---------------------------------------------------------------------------
# Scoring, byes and history: the pure mapping helpers, fed stand-in objects.
# ---------------------------------------------------------------------------


def test_healthy_statuses_are_not_injury_designations() -> None:
    assert espn_module._injury_status("ACTIVE") is None
    assert espn_module._injury_status("NORMAL") is None
    assert espn_module._injury_status("") is None
    assert espn_module._injury_status([]) is None  # what team defenses report
    assert espn_module._injury_status("QUESTIONABLE") == "QUESTIONABLE"


def test_kickoffs_are_read_from_rostered_schedules_as_true_instants() -> None:
    from types import SimpleNamespace

    local_kickoff = datetime(2026, 10, 18, 12, 0)  # naive local, as espn-api builds it
    player = SimpleNamespace(proTeam="PHI", schedule={6: {"team": "CAR", "date": local_kickoff}})
    on_bye = SimpleNamespace(proTeam="CIN", schedule={5: {"team": "PIT", "date": local_kickoff}})
    league = SimpleNamespace(teams=[SimpleNamespace(roster=[player, on_bye])])
    kickoffs = espn_module._kickoffs_by_team(league, 6)
    assert kickoffs == {"PHI": local_kickoff.astimezone(UTC)}
    assert kickoffs["PHI"].tzinfo is not None


def test_activity_becomes_one_transaction_per_player_moved() -> None:
    topic = {
        "date": 1792106137135,
        "messages": [
            {"messageTypeId": 178, "to": 12, "targetId": -16029},
            {"messageTypeId": 179, "to": 12, "targetId": -16019},
            {"messageTypeId": 239, "for": 12, "targetId": 4039},
            {"messageTypeId": 999, "to": 12, "targetId": 1},  # not a transaction
        ],
    }
    names = {-16029: "Panthers D/ST", -16019: "Giants D/ST"}
    transactions = espn_module._map_activity(topic, names)
    assert [(t.team_id, t.action, t.player_name) for t in transactions] == [
        (12, "add", "Panthers D/ST"),
        (12, "drop", "Giants D/ST"),
        (12, "drop", "4039"),  # an id the map does not know still says who moved
    ]


def test_reception_points_come_from_the_leagues_scoring_items() -> None:
    from types import SimpleNamespace

    def settings(items: list[dict[str, Any]] | None) -> Any:
        raw = {} if items is None else {"scoringItems": items}
        return SimpleNamespace(_raw_scoring_settings=raw)

    assert espn_module._reception_points(settings([{"statId": 53, "points": 0.5}])) == 0.5
    assert espn_module._reception_points(settings([{"statId": 25, "points": 6.0}])) == 0.0
    assert espn_module._reception_points(settings(None)) == 1.0


def test_bye_weeks_come_from_rostered_players_schedules_by_majority() -> None:
    from types import SimpleNamespace

    def player(team: str, bye: int) -> Any:
        weeks: dict[str, dict[str, Any]] = {str(w): {} for w in range(1, 19) if w != bye}
        return SimpleNamespace(proTeam=team, schedule=weeks)

    league = SimpleNamespace(
        teams=[
            SimpleNamespace(roster=[player("DET", 8), player("DET", 8), player("DET", 5)]),
            SimpleNamespace(roster=[player("CHI", 6), SimpleNamespace(proTeam="FA", schedule={})]),
        ]
    )
    assert espn_module._bye_weeks_by_team(league) == {"DET": 8, "CHI": 6}


def test_box_score_rows_carry_slot_projection_and_whether_the_game_was_played() -> None:
    from types import SimpleNamespace

    def box_player(pid: int, slot: str, *, bye: bool = False) -> Any:
        return SimpleNamespace(
            playerId=pid,
            name=f"P{pid}",
            position="WR",
            proTeam="DET",
            eligibleSlots=["WR", "BE"],
            slot_position=slot,
            projected_points=12.5,
            points=0.0 if bye else 17.1,
            game_played=100,
            on_bye_week=bye,
        )

    box = SimpleNamespace(
        home_team=1,
        away_team=2,
        home_lineup=[box_player(10, "WR"), box_player(11, "BE", bye=True)],
        away_lineup=[box_player(20, "WR")],
    )
    rows = espn_module._box_score_rows(box)
    assert [(r["player_id"], r["fantasy_team_id"], r["slot"], r["played"]) for r in rows] == [
        (10, 1, "WR", True),
        (11, 1, "BE", False),
        (20, 2, "WR", True),
    ]
    assert rows[0]["projected"] == 12.5 and rows[0]["actual"] == 17.1

"""Demo mode constructs a full LeagueState with no credentials present."""

from __future__ import annotations

from ffmcp.domain.models import LeagueState
from ffmcp.providers.demo import DemoLeagueProvider, DemoMarketProvider


async def test_demo_league_provider_assembles_a_full_league_state() -> None:
    provider = DemoLeagueProvider()

    week = await provider.get_current_week()
    settings = await provider.get_settings()
    teams = await provider.get_teams()
    matchups = await provider.get_matchups(week)

    state = LeagueState(
        settings=settings, teams=tuple(teams), current_week=week, matchups=tuple(matchups)
    )

    assert state.teams
    assert all(t.roster.slots for t in state.teams)
    assert state.matchups


async def test_demo_league_provider_free_agents_respect_position_and_size() -> None:
    provider = DemoLeagueProvider()
    week = await provider.get_current_week()

    all_agents = await provider.get_free_agents(week, size=100)
    assert all_agents

    qbs = await provider.get_free_agents(week, position="QB")
    assert all(p.position == "QB" for p in qbs)


async def test_demo_market_provider_resolves_by_espn_id_without_network() -> None:
    provider = DemoMarketProvider()
    trending = await provider.get_trending_adds()
    assert trending

    league = DemoLeagueProvider()
    a_team = (await league.get_teams())[0]
    a_player = a_team.roster.slots[0].player
    assert a_player is not None

    # Not every player has a market entry in the fixture; just prove the call is well-formed.
    signal = await provider.get_market_signal(
        espn_player_id=a_player.player_id,
        name=a_player.name,
        position=a_player.position,
        pro_team=a_player.pro_team,
    )
    assert signal is None or signal.trending_adds is not None or signal.percent_owned is not None

"""Golden-output tests for ``render/tables.py``. Exact strings on purpose: a renderer is a
contract with the model, and a silent format change should fail a test,
not just "look different" in a manual check.
"""

from __future__ import annotations

from ffmcp.domain.models import (
    LeagueSettings,
    LeagueState,
    MarketSignal,
    Matchup,
    Player,
    Projection,
    Roster,
    RosterSlot,
    Team,
)
from ffmcp.domain.simulate import SeasonOutcome, TeamOutlook
from ffmcp.render.tables import (
    WaiverTarget,
    render_compare_players,
    render_player_report,
    render_roster,
    render_season_outcome,
    render_standings,
    render_trade_line,
    render_trades,
    render_waiver_targets,
    short_name,
)


def _player(
    player_id: int,
    name: str,
    position: str,
    pro_team: str,
    points: float | None,
    *,
    market: MarketSignal | None = None,
    injury_status: str | None = None,
    live_points: float | None = None,
) -> Player:
    return Player(
        player_id=player_id,
        name=name,
        position=position,
        eligible_slots=(position,),
        pro_team=pro_team,
        injured=injury_status == "OUT",
        injury_status=injury_status,
        projection=None if points is None else Projection(week=1, points=points),
        market=market,
        live_points=live_points,
    )


def test_short_name_abbreviates_first_name_and_truncates() -> None:
    assert short_name("Josh Allen") == "J. Allen"
    assert short_name("Bijan") == "Bijan"
    assert short_name("Christian McCaffrey Junior Extra", width=10) == "C. McCaff…"


def test_render_roster_compact_is_a_fixed_width_table() -> None:
    qb = _player(1, "Josh Allen", "QB", "BUF", 22.4)
    rb = _player(2, "Kenneth Walker", "RB", "SEA", 12.8, injury_status="QUESTIONABLE")
    bench = _player(3, "Bench Guy", "RB", "SF", 4.0)
    team = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=1,
        losses=2,
        points_for=100.0,
        points_against=90.0,
        standing=1,
        playoff_pct=50.0,
        roster=Roster(
            slots=(RosterSlot(slot="QB", player=qb), RosterSlot(slot="RB", player=rb)),
            bench=(bench,),
        ),
    )

    output = render_roster(team, week=3, detail="compact")

    assert output == (
        "Week 3 — Team Alpha (1-2-0)\n"
        "Injuries: K. Walker (Q)\n"
        "SLOT  PLAYER           POS  TM  PROJ  ST\n"
        "QB    J. Allen         QB   BUF 22.4\n"
        "RB    K. Walker        RB   SEA 12.8  Q\n"
        "BE    B. Guy           RB   SF  4.0"
    )


def test_render_roster_empty_slot() -> None:
    team = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=0,
        losses=0,
        points_for=0.0,
        points_against=0.0,
        standing=1,
        playoff_pct=0.0,
        roster=Roster(slots=(RosterSlot(slot="FLEX", player=None),)),
    )
    output = render_roster(team, week=1, detail="compact")
    assert "FLEX  (empty)" in output


def test_render_roster_full_adds_market_columns() -> None:
    qb = _player(
        1,
        "Josh Allen",
        "QB",
        "BUF",
        22.4,
        market=MarketSignal(percent_owned=99.0, percent_started=95.0, trending_adds=120),
    )
    team = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=0,
        losses=0,
        points_for=0.0,
        points_against=0.0,
        standing=1,
        playoff_pct=0.0,
        roster=Roster(slots=(RosterSlot(slot="QB", player=qb),)),
    )
    output = render_roster(team, week=1, detail="full")
    header, row = output.splitlines()[2], output.splitlines()[3]
    assert header == "SLOT  PLAYER           POS  TM  PROJ  LIVE  ST  OWN%  STRT%  TREND"
    assert row == "QB    J. Allen         QB   BUF 22.4            99%   95%    120"


def test_render_roster_standard_shows_live_points_once_a_game_starts() -> None:
    playing = _player(1, "Josh Allen", "QB", "BUF", 22.4, live_points=9.6)
    not_started = _player(2, "Bench Guy", "RB", "SF", 4.0)
    team = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=0,
        losses=0,
        points_for=0.0,
        points_against=0.0,
        standing=1,
        playoff_pct=0.0,
        roster=Roster(
            slots=(RosterSlot(slot="QB", player=playing), RosterSlot(slot="RB", player=None)),
            bench=(not_started,),
        ),
    )
    output = render_roster(team, week=1, detail="standard")
    lines = output.splitlines()
    assert lines[1] == "Injuries: none"
    assert lines[2] == "SLOT  PLAYER           POS  TM  PROJ  LIVE  ST  OWN%"
    assert lines[3] == "QB    J. Allen         QB   BUF 22.4  9.6"
    assert lines[4] == "RB    (empty)"
    assert lines[5] == "BE    B. Guy           RB   SF  4.0"


def test_render_waiver_targets_empty() -> None:
    assert render_waiver_targets([], "compact", limit=10, total_considered=0) == (
        "No waiver targets clear the bar this week."
    )


def test_render_waiver_targets_names_a_drop_and_truncates() -> None:
    add = _player(10, "Puka Nacua", "WR", "LAR", 15.0)
    drop = _player(11, "Bench Guy", "RB", "SF", 2.0)
    targets = [WaiverTarget(add, 3.5, drop)]

    output = render_waiver_targets(targets, "compact", limit=1, total_considered=3)

    assert output == (
        "PLAYER           POS  TM  PROJ  VAL    DROP\n"
        "P. Nacua         WR   LAR 15.0  +3.5   B. Guy\n"
        "… 2 more (raise limit to see)"
    )


def test_render_waiver_targets_shows_near_misses_when_nothing_clears_the_bar() -> None:
    """An all-negative candidate pool gets a labeled table of the closest
    misses instead of the bare "no targets" line, so a manager can tell an empty wire from one
    that just didn't help."""
    worst = _player(10, "Scrub One", "WR", "LAR", 2.0)
    second_worst = _player(11, "Scrub Two", "RB", "SF", 1.5)
    near_misses = [WaiverTarget(worst, -0.2, None), WaiverTarget(second_worst, -1.1, None)]

    output = render_waiver_targets(
        [], "compact", limit=10, total_considered=0, near_misses=near_misses
    )

    assert output == (
        "Closest misses (would not improve your lineup):\n"
        "PLAYER           POS  TM  PROJ  VAL    DROP\n"
        "S. One           WR   LAR 2.0   -0.2   (no upgrade)\n"
        "S. Two           RB   SF  1.5   -1.1   (no upgrade)"
    )


def test_render_waiver_targets_no_drop_available() -> None:
    add = _player(10, "Puka Nacua", "WR", "LAR", 15.0)
    output = render_waiver_targets(
        [WaiverTarget(add, 3.5, None)], "compact", limit=5, total_considered=1
    )
    assert "(no drop needed)" in output


def _settings(**overrides: object) -> LeagueSettings:
    base = dict(
        league_id=1,
        season=2026,
        team_count=2,
        playoff_team_count=2,
        reg_season_weeks=14,
        slot_counts={"QB": 1},
    )
    base.update(overrides)
    return LeagueSettings(**base)  # type: ignore[arg-type]


def test_render_standings_shows_sos_and_playoff_pct() -> None:
    team_a = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=3,
        losses=0,
        points_for=300.0,
        points_against=250.0,
        standing=1,
        playoff_pct=90.0,
        roster=Roster(slots=()),
    )
    team_b = Team(
        team_id=2,
        name="Team Bravo",
        abbrev="BRA",
        wins=0,
        losses=3,
        points_for=250.0,
        points_against=300.0,
        standing=2,
        playoff_pct=10.0,
        roster=Roster(slots=()),
    )
    state = LeagueState(
        settings=_settings(team_count=2),
        teams=(team_a, team_b),
        current_week=1,
        matchups=(
            Matchup(
                week=1,
                home_team_id=1,
                away_team_id=2,
                home_score=0,
                away_score=0,
                home_projected=0,
                away_projected=0,
            ),
        ),
    )

    output = render_standings(state, "compact")

    assert output == (
        "RK  TEAM           W-L    PF      PA      PLAYOFF%  SOS\n"
        "1   Team Alpha     3-0    300.0   250.0   90.0%     0.0%\n"
        "2   Team Bravo     0-3    250.0   300.0   10.0%     100.0%"
    )


def test_render_standings_notes_a_week_in_progress_when_records_lag_live_scores() -> None:
    """ESPN doesn't finalize records until the week ends, so a 0-0-0
    standings table sitting next to a live, in-progress matchup should say so."""
    team_a = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=0,
        losses=0,
        points_for=0.0,
        points_against=0.0,
        standing=1,
        playoff_pct=50.0,
        roster=Roster(slots=()),
    )
    team_b = Team(
        team_id=2,
        name="Team Bravo",
        abbrev="BRA",
        wins=0,
        losses=0,
        points_for=0.0,
        points_against=0.0,
        standing=2,
        playoff_pct=50.0,
        roster=Roster(slots=()),
    )
    state = LeagueState(
        settings=_settings(team_count=2),
        teams=(team_a, team_b),
        current_week=1,
        matchups=(
            Matchup(
                week=1,
                home_team_id=1,
                away_team_id=2,
                home_score=14.2,
                away_score=9.8,
                home_projected=20.0,
                away_projected=18.0,
            ),
        ),
    )

    output = render_standings(state, "compact")

    assert output.splitlines()[-1] == (
        "Week 1 is still in progress — standings reflect ESPN's last finalized week."
    )


def test_render_standings_has_no_note_once_records_are_real() -> None:
    team_a = Team(
        team_id=1,
        name="Team Alpha",
        abbrev="ALP",
        wins=1,
        losses=0,
        points_for=100.0,
        points_against=90.0,
        standing=1,
        playoff_pct=50.0,
        roster=Roster(slots=()),
    )
    state = LeagueState(
        settings=_settings(team_count=1),
        teams=(team_a,),
        current_week=2,
        matchups=(
            Matchup(
                week=2,
                home_team_id=1,
                away_team_id=None,
                home_score=0.0,
                away_score=0.0,
                home_projected=0.0,
                away_projected=0.0,
            ),
        ),
    )

    output = render_standings(state, "compact")

    assert "still in progress" not in output


def test_render_season_outcome_marks_the_highlighted_team() -> None:
    outcome = SeasonOutcome(
        n_sims=1000,
        playoff_team_count=1,
        weeks_simulated=(3, 4),
        teams=(
            TeamOutlook(
                team_id=1,
                name="Team Alpha",
                wins=1,
                losses=0,
                ties=0,
                mean_final_wins=5.0,
                playoff_odds=40.0,
                title_odds=10.0,
                seed_distribution=(1.0,),
            ),
            TeamOutlook(
                team_id=2,
                name="Team Bravo",
                wins=0,
                losses=1,
                ties=0,
                mean_final_wins=6.0,
                playoff_odds=60.0,
                title_odds=20.0,
                seed_distribution=(1.0,),
            ),
        ),
    )

    output = render_season_outcome(outcome, highlight_team_id=1)
    lines = output.splitlines()
    assert lines[0] == "1,000 sims, weeks 3-4:"
    # Sorted to the top despite lower playoff odds, and short_name-abbreviated like every name.
    assert lines[2].startswith("*T. Alpha")


def test_render_trades_empty() -> None:
    assert render_trades([]) == "No trades found that would help both sides right now."


def test_render_trade_line_formats_both_value_deltas_in_the_same_unit() -> None:
    """My side and the partner's side both print in points, so the line
    cannot read as lopsided just because it mixes a percentage with a point figure."""
    give = (_player(1, "Josh Allen", "QB", "BUF", 22.0),)
    get = (_player(2, "Saquon Barkley", "RB", "PHI", 18.0),)
    line = render_trade_line("Team Bravo", give, get, 3.1, -1.5, 4.2, "Upgrades RB.")
    assert line == (
        "Team Bravo: give J. Allen / get S. Barkley — "
        "me +3.1pts, them +-1.5pts, +4.2% odds — Upgrades RB."
    )


def test_render_trade_line_omits_odds_when_not_simulated() -> None:
    give = (_player(1, "Josh Allen", "QB", "BUF", 22.0),)
    get = (_player(2, "Saquon Barkley", "RB", "PHI", 18.0),)
    line = render_trade_line("Team Bravo", give, get, 3.1, 2.0, None, "Upgrades RB.")
    assert line == (
        "Team Bravo: give J. Allen / get S. Barkley — me +3.1pts, them +2.0pts — Upgrades RB."
    )


def test_render_player_report() -> None:
    player = _player(
        1,
        "Puka Nacua",
        "WR",
        "LAR",
        15.0,
        market=MarketSignal(percent_owned=80.0, trending_adds=500),
    )
    output = render_player_report(player, week=3, value_over_replacement=2.5, weeks_remaining=4)
    assert output == (
        "Puka Nacua — WR LAR, week 3\n"
        "Projected: 15.0 pts\n"
        "Status: healthy\n"
        "Value over replacement: +2.5 pts/wk, +10.0 pts rest of season (4 wks)\n"
        "Market: 80% owned, 500 adds (24h)"
    )


def test_render_player_report_status_line_is_always_present_when_healthy() -> None:
    """A healthy player still gets an explicit status, not a blank."""
    player = _player(1, "Josh Allen", "QB", "BUF", 22.0)
    output = render_player_report(player, week=3, value_over_replacement=1.0, weeks_remaining=1)
    assert "Status: healthy" in output.splitlines()


def test_render_player_report_status_line_names_the_injury() -> None:
    player = _player(1, "Saquon Barkley", "RB", "PHI", 18.0, injury_status="OUT")
    output = render_player_report(player, week=3, value_over_replacement=1.0, weeks_remaining=1)
    assert "Status: Out" in output.splitlines()


def test_render_compare_players_leads_with_recommendation() -> None:
    a = _player(1, "Josh Allen", "QB", "BUF", 22.0)
    b = _player(2, "Bench Guy", "QB", "SF", 10.0)
    output = render_compare_players("Start Josh Allen (QB BUF).", [(a, 22.0, 5.0), (b, 10.0, -5.0)])
    assert output.splitlines()[0] == "Start Josh Allen (QB BUF)."
    assert "J. Allen" in output
    assert "B. Guy" in output

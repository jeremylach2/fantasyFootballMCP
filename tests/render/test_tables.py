"""Golden-output tests for ``render/tables.py``. Exact strings on purpose: a renderer is a
contract with the model, and a silent format change should fail a test,
not just "look different" in a manual check.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ffmcp.domain.models import (
    GameLine,
    LeagueSettings,
    LeagueState,
    MarketSignal,
    Matchup,
    Player,
    Projection,
    Roster,
    RosterSlot,
    Team,
    Transaction,
)
from ffmcp.domain.roster import ByeWeek, DropCandidate, Pickup
from ffmcp.domain.simulate import SeasonOutcome, TeamOutlook
from ffmcp.render.tables import (
    deadline_note,
    defense_matchups,
    fmt_count,
    render_activity,
    render_compare_players,
    render_player_report,
    render_roster,
    render_roster_plan,
    render_season_outcome,
    render_standings,
    render_trade_line,
    render_trades,
    render_waiver_targets,
    second_opinion,
    short_name,
    start_sit_sentence,
    status_flag,
    team_label,
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


def test_team_names_are_truncated_never_initialised() -> None:
    # Through short_name these became "B. Team Ever" and "0. out of 5".
    assert team_label("Best Team Ever") == "Best Team Ever"
    assert team_label("0 out of 5") == "0 out of 5"
    assert team_label("Down Bad In The Standings Again", 14) == "Down Bad In T…"
    assert team_label("Syndrome  ") == "Syndrome"


def test_a_zero_projection_in_a_players_bye_week_is_a_bye() -> None:
    # ESPN projects 0.0 for a player on bye, not nothing; the flag reads the bye week.
    on_bye = _player(1, "Vikings D/ST", "D/ST", "MIN", 0.0).model_copy(update={"bye_week": 1})
    assert status_flag(on_bye, 0.0, 1) == "BYE"
    assert status_flag(on_bye, 0.0, 2) == ""


def test_big_counts_render_compactly() -> None:
    assert fmt_count(126385) == "126k"
    assert fmt_count(950) == "950"
    assert fmt_count(1_450_000) == "1.4M"
    assert fmt_count(None) == ""


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
            # ESPN's own order puts RB before QB; the table does not.
            slots=(RosterSlot(slot="RB", player=rb), RosterSlot(slot="QB", player=qb)),
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


def test_render_roster_flags_locked_players() -> None:
    qb = _player(1, "Josh Allen", "QB", "BUF", 22.4)
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
    output = render_roster(team, week=1, detail="compact", locked={1})
    assert output.splitlines()[-1] == "QB    J. Allen         QB   BUF 22.4  LCK"


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
        market=MarketSignal(percent_owned=99.0, percent_started=95.0, trending_adds=126385),
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
    assert header == "SLOT  PLAYER           POS  TM  PROJ  LIVE  ST  OWN%  STRT%  ADD/24H"
    assert row == "QB    J. Allen         QB   BUF 22.4            99%   95%    126k"


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
    assert render_waiver_targets([], "compact", limit=10) == "No waiver targets clear the bar."


def test_render_waiver_targets_ranks_by_season_and_names_a_drop() -> None:
    add = _player(10, "Puka Nacua", "WR", "LAR", 15.0)
    drop = _player(11, "Bench Guy", "RB", "SF", 2.0)
    pickups = [Pickup(player=add, week_value=3.5, season_value=41.2, drop=drop)]

    output = render_waiver_targets(pickups, "compact", limit=1)

    assert output == (
        "PLAYER           POS  TM  PROJ  ROS   VAL   DROP\n"
        "P. Nacua         WR   LAR 15.0  +41   +3.5  B. Guy\n"
        "(raise limit to see more)"
    )


def test_render_waiver_targets_appends_handcuff_notes() -> None:
    output = render_waiver_targets(
        [],
        "compact",
        limit=10,
        handcuffs=["J. Gibbs -> D. Montgomery (free agent, 41% snaps)"],
    )
    assert output == (
        "No waiver targets clear the bar.\n"
        "Handcuffs: J. Gibbs -> D. Montgomery (free agent, 41% snaps)"
    )


def test_render_waiver_targets_shows_near_misses_when_nothing_helps_the_season() -> None:
    """A wire where nothing helps gets the closest moves, so a manager can tell an empty wire
    from one that came a point short."""
    scrub = _player(10, "Scrub One", "WR", "LAR", 2.0)
    cut = _player(12, "Bench Guy", "RB", "SF", 4.0)
    near_misses = [Pickup(player=scrub, week_value=0.0, season_value=-3.4, drop=cut)]

    output = render_waiver_targets([], "compact", limit=10, near_misses=near_misses)

    assert output == (
        "No move helps your season. Closest:\n"
        "PLAYER           POS  TM  PROJ  ROS   VAL   DROP\n"
        "S. One           WR   LAR 2.0   -3    +0.0  B. Guy"
    )


def test_render_waiver_targets_lists_ir_stashes_with_their_return() -> None:
    back = _player(20, "Jonathon Brooks", "RB", "CAR", 0.0, injury_status="INJURY_RESERVE")
    back = back.model_copy(
        update={
            "market": MarketSignal(percent_owned=41.0),
            "outlook": (Projection(week=7, points=0.0), Projection(week=9, points=5.8)),
        }
    )
    gone = _player(21, "Season Over", "WR", "GB", 0.0, injury_status="INJURY_RESERVE")
    stashes = [
        Pickup(player=back, week_value=0.0, season_value=6.2),
        Pickup(player=gone, week_value=0.0, season_value=0.0),
    ]
    output = render_waiver_targets([], "compact", limit=5, stashes=stashes, open_ir_slots=1)
    assert output.splitlines()[-1] == (
        "Stash (1 open IR slot): J. Brooks RB CAR (41% owned, back wk 9, ROS +6); "
        "S. Over WR GB (no return projected)"
    )


def test_render_waiver_targets_no_drop_needed() -> None:
    add = _player(10, "Puka Nacua", "WR", "LAR", 15.0)
    output = render_waiver_targets(
        [Pickup(player=add, week_value=3.5, season_value=12.0)], "compact", limit=5
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

    # The simulation's odds, not ESPN's own (90/10 on the Team objects).
    output = render_standings(state, "compact", {1: 81.5, 2: 18.5})

    assert output == (
        "RK  ID  TEAM             W-L    PF      PA      PLAYOFF%  SOS\n"
        "1   1   Team Alpha       3-0    300.0   250.0   81.5%     0.0%\n"
        "2   2   Team Bravo       0-3    250.0   300.0   18.5%     100.0%"
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
    # Sorted to the top despite lower playoff odds, and named in full, not initialised.
    assert lines[2].startswith("*Team Alpha")


def test_render_trades_empty() -> None:
    assert render_trades([]) == "No trades found that would help both sides right now."


def test_render_trade_line_formats_both_value_deltas_in_the_same_unit() -> None:
    """My side and the partner's side both print in points, so the line
    cannot read as lopsided just because it mixes a percentage with a point figure."""
    give = (_player(1, "Josh Allen", "QB", "BUF", 22.0),)
    get = (_player(2, "Saquon Barkley", "RB", "PHI", 18.0),)
    line = render_trade_line("Team Bravo", 2, give, get, 3.1, -1.5, 4.2, "Upgrades RB.")
    assert line == (
        "Team Bravo (id 2): give J. Allen / get S. Barkley — "
        "me +3.1pts, them -1.5pts, +4.2% odds — Upgrades RB."
    )


def test_render_trade_line_shows_market_values_from_my_side() -> None:
    give = (_player(1, "Jacoby Brissett", "QB", "ARI", 18.0),)
    get = (_player(2, "Mike Evans", "WR", "TB", 14.0),)
    line = render_trade_line(
        "Team Bravo", 2, give, get, 53.6, 3.9, 3.6, "Strengthens WR.", (267.0, 3190.0)
    )
    assert "— mkt 267 vs 3,190 —" in line


def test_deadline_note_counts_days_or_says_it_passed() -> None:
    deadline = datetime(2026, 12, 2, 17, 0, tzinfo=UTC)
    assert deadline_note(deadline, datetime(2026, 10, 15, tzinfo=UTC)) == (
        "Trade deadline in 48 days (Dec 2)."
    )
    assert deadline_note(deadline, datetime(2026, 12, 3, tzinfo=UTC)) == (
        "Trade deadline passed (Dec 2)."
    )
    assert deadline_note(None, datetime(2026, 12, 3, tzinfo=UTC)) is None


def test_render_trade_line_omits_odds_when_not_simulated() -> None:
    give = (_player(1, "Josh Allen", "QB", "BUF", 22.0),)
    get = (_player(2, "Saquon Barkley", "RB", "PHI", 18.0),)
    line = render_trade_line("Team Bravo", 2, give, get, 3.1, 2.0, None, "Upgrades RB.")
    assert line == (
        "Team Bravo (id 2): give J. Allen / get S. Barkley — "
        "me +3.1pts, them +2.0pts — Upgrades RB."
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
    output = render_player_report(player, week=3, value_over_replacement=2.5, effective_weeks=4)
    assert output == (
        "Puka Nacua — WR LAR, week 3\n"
        "Projected: 15.0 pts\n"
        "Status: healthy\n"
        "Value over replacement: +2.5 pts/wk, +10.0 pts rest of season (4.0 wks, byes and "
        "playoff odds counted)\n"
        "Market: 80% owned, 500 adds (24h)"
    )


def test_render_player_report_shows_range_second_opinion_and_bye() -> None:
    player = _player(1, "Puka Nacua", "WR", "LAR", 15.0).model_copy(update={"bye_week": 9})
    output = render_player_report(
        player,
        week=3,
        value_over_replacement=2.5,
        effective_weeks=4,
        sd=6.0,
        alternative=9.0,
    )
    lines = output.splitlines()
    assert lines[1] == "Projected: 15.0 pts (range 7.3-22.7, p10-p90)"
    assert lines[2] == "Second opinion: Sleeper 9.0; sources disagree by 40%, so the range is wider"
    assert lines[3] == "Status: healthy · bye week 9"


def test_render_player_report_values_a_bye_week_player_at_his_season_rate() -> None:
    player = _player(1, "Puka Nacua", "WR", "LAR", None).model_copy(
        update={"bye_week": 3, "season_rate": 14.2}
    )
    output = render_player_report(player, week=3, value_over_replacement=1.0, effective_weeks=4)
    assert "Projected: no game this week (bye, season rate 14.2 pts/g)" in output.splitlines()


def test_render_player_report_status_line_is_always_present_when_healthy() -> None:
    """A healthy player still gets an explicit status, not a blank."""
    player = _player(1, "Josh Allen", "QB", "BUF", 22.0)
    output = render_player_report(player, week=3, value_over_replacement=1.0, effective_weeks=1)
    assert "Status: healthy" in output.splitlines()


def test_render_player_report_says_bye_with_the_week_when_espn_projects_zero() -> None:
    player = _player(1, "Vikings D/ST", "D/ST", "MIN", 0.0).model_copy(
        update={"bye_week": 3, "season_rate": 5.4, "projection": Projection(week=3, points=0.0)}
    )
    output = render_player_report(player, week=3, value_over_replacement=0.5, effective_weeks=4)
    lines = output.splitlines()
    assert "Projected: no game this week (bye, season rate 5.4 pts/g)" in lines
    assert "Status: Bye (week 3)" in lines


def test_second_opinion_says_espn_expects_him_out_rather_than_a_percentage() -> None:
    # 10.7 against 0.0 used to print "sources disagree by 1070%".
    assert second_opinion(0.0, 10.7) == (
        "ESPN projects 0 (expects him out); Sleeper 10.7. Check final inactives 90 min "
        "before kickoff."
    )
    assert second_opinion(4.0, 12.0) == (
        "Second opinion: Sleeper 12.0; sources disagree sharply, so the range is wider"
    )
    assert second_opinion(10.0, 10.5) == "Second opinion: Sleeper 10.5"


def test_render_player_report_shows_an_injured_players_return() -> None:
    player = _player(1, "Jonathon Brooks", "RB", "CAR", 0.0, injury_status="INJURY_RESERVE")
    player = player.model_copy(
        update={
            "outlook": (
                Projection(week=7, points=0.0),
                Projection(week=9, points=6.0),
                Projection(week=10, points=6.4),
            )
        }
    )
    output = render_player_report(
        player, week=6, value_over_replacement=0.0, effective_weeks=8, trade_value=410.0
    )
    lines = output.splitlines()
    assert "Outlook: back week 9 (Sleeper), 6.2 pts/g after" in lines
    assert lines[-1] == "Market: trade value 410 (FantasyCalc)"


def test_render_player_report_status_line_names_the_injury() -> None:
    player = _player(1, "Saquon Barkley", "RB", "PHI", 18.0, injury_status="OUT")
    output = render_player_report(player, week=3, value_over_replacement=1.0, effective_weeks=1)
    assert "Status: Out" in output.splitlines()


def test_start_sit_sentence_names_each_slot() -> None:
    swift = _player(1, "D'Andre Swift", "RB", "CHI", 15.0)
    kittle = _player(2, "George Kittle", "TE", "SF", 14.8)
    barkley = _player(3, "Saquon Barkley", "RB", "PHI", 0.0)
    assert start_sit_sentence([(swift, "RB"), (kittle, "RB/WR/TE")], [barkley]) == (
        "Start D. Swift (RB), G. Kittle (FLEX); sit S. Barkley."
    )
    assert start_sit_sentence([], [barkley]) == "Sit S. Barkley."


def test_render_roster_plan_names_bye_holes_and_cheap_drops() -> None:
    love = _player(1, "Jordan Love", "QB", "GB", 25.7)
    boswell = _player(2, "Chris Boswell", "K", "PIT", 8.8)
    mcpherson = _player(3, "Evan McPherson", "K", "CIN", 0.0)
    byes = [ByeWeek(week=11, projected=98.1, typical=119.6, missing=(love,))]
    drops = [
        DropCandidate(
            player=mcpherson, season_value=7.7, weeks_started=(9,), covers=((9, boswell),)
        )
    ]
    output = render_roster_plan(byes, drops, typical=119.6)
    lines = output.splitlines()
    assert lines[0] == "Bye weeks ahead (typical week 119.6):"
    assert lines[1] == "  Wk 11: J. Love out -> 98.1 (-21.5)"
    assert lines[-1].startswith("E. McPherson     K    CIN -7.7   only starts to cover C. Boswell")


def test_render_activity_groups_one_teams_moves() -> None:
    when = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)
    moves = [
        Transaction(when=when, team_id=12, action="add", player_name="Panthers D/ST"),
        Transaction(when=when, team_id=12, action="drop", player_name="Giants D/ST"),
    ]
    output = render_activity(moves, {12: "Team Lima"}, limit=5)
    assert output == "Oct 15 Team Lima (id 12): add Panthers D/ST, drop Giants D/ST"


def test_render_compare_players_leads_with_recommendation() -> None:
    a = _player(1, "Josh Allen", "QB", "BUF", 22.0)
    b = _player(2, "Bench Guy", "QB", "SF", 10.0)
    output = render_compare_players("Start Josh Allen (QB BUF).", [(a, 22.0, 5.0), (b, 10.0, -5.0)])
    assert output.splitlines()[0] == "Start Josh Allen (QB BUF)."
    assert "J. Allen" in output
    assert "B. Guy" in output


def test_defense_matchups_show_the_opponents_implied_points() -> None:
    bucs = _player(1, "Buccaneers D/ST", "D/ST", "TB", 5.8)
    lines = {
        "TB": [
            GameLine(pro_team="TB", opponent="NYG", implied_total=24.0, spread=-6.0, total=42.0),
            GameLine(pro_team="TB", opponent="ATL", implied_total=23.0, spread=-1.5, total=44.5),
        ]
    }
    assert defense_matchups([bucs], lines) == (
        "Matchups (opponent implied pts, lower is better): Buccaneers D/ST NYG 18.0, ATL 21.5"
    )
    assert defense_matchups([bucs], {}) is None

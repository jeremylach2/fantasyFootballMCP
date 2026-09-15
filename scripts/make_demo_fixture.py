"""Generate the committed demo league (``tests/fixtures/demo_league.json`` and
``demo_market.json``) that ``FFMCP_MODE=demo`` serves.

This league is synthetic, and deliberately so. Publishing a real league means publishing other
people's rosters, so the demo ships an invented one instead: fictional players, invented
projections, real NFL team abbreviations for flavour. ``scripts/record_fixtures.py`` is the
other path: it anonymizes a *real* league for someone who has credentials and their league's
consent. Nothing here should ever be read as a real projection.

Everything is derived rather than typed, so the fixture is internally consistent and a reviewer
can re-derive it:

* the 14-week schedule is a circle-method round robin, and weeks 1-7 are *played*. Each
  team's record, points for and points against are the accumulated results of those games, so
  the standings agree with the schedule instead of being asserted alongside it;
* week 8 is the current week, leaving seven weeks for ``simulate_season`` to actually simulate;
* four NFL teams are on bye in week 8 (``BYE_TEAMS``), and their players carry no projection,
  which is how the optimizer's "unavailable" path gets exercised by ordinary data;
* ``playoff_pct`` on each team is filled in by running this repo's own simulator over the
  finished league, so the fixture's stored odds and ``simulate_season``'s output agree.

Three situations are placed on purpose to demonstrate the core features:

1. Team Alpha's lineup is wrong. It is Sunday morning and the manager has not set it: a
   ruled-out running back and a receiver on bye are still in the lineup, and the flex holds a
   tight end while a better back sits. ``optimize_lineup`` has something to say.
2. Team Alpha and Team Bravo fit each other. Alpha rosters two startable quarterbacks in a
   one-QB league, where the second scores nothing every week he sits, and a thin backfield;
   Bravo is the mirror image. That asymmetry is the entire reason trades exist, and
   ``find_trades`` should find it without being told.
3. It is now Monday morning and Team Alpha's games disagreed with Thursday's projections.
   The starting WR1 goes cold live while a bench receiver explodes, so the lineup that would
   have actually maximized points differs from both the stale lineup above *and* the pregame
   optimum. Every other player on the roster has no ``live_points`` yet, standing in for games
   that have not kicked off, so the recap has to fall back to the pregame projection for them.

    uv run poe fixtures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ffmcp.domain.models import LeagueSettings, LeagueState, Matchup, Team
from ffmcp.domain.optimizer import optimize
from ffmcp.domain.simulate import simulate_rest_of_season

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ANONYMOUS_LEAGUE_ID = 1234567
SEASON = 2026
CURRENT_WEEK = 8
REG_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 4
SEED = 20260913

TEAM_NAMES = (
    "Team Alpha",
    "Team Bravo",
    "Team Charlie",
    "Team Delta",
    "Team Echo",
    "Team Foxtrot",
    "Team Golf",
    "Team Hotel",
    "Team India",
    "Team Juliet",
)

SLOT_COUNTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "RB/WR/TE": 1,
    "D/ST": 1,
    "K": 1,
    "BE": 6,
    "IR": 1,
}

# fmt: off
PRO_TEAMS = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WSH",
)
# fmt: on

BYE_TEAMS = frozenset({"CLE", "LV", "SEA", "TB"})
"""NFL teams on bye in the current week. Their players carry no projection, which is how a bye
reaches the optimizer: it has no calendar, only a missing projection (domain/optimizer.py)."""

_PLAYING_TEAMS = tuple(t for t in PRO_TEAMS if t not in BYE_TEAMS)

ELIGIBLE_SLOTS = {
    "QB": ("QB",),
    "RB": ("RB", "RB/WR/TE"),
    "WR": ("WR", "RB/WR/TE"),
    "TE": ("TE", "RB/WR/TE"),
    "K": ("K",),
    "D/ST": ("D/ST",),
}

DEPTH_CHART: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("QB", (21.0, 13.5)),
    ("RB", (16.0, 13.0, 10.5, 8.0, 5.5)),
    ("WR", (17.0, 14.0, 11.5, 9.0, 6.5)),
    ("TE", (11.0, 6.0)),
    ("K", (8.5,)),
    ("D/ST", (7.5,)),
)
"""Per-team depth at each position: the nth-best player's baseline projection. Scaled by a
team-quality factor and jittered, this produces rosters that look like a real league's: a
handful of genuine starters and a long tail of interchangeable depth."""

# fmt: off
FIRST_NAMES = (
    "Marcus", "Devin", "Elijah", "Tobias", "Rashad", "Cormac", "Jalen", "Xavier",
    "Silas", "Damon", "Ronan", "Isaiah", "Beckett", "Malik", "Ezra", "Quentin",
    "Dominic", "Kieran", "Amari", "Theo", "Caleb", "Nico", "Jonah", "Roman",
    "Emmett", "Tariq", "Griffin", "Soren", "Luca", "Idris",
)
# fmt: on
# fmt: off
LAST_NAMES = (
    "Okafor", "Whitlock", "Brennan", "Vance", "Ferraro", "Delgado", "Ashworth",
    "Kimura", "Rowland", "Santoro", "Abernathy", "Nwosu", "Castellan", "Mbeki",
    "Holloway", "Prescott", "Ivanov", "Larkin", "Duval", "Sarsgaard", "Quintero",
    "Fairbanks", "Adeyemi", "Novak", "Ellington", "Rios", "Thackeray", "Osei",
    "Lindqvist", "Battaglia", "Moreau", "Vasquez", "Hargrove", "Sinclair",
    "Petrov", "Caldwell", "Nakamura", "Obi", "Strand", "Villanueva",
)
# fmt: on


class NameBank:
    """Unique fictional player names, drawn deterministically."""

    def __init__(self, rng: np.random.Generator) -> None:
        pairs = [f"{first} {last}" for last in LAST_NAMES for first in FIRST_NAMES]
        rng.shuffle(pairs)
        self._pairs = iter(pairs)

    def take(self) -> str:
        return next(self._pairs)


def _make_player(
    player_id: int,
    name: str,
    position: str,
    pro_team: str,
    points: float,
    *,
    injury_status: str | None = None,
) -> dict[str, Any]:
    """One player in fixture shape. A player on a bye team gets no projection at all, and a
    ruled-out player keeps his: being out is a different fact from having no game."""
    on_bye = pro_team in BYE_TEAMS
    projection = None if on_bye else {"week": CURRENT_WEEK, "points": round(points, 1)}
    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "eligible_slots": list(ELIGIBLE_SLOTS[position]),
        "pro_team": pro_team,
        "injured": injury_status is not None,
        "injury_status": injury_status,
        "projection": None if projection is None else {**projection, "source": "espn"},
        "market": None,
    }


def _defense_name(pro_team: str) -> str:
    return f"{pro_team} D/ST"


def _points(player: dict[str, Any]) -> float:
    """Projected points, or ``-1`` for a player with no game: low enough to sort last
    everywhere this module ranks a roster."""
    projection = player["projection"]
    return float(projection["points"]) if projection is not None else -1.0


def _set_points(player: dict[str, Any], points: float) -> None:
    """Force a projection. Used only by the hand-placed situations, whose whole point is that
    the named players are startable.

    Moves the player off a bye team if he was on one, because a projection for a player whose
    NFL team is not playing would be the one internally inconsistent fact in the fixture.
    """
    if player["pro_team"] in BYE_TEAMS:
        player["pro_team"] = _PLAYING_TEAMS[player["player_id"] % len(_PLAYING_TEAMS)]
    player["projection"] = {"week": CURRENT_WEEK, "points": points, "source": "espn"}


def _set_on_bye(player: dict[str, Any], pro_team: str) -> None:
    """Move a player onto a team that is on bye this week, and drop his projection with him."""
    assert pro_team in BYE_TEAMS
    player["pro_team"] = pro_team
    player["projection"] = None
    if player["position"] == "D/ST":
        player["name"] = _defense_name(pro_team)


def _build_roster_pool(
    team_index: int, quality: float, names: NameBank, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """The 16 players a team rosters, before any of them are assigned to a lineup slot."""
    pool: list[dict[str, Any]] = []
    next_id = 2000 + team_index * 100
    # Each team draws from its own shuffled slice of the NFL, so no two fixture teams roster
    # the same defense and the bye teams land on a realistic handful of rosters.
    pro_teams = list(PRO_TEAMS)
    rng.shuffle(pro_teams)

    for position, ladder in DEPTH_CHART:
        for baseline in ladder:
            pro_team = pro_teams[len(pool) % len(pro_teams)]
            points = baseline * quality + float(rng.normal(0.0, 0.8))
            name = _defense_name(pro_team) if position == "D/ST" else names.take()
            pool.append(_make_player(next_id, name, position, pro_team, max(0.5, points)))
            next_id += 1
    return pool


def _apply_team_shape(index: int, pool: list[dict[str, Any]]) -> None:
    """The three hand-placed situations (module docstring), plus a plausible injury everywhere.

    Everything else about a roster is generated. These are the facts the demo is built to
    show, and placing them here is honest in a way inventing a transcript would not be.
    """
    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in pool:
        by_position.setdefault(player["position"], []).append(player)
    for players in by_position.values():
        players.sort(key=_points, reverse=True)

    if index == 0:  # Team Alpha: two startable QBs, a thin and banged-up backfield.
        _set_points(by_position["QB"][0], 22.6)
        _set_points(by_position["QB"][1], 19.4)
        for rank, points in enumerate((11.8, 9.6, 8.4, 6.9, 5.2)):
            _set_points(by_position["RB"][rank], points)
        by_position["RB"][0]["injury_status"] = "OUT"
        by_position["RB"][0]["injured"] = True
        # The second receiver is on bye, and still in last week's lineup, below.
        _set_on_bye(by_position["WR"][1], "SEA")
        # Monday-morning recap material: the started WR1 goes cold live, and a bench receiver
        # who barely projected explodes, so the actual-best lineup swaps them, a move neither
        # the stale lineup above nor Thursday's pregame projection would suggest.
        by_position["WR"][0]["live_points"] = round(_points(by_position["WR"][0]) * 0.15, 1)
        by_position["WR"][3]["live_points"] = round(_points(by_position["WR"][3]) * 2.8, 1)
    elif index == 1:  # Team Bravo: a deep backfield and nothing at quarterback.
        _set_points(by_position["QB"][0], 12.9)
        _set_points(by_position["QB"][1], 8.1)
        for rank, points in enumerate((18.2, 16.4, 15.1, 7.8, 5.4)):
            _set_points(by_position["RB"][rank], points)

    # One player per team is on injured reserve. A real roster almost always has one, and it
    # keeps the IR slot in the rendered output from being dead weight. Never a tight end: with
    # only two rostered, losing one would leave a flex nothing to hold.
    ir_candidate = by_position["WR"][-1] if index % 2 else by_position["RB"][-1]
    ir_candidate["injury_status"] = "INJURY_RESERVE"
    ir_candidate["injured"] = True


def _assign_lineup(
    pool: list[dict[str, Any]], *, stale: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a roster pool into (starting slots, bench, IR).

    With ``stale=False`` the manager has set an optimal lineup. With ``stale=True`` they have
    not touched it since last week: a ruled-out back and a receiver now on bye are still in,
    and the flex holds a tight end while a better back sits, which is exactly the state
    ``optimize_lineup`` exists to fix.
    """
    ir_ids = {p["player_id"] for p in pool if p["injury_status"] == "INJURY_RESERVE"}
    ir = [p for p in pool if p["player_id"] in ir_ids]
    available = [p for p in pool if p["player_id"] not in ir_ids]

    used: set[int] = set()
    slots: list[dict[str, Any]] = []

    def candidates(*positions: str) -> list[dict[str, Any]]:
        return [p for p in available if p["position"] in positions and p["player_id"] not in used]

    def place(slot: str, player: dict[str, Any] | None) -> None:
        if player is not None:
            used.add(player["player_id"])
        slots.append({"slot": slot, "player": player})

    def best(*positions: str) -> dict[str, Any] | None:
        pool_ = candidates(*positions)
        return max(pool_, key=_points) if pool_ else None

    if stale:
        place("QB", best("QB"))
        out_rb = next((p for p in candidates("RB") if p["injury_status"] == "OUT"), None)
        place("RB", out_rb or best("RB"))
        place("RB", best("RB"))
        bye_wr = next((p for p in candidates("WR") if p["projection"] is None), None)
        place("WR", bye_wr or best("WR"))
        place("WR", best("WR"))
        place("TE", best("TE"))
        place("RB/WR/TE", best("TE"))  # last week's flex, never revisited
    else:
        place("QB", best("QB"))
        place("RB", best("RB"))
        place("RB", best("RB"))
        place("WR", best("WR"))
        place("WR", best("WR"))
        place("TE", best("TE"))
        place("RB/WR/TE", best("RB", "WR", "TE"))
    place("D/ST", best("D/ST"))
    place("K", best("K"))

    bench = [p for p in available if p["player_id"] not in used]
    return slots, bench, ir


def _round_robin(team_ids: list[int], weeks: int) -> dict[int, list[tuple[int, int]]]:
    """Circle-method schedule: one team fixed, the rest rotated. With an even number of teams
    this yields ``n - 1`` distinct rounds, which are then repeated to fill the season."""
    rotation = team_ids[1:]
    rounds: list[list[tuple[int, int]]] = []
    for _ in range(len(team_ids) - 1):
        pairs = [(team_ids[0], rotation[0])]
        pairs += [(rotation[i], rotation[-i]) for i in range(1, len(team_ids) // 2)]
        rounds.append(pairs)
        rotation = rotation[1:] + rotation[:1]
    return {week: rounds[(week - 1) % len(rounds)] for week in range(1, weeks + 1)}


def _settings() -> LeagueSettings:
    return LeagueSettings(
        league_id=ANONYMOUS_LEAGUE_ID,
        season=SEASON,
        team_count=len(TEAM_NAMES),
        playoff_team_count=PLAYOFF_TEAMS,
        reg_season_weeks=REG_SEASON_WEEKS,
        slot_counts=SLOT_COUNTS,
        scoring_type="PPR",
    )


def _optimal_points(team: dict[str, Any]) -> float:
    """What this team scores if it starts its best legal lineup: the mean of its weekly
    score. Uses the repo's own optimizer, so the fixture and the server agree on strength."""
    parsed = Team.model_validate(team)
    return optimize(parsed.roster.players, _settings().starting_slots).projected_points


def _record_result(
    home: dict[str, Any], away: dict[str, Any], home_score: float, away_score: float
) -> None:
    home["points_for"] += home_score
    home["points_against"] += away_score
    away["points_for"] += away_score
    away["points_against"] += home_score
    if home_score > away_score:
        home["wins"] += 1
        away["losses"] += 1
    elif away_score > home_score:
        away["wins"] += 1
        home["losses"] += 1
    else:
        home["ties"] += 1
        away["ties"] += 1


def _play_season(
    teams: list[dict[str, Any]],
    schedule: dict[int, list[tuple[int, int]]],
    strengths: dict[int, float],
    rng: np.random.Generator,
) -> dict[str, list[dict[str, Any]]]:
    """Play weeks 1..current-1 and record the results. Leave the rest scheduled but unplayed.

    Records, points for and points against all fall out of these games, so nothing about the
    standings has to be asserted separately, and nothing can contradict anything else.
    """
    by_id = {t["team_id"]: t for t in teams}
    matchups: dict[str, list[dict[str, Any]]] = {}

    for week in range(1, REG_SEASON_WEEKS + 1):
        week_matchups: list[dict[str, Any]] = []
        for home_id, away_id in schedule[week]:
            home, away = by_id[home_id], by_id[away_id]
            if week < CURRENT_WEEK:
                # A manager does not start a perfect lineup every week, hence the 0.94.
                home_score = round(float(rng.normal(strengths[home_id] * 0.94, 19.0)), 1)
                away_score = round(float(rng.normal(strengths[away_id] * 0.94, 19.0)), 1)
                _record_result(home, away, home_score, away_score)
            else:
                home_score = away_score = 0.0
            week_matchups.append(
                {
                    "week": week,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_projected": round(strengths[home_id], 1),
                    "away_projected": round(strengths[away_id], 1),
                    "is_playoff": False,
                }
            )
        matchups[str(week)] = week_matchups

    for team in teams:
        team["points_for"] = round(team["points_for"], 1)
        team["points_against"] = round(team["points_against"], 1)
    for standing, team in enumerate(
        sorted(teams, key=lambda t: (-t["wins"], -t["points_for"])), start=1
    ):
        team["standing"] = standing
    return matchups


def _free_agents(names: NameBank, rng: np.random.Generator) -> list[dict[str, Any]]:
    """The waiver wire: mostly replacement-level, with a few genuinely startable names so
    ``find_waiver_targets`` has something to rank and value over replacement is not zero."""
    # fmt: off
    plan: tuple[tuple[str, float], ...] = (
        ("RB", 12.4), ("RB", 7.1), ("RB", 5.8), ("RB", 4.2),
        ("WR", 13.1), ("WR", 10.6), ("WR", 6.4), ("WR", 5.1), ("WR", 3.9),
        ("TE", 9.3), ("TE", 4.6), ("TE", 3.1),
        ("QB", 14.2), ("QB", 8.7),
        ("K", 7.9), ("K", 6.2),
        ("D/ST", 8.8), ("D/ST", 5.5),
    )
    # fmt: on
    # Free agents are never on bye here: a waiver target you cannot start this week is a
    # different (and much less interesting) recommendation than the one this tool makes.
    pro_teams = [t for t in PRO_TEAMS if t not in BYE_TEAMS]
    rng.shuffle(pro_teams)
    agents: list[dict[str, Any]] = []
    for index, (position, points) in enumerate(plan):
        pro_team = pro_teams[index % len(pro_teams)]
        name = _defense_name(pro_team) if position == "D/ST" else names.take()
        agents.append(_make_player(9000 + index, name, position, pro_team, points))
    return agents


def build_league(rng: np.random.Generator) -> dict[str, Any]:
    names = NameBank(rng)
    qualities = np.linspace(1.12, 0.88, len(TEAM_NAMES))
    rng.shuffle(qualities)

    teams: list[dict[str, Any]] = []
    for index, name in enumerate(TEAM_NAMES):
        pool = _build_roster_pool(index, float(qualities[index]), names, rng)
        _apply_team_shape(index, pool)
        slots, bench, ir = _assign_lineup(pool, stale=index == 0)
        teams.append(
            {
                "team_id": index + 1,
                "name": name,
                "abbrev": name.split()[1][:3].upper(),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "points_for": 0.0,
                "points_against": 0.0,
                "standing": index + 1,
                "playoff_pct": 0.0,
                "roster": {"slots": slots, "bench": bench, "ir": ir},
            }
        )

    schedule = _round_robin([t["team_id"] for t in teams], REG_SEASON_WEEKS)
    strengths = {t["team_id"]: _optimal_points(t) for t in teams}
    matchups = _play_season(teams, schedule, strengths, rng)

    return {
        "settings": _settings().model_dump(mode="json"),
        "current_week": CURRENT_WEEK,
        "teams": teams,
        "matchups": matchups,
        "free_agents": {str(CURRENT_WEEK): _free_agents(names, rng)},
    }


def build_market(league: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sleeper-shaped market signal for the fixture, keyed by ESPN player id.

    Ownership tracks projection: the players everyone starts are the players everyone owns.
    The trending counts are concentrated on the top free agents, which is what the real
    signal looks like on a Tuesday.
    """
    rostered = [
        slot["player"]
        for team in league["teams"]
        for slot in team["roster"]["slots"]
        if slot["player"] is not None
    ]
    rostered += [p for team in league["teams"] for p in team["roster"]["bench"]]

    by_espn_id: dict[str, Any] = {}
    for player in rostered:
        owned = min(99.8, 45.0 + max(0.0, _points(player)) * 3.2)
        by_espn_id[str(player["player_id"])] = {
            "percent_owned": round(owned, 1),
            "percent_started": round(max(1.0, owned - 12.0 - float(rng.uniform(0, 8))), 1),
            "trending_adds": None,
        }

    trending: dict[str, int] = {}
    free_agents = league["free_agents"][str(CURRENT_WEEK)]
    for rank, player in enumerate(sorted(free_agents, key=_points, reverse=True)):
        adds = int(28_000 * 0.55**rank)
        by_espn_id[str(player["player_id"])] = {
            "percent_owned": round(max(0.4, 38.0 - rank * 4.0), 1),
            "percent_started": round(max(0.1, 14.0 - rank * 2.0), 1),
            "trending_adds": adds,
        }
        if adds > 0:
            trending[str(player["player_id"])] = adds

    return {
        "current_season": SEASON,
        "current_week": CURRENT_WEEK,
        "trending_adds": dict(sorted(trending.items(), key=lambda kv: -kv[1])[:25]),
        "market_by_espn_id": by_espn_id,
    }


def _fill_playoff_pct(league: dict[str, Any]) -> None:
    """Replace each team's stored ``playoff_pct`` with this repo's own simulation of the league
    we just built, so the fixture's odds and ``simulate_season``'s output cannot disagree."""
    state = LeagueState(
        settings=LeagueSettings.model_validate(league["settings"]),
        teams=tuple(Team.model_validate(t) for t in league["teams"]),
        current_week=league["current_week"],
        matchups=tuple(
            Matchup.model_validate(matchup)
            for week_matchups in league["matchups"].values()
            for matchup in week_matchups
        ),
    )
    outcome = simulate_rest_of_season(state, n_sims=20_000, rng=np.random.default_rng(SEED))
    for team in league["teams"]:
        team["playoff_pct"] = round(outcome.team(team["team_id"]).playoff_odds, 1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    league = build_league(rng)
    _fill_playoff_pct(league)
    market = build_market(league, rng)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "demo_league.json").write_text(
        json.dumps(league, indent=2) + "\n", encoding="utf-8"
    )
    (FIXTURES_DIR / "demo_market.json").write_text(
        json.dumps(market, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURES_DIR / 'demo_league.json'}")
    print(f"wrote {FIXTURES_DIR / 'demo_market.json'}")


if __name__ == "__main__":
    main()

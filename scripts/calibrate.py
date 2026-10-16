"""Re-derive every fitted constant in ``domain/`` from real data, and the evidence behind it.

    uv run python scripts/calibrate.py [--season 2025] [--xfp-train 2024] [--backtest 2024 2025]

Needs a live league (``.env``, as for ``FFMCP_MODE=live``) for the projection and variance
sections, because the evidence is the league's own box scores: every rostered player's ESPN
projection beside what he scored. The expected-points section needs only nflverse's public
files. Nothing is written anywhere: the output is a report, and the constants it prints are
copied into ``domain/variance.py``, ``domain/usage.py`` and ``domain/league_intel.py`` by hand,
deliberately, so a refit is a reviewed change and not a silent drift.

Sections, each answering one design question with held-out data:

1. **Position spread.** Weekly sd of (actual - projected) as a fraction of the projection.
2. **Team spread.** Does a lineup's sd equal its players' sds in quadrature, or does it need a
   correlation factor? (It needed 0.92, not the 1.6 once assumed.)
3. **Per-player bias correction.** Does correcting a player's projection by his own recent
   misses help? (No, at any shrinkage: so the server does not do it.)
4. **Per-player spread.** Does his own history predict his spread? (No better than his
   position does, at any shrinkage: so the server uses the position's.)
5. **Second source.** Is Sleeper more accurate than ESPN, does averaging help, and does
   disagreement between them predict error? (No, barely, and yes.)
6. **Lineup accuracy.** Is it a trait worth labelling managers by? (Yes; hindsight is not.)
7. **Expected points.** Fit on one season, tested on the next: does volume predict future
   scoring better than points do, and does luck regress?
8. **Team strength.** How far does a team's actual rest-of-season scoring stray from its
   projected optimal lineup, beyond what weekly noise explains? That spread is the per-season
   team-strength draw in ``domain/simulate.py`` (``TEAM_STRENGTH_SD``).
9. **Playoff odds backtest.** Simulate each past season from several mid-season weeks, with the
   model as shipped and with the one it replaced (this week's lineup held constant all season,
   no strength term), and score both against who actually made the playoffs (Brier score).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import httpx2
import numpy as np
from espn_api.football import League

import ffmcp.domain.simulate as simulate
from ffmcp.config import load_settings
from ffmcp.domain.models import (
    LeagueSettings,
    LeagueState,
    Matchup,
    Player,
    PlayerWeek,
    Projection,
    Roster,
    Team,
)
from ffmcp.domain.optimizer import optimize
from ffmcp.domain.schedule import week_lineup
from ffmcp.providers.identity import IdentityIndex

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
SLEEPER = "https://api.sleeper.app"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")
XFP_FEATURES = {
    "QB": ("attempts", "passing_air_yards", "carries"),
    "RB": ("carries", "targets", "receiving_air_yards"),
    "WR": ("targets", "receiving_air_yards", "carries"),
    "TE": ("targets", "receiving_air_yards"),
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class Season:
    """One completed league season, as box scores: every rostered player-week, every game, each
    pro team's bye, and who made the playoffs."""

    def __init__(self, season: int) -> None:
        settings = load_settings()
        league = League(
            league_id=settings.league_id,
            year=season,
            espn_s2=settings.espn_s2.get_secret_value() if settings.espn_s2 else None,
            swid=settings.swid.get_secret_value() if settings.swid else None,
        )
        self.season = season
        self.settings = LeagueSettings(
            league_id=0,
            season=season,
            team_count=len(league.teams),
            playoff_team_count=int(league.settings.playoff_team_count),
            reg_season_weeks=int(league.settings.reg_season_count),
            slot_counts=dict(league.settings.position_slot_counts),
        )
        self.slots = self.settings.starting_slots
        self.team_ids = tuple(sorted(int(t.team_id) for t in league.teams))
        self.made_playoffs = {
            int(t.team_id): int(t.standing) <= self.settings.playoff_team_count
            for t in league.teams
        }
        self.rows: list[PlayerWeek] = []
        self.games: list[tuple[int, int, int | None, float, float]] = []
        self.byes: dict[str, int] = {}
        for week in range(1, self.settings.reg_season_weeks + 1):
            for box in league.box_scores(week):
                home, away = _team_id(box.home_team), _team_id(box.away_team)
                if home is not None:
                    self.games.append(
                        (week, home, away, float(box.home_score), float(box.away_score))
                    )
                for side in ("home", "away"):
                    for p in getattr(box, f"{side}_lineup"):
                        if getattr(p, "on_bye_week", False):
                            self.byes[str(p.proTeam)] = week
            self.rows += _week_rows(league, season, week)
            print(f"  {season} week {week}: {len(self.rows)} player-weeks", file=sys.stderr)


def _team_id(team: Any) -> int | None:
    if team is None:
        return None
    return team if isinstance(team, int) else int(team.team_id)


def _week_rows(league: League, season: int, week: int) -> list[PlayerWeek]:
    rows: list[PlayerWeek] = []
    for box in league.box_scores(week):
        for side in ("home", "away"):
            team_id = _team_id(getattr(box, f"{side}_team"))
            for p in getattr(box, f"{side}_lineup"):
                rows.append(
                    PlayerWeek(
                        season=season,
                        week=week,
                        player_id=int(p.playerId),
                        name=str(p.name),
                        position=str(p.position),
                        pro_team=str(p.proTeam),
                        eligible_slots=tuple(str(s) for s in p.eligibleSlots),
                        fantasy_team_id=team_id,
                        slot=str(p.slot_position),
                        projected=p.projected_points,
                        actual=float(p.points or 0.0),
                        played=bool(p.game_played) and not bool(p.on_bye_week),
                    )
                )
    return rows


def league_history(season: int) -> tuple[list[PlayerWeek], tuple[str, ...]]:
    settings = load_settings()
    league = League(
        league_id=settings.league_id,
        year=season,
        espn_s2=settings.espn_s2.get_secret_value() if settings.espn_s2 else None,
        swid=settings.swid.get_secret_value() if settings.swid else None,
    )
    slots = tuple(
        slot
        for slot, count in league.settings.position_slot_counts.items()
        if slot not in ("BE", "IR") and count > 0
        for _ in range(count)
    )
    last = int(league.current_week) - 1
    rows: list[PlayerWeek] = []
    for week in range(1, last + 1):
        for box in league.box_scores(week):
            for side in ("home", "away"):
                team = getattr(box, f"{side}_team")
                team_id = team if isinstance(team, int) else getattr(team, "team_id", None)
                for p in getattr(box, f"{side}_lineup"):
                    rows.append(
                        PlayerWeek(
                            season=season,
                            week=week,
                            player_id=int(p.playerId),
                            name=str(p.name),
                            position=str(p.position),
                            pro_team=str(p.proTeam),
                            eligible_slots=tuple(str(s) for s in p.eligibleSlots),
                            fantasy_team_id=team_id,
                            slot=str(p.slot_position),
                            projected=p.projected_points,
                            actual=float(p.points or 0.0),
                            played=bool(p.game_played) and not bool(p.on_bye_week),
                        )
                    )
        print(f"  {season} week {week}: {len(rows)} player-weeks", file=sys.stderr)
    return rows, slots


def sleeper_projections(
    client: httpx2.Client, season: int, weeks: Sequence[int]
) -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    for week in weeks:
        rows = client.get(f"{SLEEPER}/projections/nfl/{season}/{week}?season_type=regular").json()
        for row in rows:
            points = (row.get("stats") or {}).get("pts_ppr")
            if points is not None:
                result[(week, str(row["player_id"]))] = float(points)
    return result


def sleeper_identity(client: httpx2.Client) -> tuple[IdentityIndex, dict[int, str]]:
    raw = client.get(f"{SLEEPER}/v1/players/nfl").json()
    index = {
        sid: {
            "name": info.get("full_name")
            or f"{info.get('first_name', '')} {info.get('last_name', '')}".strip(),
            "pos": info.get("position") or "",
            "team": info.get("team") or "",
        }
        for sid, info in raw.items()
        if info.get("position")
    }
    by_espn = {int(info["espn_id"]): sid for sid, info in raw.items() if info.get("espn_id")}
    return IdentityIndex.build(index), by_espn


def nflverse_weekly(client: httpx2.Client, season: int) -> list[dict[str, str]]:
    url = f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv"
    text = client.get(url, follow_redirects=True).text
    return [
        row
        for row in csv.DictReader(io.StringIO(text))
        if row["season_type"] == "REG" and row["position"] in XFP_FEATURES
    ]


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "NA") else 0.0


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def scored(rows: Sequence[PlayerWeek]) -> list[PlayerWeek]:
    return [r for r in rows if r.played and r.projected is not None and r.projected > 0.0]


def position_spread(rows: Sequence[PlayerWeek]) -> dict[str, float]:
    print("\n1. Position spread: sd(actual - projected) / mean(projected)")
    fractions = {}
    for position in POSITIONS:
        sub = [r for r in rows if r.position == position]
        if len(sub) < 30:
            continue
        residuals = np.array([r.actual - (r.projected or 0.0) for r in sub])
        fractions[position] = float(np.std(residuals) / np.mean([r.projected for r in sub]))
        print(f"   {position:5} n={len(sub):5}  fraction {fractions[position]:.2f}")
    return fractions


def team_spread(all_rows: Sequence[PlayerWeek], fractions: dict[str, float]) -> None:
    print("\n2. Team spread: real lineup sd vs players' sds in quadrature")
    teams: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in all_rows:
        if not r.started or r.projected is None or r.fantasy_team_id is None:
            continue
        t = teams[(r.fantasy_team_id, r.week)]
        t[0] += r.projected
        t[1] += r.actual
        t[2] += max(fractions.get(r.position, 0.6) * r.projected, 2.0) ** 2
    values = np.array(list(teams.values()))
    actual_sd = float(np.std(values[:, 1] - values[:, 0]))
    quadrature = float(np.mean(np.sqrt(values[:, 2])))
    print(f"   team-weeks {len(values)}: residual sd {actual_sd:.1f}, quadrature {quadrature:.1f}")
    print(f"   -> TEAM_SD_FACTOR = {actual_sd / quadrature:.2f}")


def _halves(
    rows: Sequence[PlayerWeek], split: int
) -> list[tuple[str, list[PlayerWeek], list[PlayerWeek]]]:
    by: dict[int, list[PlayerWeek]] = defaultdict(list)
    for r in rows:
        by[r.player_id].append(r)
    out = []
    for games in by.values():
        first = [g for g in games if g.week <= split]
        second = [g for g in games if g.week > split]
        if len(first) >= 3 and len(second) >= 3:
            out.append((games[0].position, first, second))
    return out


def bias_correction(rows: Sequence[PlayerWeek]) -> None:
    print("\n3. Per-player bias correction: second-half MSE after shifting by first-half misses")
    pairs = _halves(rows, 8)
    for k in (0.0, 4.0, 8.0, 16.0, 32.0, math.inf):
        errors = []
        for _, first, second in pairs:
            bias = float(np.mean([g.actual - (g.projected or 0.0) for g in first]))
            shrink = 0.0 if math.isinf(k) else len(first) / (len(first) + k)
            errors += [(g.actual - (g.projected or 0.0) - bias * shrink) ** 2 for g in second]
        label = "none" if math.isinf(k) else f"k={k:g}"
        print(f"   {label:6} MSE {np.mean(errors):.2f}")
    print("   (lowest MSE with no correction means: do not correct)")


def personal_spread(rows: Sequence[PlayerWeek], fractions: dict[str, float]) -> None:
    print("\n4. Per-player spread: held-out log-likelihood of second-half misses")
    pairs = _halves(rows, 8)
    for k in (0.0, 4.0, 8.0, 16.0, 32.0, math.inf):
        loglik = []
        for position, first, second in pairs:
            n = len(first)
            mean_projection = float(np.mean([g.projected for g in first]))
            own = math.sqrt(np.mean([(g.actual - (g.projected or 0.0)) ** 2 for g in first])) / max(
                mean_projection, 1.0
            )
            prior = fractions.get(position, 0.6)
            fraction = prior if math.isinf(k) else math.sqrt((n * own**2 + k * prior**2) / (n + k))
            for g in second:
                sd = max(fraction * (g.projected or 0.0), 2.0)
                miss = g.actual - (g.projected or 0.0)
                loglik.append(-math.log(sd) - miss * miss / (2 * sd * sd))
        label = "position only" if math.isinf(k) else f"k={k:g}"
        print(f"   {label:14} {np.mean(loglik):.4f}")


def second_source(rows: Sequence[PlayerWeek], fractions: dict[str, float], season: int) -> None:
    print("\n5. Second source (Sleeper) against ESPN")
    with httpx2.Client(timeout=60) as client:
        identity, by_espn = sleeper_identity(client)
        sleeper = sleeper_projections(client, season, sorted({r.week for r in rows}))
    matched = []
    for r in rows:
        sid = by_espn.get(r.player_id) or identity.resolve(
            r.player_id, r.name, r.position, r.pro_team
        )
        other = sleeper.get((r.week, sid)) if sid else None
        if other is not None:
            matched.append((r, other))
    espn = np.array([abs(r.actual - (r.projected or 0.0)) for r, _ in matched])
    slp = np.array([abs(r.actual - s) for r, s in matched])
    avg = np.array([abs(r.actual - ((r.projected or 0.0) + s) / 2) for r, s in matched])
    print(
        f"   matched {len(matched)} player-weeks: MAE ESPN {espn.mean():.2f}, "
        f"Sleeper {slp.mean():.2f}, average {avg.mean():.2f}"
    )
    gaps = np.array(
        [abs((r.projected or 0.0) - s) / max(r.projected or 0.0, 1.0) for r, s in matched]
    )
    rel = np.array(
        [abs(r.actual - (r.projected or 0.0)) / max(r.projected or 0.0, 1.0) for r, _ in matched]
    )
    edges = np.quantile(gaps, [0.0, 0.5, 0.8, 0.95, 1.0])
    for lo, hi in pairwise(edges):
        band = (gaps >= lo) & (gaps <= hi)
        miss = rel[band].mean()
        print(f"   disagreement {lo:.2f}-{hi:.2f}: relative miss {miss:.2f}  (n={band.sum()})")
    print(f"   -> TYPICAL_DISAGREEMENT = {np.median(gaps):.2f}")
    for c in (0.0, 0.5, 1.0, 2.0):
        loglik = []
        for (r, _), gap in zip(matched, gaps, strict=True):
            scale = (1 + c * min(gap, 1.0)) / (1 + c * float(np.median(gaps)))
            sd = max(fractions.get(r.position, 0.6) * (r.projected or 0.0) * scale, 2.0)
            miss = r.actual - (r.projected or 0.0)
            loglik.append(-math.log(sd) - miss * miss / (2 * sd * sd))
        print(f"   DISAGREEMENT_SENSITIVITY={c:g}: log-likelihood {np.mean(loglik):.4f}")


def lineup_accuracy(all_rows: Sequence[PlayerWeek], slots: tuple[str, ...]) -> None:
    print("\n6. Lineup accuracy (decisions) vs hindsight efficiency (decisions + luck), by team")
    by: dict[tuple[int, int], list[PlayerWeek]] = defaultdict(list)
    for r in all_rows:
        if r.fantasy_team_id is not None and r.slot != "IR":
            by[(r.fantasy_team_id, r.week)].append(r)
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for (team, _), rs in by.items():
        t = totals[team]
        t[0] += sum(r.projected or 0.0 for r in rs if r.started)
        t[1] += optimize([r.as_player(hindsight=False) for r in rs], slots).projected_points
        t[2] += sum(r.actual for r in rs if r.started)
        t[3] += optimize([r.as_player() for r in rs], slots).projected_points
    for team, (sp, bp, sa, ba) in sorted(totals.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"   team {team:3}: accuracy {100 * sp / bp:5.1f}%  hindsight {100 * sa / ba:5.1f}%")


def expected_points(train: int, test: int) -> None:
    print(f"\n7. Expected points: fit on {train}, tested on {test}")
    with httpx2.Client(timeout=120) as client:
        train_rows = nflverse_weekly(client, train)
        test_rows = nflverse_weekly(client, test)
    coefficients: dict[tuple[str, str], Any] = {}
    for position, features in XFP_FEATURES.items():
        sub = [r for r in train_rows if r["position"] == position]
        x = np.array([[_f(r, k) for k in features] + [1.0] for r in sub])
        for target in ("fantasy_points", "fantasy_points_ppr"):
            y = np.array([_f(r, target) for r in sub])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            coefficients[(position, target)] = coef
            r2 = 1 - np.var(y - x @ coef) / np.var(y)
            print(
                f"   {position} {target:19} R^2 {r2:.2f}  coefficients {np.round(coef, 4).tolist()}"
            )

    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in test_rows:
        by[r["player_id"]].append(r)
    for split in (4, 8):
        past_act, past_x, future = [], [], []
        for games in by.values():
            first = [g for g in games if int(g["week"]) <= split]
            second = [g for g in games if split < int(g["week"]) <= split + 6]
            if len(first) < 3 or len(second) < 3:
                continue
            position = first[0]["position"]
            coef = coefficients[(position, "fantasy_points_ppr")]
            xfp = float(
                np.mean(
                    [
                        np.dot([_f(g, k) for k in XFP_FEATURES[position]] + [1.0], coef)
                        for g in first
                    ]
                )
            )
            act = float(np.mean([_f(g, "fantasy_points_ppr") for g in first]))
            if act < 5 and xfp < 5:
                continue
            past_act.append(act)
            past_x.append(xfp)
            future.append(float(np.mean([_f(g, "fantasy_points_ppr") for g in second])))
        a, x_, y = np.array(past_act), np.array(past_x), np.array(future)
        print(
            f"   after week {split} (n={len(y)}): next-6-week MAE from points/g "
            f"{np.abs(a - y).mean():.2f}, from xFP/g {np.abs(x_ - y).mean():.2f}"
        )
        luck = a - x_
        for lo, hi, label in ((-99, -3, "unlucky by 3+"), (3, 99, "lucky by 3+")):
            band = (luck >= lo) & (luck < hi)
            if band.any():
                change = np.mean(y[band] - a[band])
                print(f"     {label}: n={band.sum()}, future minus past points/g {change:+.2f}")


def rosters_as_of(season: Season, week: int) -> dict[int, list[Player]]:
    """Each team's roster in ``week`` (IR excluded), each player carrying that week's projection,
    his bye, and a season rate built only from what was known by then: the mean of his own
    projections in the weeks up to and including ``week``."""
    known: dict[int, list[float]] = defaultdict(list)
    for r in season.rows:
        if r.week <= week and r.projected and r.projected > 0.0:
            known[r.player_id].append(r.projected)
    rosters: dict[int, list[Player]] = defaultdict(list)
    for r in season.rows:
        if r.week != week or r.fantasy_team_id is None or r.slot == "IR":
            continue
        rates = known.get(r.player_id)
        rosters[r.fantasy_team_id].append(
            Player(
                player_id=r.player_id,
                name=r.name,
                position=r.position,
                eligible_slots=r.eligible_slots,
                pro_team=r.pro_team,
                projection=Projection(week=week, points=r.projected or 0.0),
                bye_week=season.byes.get(r.pro_team),
                season_rate=float(np.mean(rates)) if rates else None,
            )
        )
    return rosters


def team_strength(seasons: Sequence[Season]) -> float:
    print("\n8. Team strength: rest-of-season actual average around the projected lineup")
    residuals, within, counts = [], [], []
    for season in seasons:
        actual: dict[tuple[int, int], float] = defaultdict(float)
        for r in season.rows:
            if r.started and r.fantasy_team_id is not None:
                actual[(r.fantasy_team_id, r.week)] += r.actual
        last = season.settings.reg_season_weeks
        for as_of in range(3, last - 3):
            rosters = rosters_as_of(season, as_of)
            for team_id, players in rosters.items():
                misses = []
                for week in range(as_of + 1, last + 1):
                    if (team_id, week) not in actual:
                        continue
                    projected = week_lineup(players, season.slots, week, as_of).projected_points
                    misses.append(actual[(team_id, week)] - projected)
                if len(misses) >= 3:
                    residuals.append(float(np.mean(misses)))
                    within.append(float(np.var(misses, ddof=1)))
                    counts.append(len(misses))
    res = np.array(residuals)
    noise = float(np.mean(np.array(within) / np.array(counts)))
    between = float(np.var(res, ddof=1))
    sd = math.sqrt(max(between - noise, 0.0))
    print(
        f"   {len(res)} team-horizons: league-wide bias {res.mean():+.1f} pts/wk (shared by all "
        "teams, so it moves nobody's odds)"
    )
    print(
        f"   variance of a team's average miss {between:.1f}, of which weekly noise explains "
        f"{noise:.1f}"
    )
    print(f"   -> TEAM_STRENGTH_SD = {sd:.1f}")
    return sd


def _state_as_of(season: Season, week: int, *, legacy: bool) -> LeagueState:
    """The league as it stood going into ``week``: records from the games before it, rosters
    and projections from that week. ``legacy`` rebuilds what the replaced model saw: every
    player's later weeks pinned to this week's projection, byes and all."""
    records: dict[int, list[float]] = {t: [0.0, 0.0, 0.0, 0.0] for t in season.team_ids}
    matchups = []
    for game_week, home, away, home_score, away_score in season.games:
        if game_week < week and away is not None:
            for team, mine, theirs in (
                (home, home_score, away_score),
                (away, away_score, home_score),
            ):
                rec = records[team]
                rec[0 if mine > theirs else 1 if mine < theirs else 2] += 1
                rec[3] += mine
        elif game_week >= week:
            matchups.append(
                Matchup(
                    week=game_week,
                    home_team_id=home,
                    away_team_id=away,
                    home_score=0.0,
                    away_score=0.0,
                    home_projected=0.0,
                    away_projected=0.0,
                )
            )
    rosters = rosters_as_of(season, week)
    teams = []
    for team_id in season.team_ids:
        players = rosters.get(team_id, [])
        if legacy:
            players = [
                p.model_copy(
                    update={
                        "season_rate": 0.0
                        if p.bye_week == week
                        else (p.projection.points if p.projection else 0.0),
                        "bye_week": None,
                    }
                )
                for p in players
            ]
        wins, losses, ties, points_for = records[team_id]
        teams.append(
            Team(
                team_id=team_id,
                name=str(team_id),
                abbrev=str(team_id),
                wins=int(wins),
                losses=int(losses),
                ties=int(ties),
                points_for=points_for,
                points_against=0.0,
                standing=0,
                playoff_pct=0.0,
                roster=Roster(slots=(), bench=tuple(players)),
            )
        )
    return LeagueState(
        settings=season.settings, teams=tuple(teams), current_week=week, matchups=tuple(matchups)
    )


def playoff_backtest(seasons: Sequence[Season], strength_sd: float) -> None:
    print("\n9. Playoff odds backtest: Brier score against who actually made it (lower is better)")
    weeks = (4, 6, 8, 10)
    models = (
        ("replaced (this week held constant)", True, 0.0),
        ("per-week means, no strength term", False, 0.0),
        (f"as shipped (strength sd {strength_sd:.1f})", False, strength_sd),
    )
    original = simulate.TEAM_STRENGTH_SD
    try:
        for label, legacy, sd in models:
            simulate.TEAM_STRENGTH_SD = sd
            scores, confident_misses = [], 0
            by_week: dict[int, list[float]] = defaultdict(list)
            for season in seasons:
                for week in weeks:
                    state = _state_as_of(season, week, legacy=legacy)
                    outcome = simulate.simulate_rest_of_season(
                        state, n_sims=5_000, rng=np.random.default_rng(week)
                    )
                    for team in outcome.teams:
                        p = team.playoff_odds / 100.0
                        made = season.made_playoffs[team.team_id]
                        scores.append((p - made) ** 2)
                        by_week[week].append((p - made) ** 2)
                        confident_misses += (p >= 0.9 and not made) or (p <= 0.1 and made)
            print(
                f"   {label:42} Brier {np.mean(scores):.4f}  "
                f"(n={len(scores)}; 90%+ calls that missed: {confident_misses})"
            )
            print(
                "     by week: "
                + ", ".join(f"{w}: {np.mean(v):.3f}" for w, v in sorted(by_week.items()))
            )
    finally:
        simulate.TEAM_STRENGTH_SD = original
    spots = np.mean([s.settings.playoff_team_count / len(s.team_ids) for s in seasons])
    print(
        f"   reference: a flat {100 * spots:.0f}% for every team scores {spots * (1 - spots):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--season", type=int, default=2025, help="completed league season to fit on"
    )
    parser.add_argument("--xfp-train", type=int, default=2024, help="nflverse season to fit xFP on")
    parser.add_argument(
        "--backtest",
        type=int,
        nargs="*",
        default=[2024, 2025],
        help="completed league seasons for sections 8 and 9",
    )
    parser.add_argument(
        "--only-season-model", action="store_true", help="run sections 8 and 9 only"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if args.only_season_model:
        seasons = [Season(year) for year in args.backtest]
        playoff_backtest(seasons, team_strength(seasons))
        return

    all_rows, slots = league_history(args.season)
    rows = scored(all_rows)
    print(f"{args.season}: {len(rows)} scored player-weeks")
    fractions = position_spread(rows)
    team_spread(all_rows, fractions)
    bias_correction(rows)
    personal_spread(rows, fractions)
    second_source(rows, fractions, args.season)
    lineup_accuracy(all_rows, slots)
    expected_points(args.xfp_train, args.xfp_train + 1)
    seasons = [Season(year) for year in args.backtest]
    playoff_backtest(seasons, team_strength(seasons))


if __name__ == "__main__":
    main()

"""Generate ``tests/fixtures/demo_insights.json``: the history, usage, second-opinion
projections and game lines that demo mode serves beside the synthetic league.

Derived from ``demo_league.json`` rather than generated alongside it, so the league fixture (and
every README transcript built from it) is untouched by adding this. Like the league, everything
here is invented: see ``make_demo_fixture.py``'s docstring for why the demo is synthetic.

Weeks 1-7 are the completed weeks. Each rostered player gets a projection near his current one
and an actual score drawn around it with his position's *measured* spread (``domain.variance``),
so the demo exercises the same variance model live mode does. A few situations are placed on
purpose, each to give one new tool something real to say:

1. **Sell high.** Team Alpha's Tariq Whitlock has scored well above what his targets justify:
   touchdowns on thin volume. ``buy_low_sell_high`` should flag him.
2. **Buy low.** A Team Bravo receiver has drawn heavy targets and scored little. The same tool
   should name him as the trade target.
3. **A disputed projection.** The second source projects Damon Rios well below ESPN this week,
   which widens his range in ``player_report``.
4. **An inattentive manager.** Team Echo leaves players on bye in its lineup and starts bench
   players on hunches, which shows up as low lineup accuracy in ``power_rankings``.

    uv run python scripts/make_demo_insights.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ffmcp.domain.models import LeagueSettings, Player, PlayerWeek, Team, UsageWeek
from ffmcp.domain.optimizer import optimize
from ffmcp.domain.variance import position_fraction

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SEED = 20260922
COMPLETED_WEEKS = range(1, 8)
CURRENT_WEEK = 8

SELL_HIGH = "Tariq Whitlock"
DISPUTED = "Damon Rios"
INATTENTIVE_TEAM_ID = 5
BUY_LOW_TEAM_ID = 2

_DEFAULT_RATES = {"QB": 17.0, "RB": 11.0, "WR": 11.0, "TE": 8.0, "K": 8.0, "D/ST": 7.0}
_OTHER_BYE_WEEKS = (5, 6, 7, 9, 10, 11, 12, 13, 14)


def _bye_weeks(
    pro_teams: list[str], on_bye_now: set[str], rng: np.random.Generator
) -> dict[str, int]:
    others = sorted(t for t in pro_teams if t not in on_bye_now)
    rng.shuffle(others)
    byes = {team: CURRENT_WEEK for team in on_bye_now}
    for index, team in enumerate(others):
        byes[team] = _OTHER_BYE_WEEKS[index % len(_OTHER_BYE_WEEKS)]
    return byes


def _rate(player: Player, rng: np.random.Generator) -> float:
    if player.projection is not None:
        return player.projection.points
    return round(_DEFAULT_RATES.get(player.position, 8.0) * float(rng.uniform(0.9, 1.2)), 1)


def _volume(position: str, xfp: float) -> dict[str, float]:
    """Invert the opportunity model roughly: volume that would earn about ``xfp`` points."""
    xfp = max(xfp, 1.0)
    if position == "WR":
        targets = xfp / 1.71
        return {"targets": targets, "receiving_air_yards": 9.0 * targets}
    if position == "TE":
        targets = xfp / 1.77
        return {"targets": targets, "receiving_air_yards": 7.0 * targets}
    if position == "RB":
        return {
            "carries": 0.7 * xfp / 0.72,
            "targets": 0.3 * xfp / 1.38,
            "receiving_air_yards": 0.3 * xfp / 1.38,
        }
    attempts = max(xfp - 3.8, 1.0) / 0.3875
    return {"attempts": attempts, "passing_air_yards": 7.5 * attempts, "carries": 4.0}


def main() -> None:
    rng = np.random.default_rng(SEED)
    league = json.loads((FIXTURES_DIR / "demo_league.json").read_text(encoding="utf-8"))
    settings = LeagueSettings.model_validate(league["settings"])
    teams = [Team.model_validate(t) for t in league["teams"]]
    free_agents = [Player.model_validate(p) for p in league["free_agents"][str(CURRENT_WEEK)]]

    rostered: list[tuple[int, Player, bool]] = []  # (team id, player, on IR)
    for team in teams:
        rostered.extend((team.team_id, p, False) for p in team.roster.players)
        rostered.extend((team.team_id, p, True) for p in team.roster.ir)
    everyone = [p for _, p, _ in rostered] + free_agents

    pro_teams = sorted({p.pro_team for p in everyone})
    on_bye_now = {p.pro_team for p in everyone if p.projection is None and not p.injured}
    byes = _bye_weeks(pro_teams, on_bye_now, rng)
    rates = {p.player_id: _rate(p, rng) for p in everyone}
    season_rates = {str(p.player_id): rates[p.player_id] for p in everyone if p.projection is None}

    buy_low_id = next(
        p.player_id
        for team in teams
        if team.team_id == BUY_LOW_TEAM_ID
        for p in sorted(team.roster.players, key=lambda q: -rates[q.player_id])
        if p.position == "WR"
    )

    recorded_scores: dict[tuple[int, int], float] = {}
    for week_key, matchups in league["matchups"].items():
        for m in matchups:
            recorded_scores[(int(week_key), m["home_team_id"])] = m["home_score"]
            if m["away_team_id"] is not None:
                recorded_scores[(int(week_key), m["away_team_id"])] = m["away_score"]

    # --- history: projected and actual for every rostered player, every completed week
    projected: dict[tuple[int, int], float] = {}
    actual: dict[tuple[int, int], float] = {}
    history: list[PlayerWeek] = []
    for week in COMPLETED_WEEKS:
        by_team: dict[int, list[tuple[Player, bool]]] = {}
        for team_id, player, on_ir in rostered:
            by_team.setdefault(team_id, []).append((player, on_ir))
        for team_id, members in by_team.items():
            week_proj: dict[int, float] = {}
            for player, _ in members:
                if byes.get(player.pro_team) == week:
                    continue
                points = round(rates[player.player_id] * float(rng.uniform(0.9, 1.1)), 1)
                week_proj[player.player_id] = points
                projected[(week, player.player_id)] = points
                sd = position_fraction(player.position) * points
                outcome = points + float(rng.normal(0.0, sd))
                if player.name == SELL_HIGH:
                    outcome = points + 9.0 + float(rng.normal(0.0, 2.0))
                if player.player_id == buy_low_id:
                    outcome = points - 6.5 + float(rng.normal(0.0, 1.5))
                actual[(week, player.player_id)] = round(max(0.0, outcome), 1)

            chosen = {pid: pts * float(rng.uniform(0.97, 1.03)) for pid, pts in week_proj.items()}
            # This week's injuries belong to this week: weeks 1-7 were played healthy.
            active = [
                p.model_copy(update={"injured": False, "injury_status": None})
                for p, on_ir in members
                if not on_ir
            ]
            if team_id == INATTENTIVE_TEAM_ID:
                # This manager sets the lineup from memory: byes go unnoticed, so a player
                # with no game can stay in the lineup, and most weeks a hunch starts a bench
                # player. Both are how real lineups lose projected points.
                for player, _ in members:
                    if player.player_id not in chosen:
                        chosen[player.player_id] = rates[player.player_id]
                if rng.random() < 0.8:
                    best = optimize(active, settings.starting_slots, chosen)
                    starting = {p.player_id for p in best.started}
                    bench = sorted(pid for pid in chosen if pid not in starting)
                    if bench:
                        chosen[bench[int(rng.integers(len(bench)))]] = 99.0
            lineup = optimize(active, settings.starting_slots, chosen)
            slot_of = {s.player.player_id: s.slot for s in lineup.slots if s.player is not None}
            # Scale the starters so the week adds up to the score the league fixture already
            # recorded for this team. The standings, the schedule and this history then tell
            # one consistent story, which is what makes all-play and luck meaningful here.
            recorded = recorded_scores[(week, team_id)]
            started_total = sum(actual.get((week, pid), 0.0) for pid in slot_of)
            if started_total > 0.0:
                for pid in slot_of:
                    if (week, pid) in actual:
                        actual[(week, pid)] = round(
                            actual[(week, pid)] * recorded / started_total, 1
                        )
            for player, on_ir in members:
                played = byes.get(player.pro_team) != week
                history.append(
                    PlayerWeek(
                        season=settings.season,
                        week=week,
                        player_id=player.player_id,
                        name=player.name,
                        position=player.position,
                        pro_team=player.pro_team,
                        eligible_slots=player.eligible_slots,
                        fantasy_team_id=team_id,
                        slot="IR" if on_ir else slot_of.get(player.player_id, "BE"),
                        projected=projected.get((week, player.player_id)),
                        actual=actual.get((week, player.player_id), 0.0),
                        played=played,
                    )
                )

    # --- usage: volume consistent with each player's rate, points from history where rostered
    depth: dict[tuple[str, str], list[Player]] = {}
    for p in everyone:
        if p.position in ("QB", "RB", "WR", "TE"):
            depth.setdefault((p.pro_team, p.position), []).append(p)
    snap_pct: dict[int, float] = {}
    for group in depth.values():
        group.sort(key=lambda q: -rates[q.player_id])
        ladder = (0.88, 0.62, 0.35, 0.2) if group[0].position != "QB" else (1.0, 0.05, 0.0, 0.0)
        for rank, p in enumerate(group):
            snap_pct[p.player_id] = ladder[min(rank, len(ladder) - 1)]

    usage: list[UsageWeek] = []
    for week in COMPLETED_WEEKS:
        for p in everyone:
            if p.position not in ("QB", "RB", "WR", "TE") or byes.get(p.pro_team) == week:
                continue
            xfp = rates[p.player_id] * float(rng.uniform(0.85, 1.15))
            if p.name == SELL_HIGH:
                xfp *= 0.7
            if p.player_id == buy_low_id:
                xfp *= 1.25
            volume = _volume(p.position, xfp)
            scored = actual.get((week, p.player_id))
            if scored is None:
                scored = round(max(0.0, xfp + float(rng.normal(0.0, 0.5 * xfp))), 1)
            targets = volume.get("targets", 0.0)
            usage.append(
                UsageWeek(
                    season=settings.season,
                    week=week,
                    player_id=p.player_id,
                    name=p.name,
                    position=p.position,
                    pro_team=p.pro_team,
                    snap_pct=snap_pct[p.player_id],
                    attempts=round(volume.get("attempts", 0.0), 1),
                    passing_air_yards=round(volume.get("passing_air_yards", 0.0), 1),
                    carries=round(volume.get("carries", 0.0), 1),
                    targets=round(targets, 1),
                    receiving_air_yards=round(volume.get("receiving_air_yards", 0.0), 1),
                    target_share=round(targets / 35.0, 3) if targets else None,
                    fantasy_points=scored,
                )
            )

    # --- second-opinion projections: close to ESPN, except where planted
    alt: dict[str, dict[str, float]] = {}
    for week in [*COMPLETED_WEEKS, CURRENT_WEEK]:
        week_alt: dict[str, float] = {}
        for p in everyone:
            base = (
                projected.get((week, p.player_id))
                if week != CURRENT_WEEK
                else (p.projection.points if p.projection else None)
            )
            if base is None:
                continue
            factor = float(rng.normal(1.0, 0.08))
            if week == CURRENT_WEEK and p.name == DISPUTED:
                factor = 0.55
            week_alt[str(p.player_id)] = round(max(0.0, base * factor), 1)
        alt[str(week)] = week_alt

    # --- this week's game lines for every NFL team that plays
    playing = sorted(t for t in pro_teams if byes.get(t) != CURRENT_WEEK)
    rng.shuffle(playing)
    lines = []
    for home, away in zip(playing[0::2], playing[1::2], strict=False):
        total = round(float(rng.uniform(38.0, 52.0)) * 2) / 2
        spread = round(float(rng.normal(0.0, 4.5)) * 2) / 2
        lines.append({"home": home, "away": away, "total": total, "home_spread": spread})

    payload: dict[str, Any] = {
        "bye_weeks": byes,
        "season_rates": season_rates,
        "history": [row.model_dump(mode="json") for row in history],
        "usage": [row.model_dump(mode="json", exclude_none=True) for row in usage],
        "alt_projections": alt,
        "game_lines": lines,
    }
    out = FIXTURES_DIR / "demo_insights.json"
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB): {len(history)} player-weeks, "
        f"{len(usage)} usage rows, {len(lines)} games"
    )


if __name__ == "__main__":
    main()

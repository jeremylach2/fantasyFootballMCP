"""How good is each team really, and how well is it being managed?

A win-loss record is a noisy measurement of team strength in head-to-head fantasy, because the
schedule decides who you play on the week you score 140. Two corrections, both computed from
completed weeks only:

* **All-play record.** Every week, each team is scored against *every* other team, not only its
  scheduled opponent. Scoring 130 in a week where 9 of 11 teams scored less is 9-2 in all-play,
  whoever you happened to face. The gap between actual wins and all-play expected wins is
  **luck**, and luck does not persist: a lucky team's record overstates it.
* **Lineup accuracy.** Projected points a team started, over the most it could have started
  by the projections *available at the time* (the same exact optimizer as ``optimize_lineup``).
  This isolates the manager's decisions from luck. The obvious alternative, actual points over
  the best lineup in hindsight, was measured on a real league and is mostly noise: a manager
  with 99.8% accuracy scored 83% of hindsight-optimal, because nobody knows which receiver will
  boom. Accuracy, by contrast, behaves like a trait: the same manager lagged the league in both
  2025 and 2026. A manager who leaves points on the bench has a better roster than his scores
  suggest, which matters when you are deciding whom to trade with. Hindsight bench points are
  still reported, as context rather than as a grade.

Both come from the same source: every rostered player-week of the completed weeks, where a
team's score is the sum of what its starters actually scored. Wins come from the standings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ffmcp.domain.models import Frozen, LeagueState, PlayerWeek
from ffmcp.domain.optimizer import optimize

LUCK_LABEL_WINS = 1.0
"""Wins of luck, either way, before a team is labelled lucky or unlucky."""

INATTENTIVE_ACCURACY = 97.0
"""Lineup accuracy, in percent, below which a manager is labelled inattentive. In the real
16-team league this was calibrated on, most managers ran 97-100% and the few who did not ran
90-96%, flagging 3 of 14 teams in 2025 and 3 of 16 in 2026."""


class ManagerReport(Frozen):
    team_id: int
    name: str
    games: int
    wins: float
    """Actual wins in the weeks analysed, ties counting half."""
    all_play_wins: int
    all_play_losses: int
    points_per_game: float
    lineup_accuracy: float | None = None
    """Percent of the best available *projected* lineup actually started, or ``None`` without
    lineup data: the manager's decisions, net of luck."""
    bench_points_per_game: float | None = None
    """Points per game the hindsight-optimal lineup would have added: luck plus decisions."""

    @property
    def all_play_pct(self) -> float:
        total = self.all_play_wins + self.all_play_losses
        return 100.0 * self.all_play_wins / total if total else 0.0

    @property
    def expected_wins(self) -> float:
        return self.games * self.all_play_pct / 100.0

    @property
    def luck(self) -> float:
        """Actual wins minus the wins an average schedule would have produced."""
        return self.wins - self.expected_wins

    @property
    def tags(self) -> tuple[str, ...]:
        tags = []
        if self.luck >= LUCK_LABEL_WINS:
            tags.append("lucky")
        elif self.luck <= -LUCK_LABEL_WINS:
            tags.append("unlucky")
        if self.lineup_accuracy is not None and self.lineup_accuracy < INATTENTIVE_ACCURACY:
            tags.append("inattentive")
        return tuple(tags)


def manager_reports(state: LeagueState, history: Sequence[PlayerWeek]) -> list[ManagerReport]:
    """One report per team, from ``history`` (every rostered player-week of the completed
    weeks), ordered by all-play percentage: the "power ranking" a record cannot give you."""
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for row in history:
        if row.fantasy_team_id is not None and row.started:
            scores[row.week][row.fantasy_team_id] += row.actual

    lineups = _lineup_quality(state.settings.starting_slots, history)

    reports = []
    for team in state.teams:
        week_scores = [
            (week, by_team[team.team_id])
            for week, by_team in sorted(scores.items())
            if team.team_id in by_team
        ]
        ap_wins = ap_losses = 0
        for week, mine in week_scores:
            others = [s for tid, s in scores[week].items() if tid != team.team_id]
            ap_wins += sum(1 for other in others if mine > other)
            ap_losses += sum(1 for other in others if mine < other)
        games = len(week_scores)
        quality = lineups.get(team.team_id)
        reports.append(
            ManagerReport(
                team_id=team.team_id,
                name=team.name,
                games=games,
                wins=min(team.wins + 0.5 * team.ties, float(games)),
                all_play_wins=ap_wins,
                all_play_losses=ap_losses,
                points_per_game=sum(s for _, s in week_scores) / games if games else 0.0,
                lineup_accuracy=None if quality is None else quality[0],
                bench_points_per_game=None if quality is None else quality[1],
            )
        )
    reports.sort(key=lambda r: (-r.all_play_pct, -r.points_per_game, r.team_id))
    return reports


def _lineup_quality(
    slots: Sequence[str], history: Sequence[PlayerWeek]
) -> dict[int, tuple[float, float]]:
    """``{team_id: (lineup accuracy %, hindsight bench points per game)}``."""
    by_team_week: dict[tuple[int, int], list[PlayerWeek]] = defaultdict(list)
    for row in history:
        if row.fantasy_team_id is not None and row.slot != "IR":
            by_team_week[(row.fantasy_team_id, row.week)].append(row)

    started_projected: dict[int, float] = defaultdict(float)
    best_projected: dict[int, float] = defaultdict(float)
    bench_points: dict[int, float] = defaultdict(float)
    weeks: dict[int, int] = defaultdict(int)
    for (team_id, _week), rows in by_team_week.items():
        started = [row for row in rows if row.started]
        started_projected[team_id] += sum(row.projected or 0.0 for row in started)
        best_projected[team_id] += optimize(
            [row.as_player(hindsight=False) for row in rows], slots
        ).projected_points
        started_actual = sum(row.actual for row in started)
        hindsight = optimize([row.as_player() for row in rows], slots).projected_points
        bench_points[team_id] += max(hindsight - started_actual, 0.0)
        weeks[team_id] += 1

    return {
        team_id: (
            min(100.0, 100.0 * started_projected[team_id] / best_projected[team_id]),
            bench_points[team_id] / weeks[team_id],
        )
        for team_id in weeks
        if best_projected[team_id] > 0.0
    }

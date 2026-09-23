"""Monte Carlo simulation of the rest of the season.

Each team's weekly score is Normal(mean, sd), clipped at zero. The mean is that team's
*optimal-lineup* projection (see ``domain.optimizer``), since every manager is assumed to start
their best legal lineup from here on, held constant across the remaining weeks because ESPN
publishes no future-week projections through this path. The sd is each starter's sd, summed
in quadrature and scaled by one measured factor for the correlation between them.

Per-player sds come from ``domain.variance``, where they were fit to a real league's season
rather than assumed. A caller that knows more about specific players (their own history, a
second projection source) passes a ``{player_id: sd}`` map; without one, every player gets his
position's measured spread.

Randomness arrives only through an injected ``numpy.random.Generator``: no module-level global
state, so a seeded run reproduces exactly. ``win_prob_delta`` evaluates two scenarios against a
single shared noise array, using common random numbers, for the reason given in its docstring.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ffmcp.domain.models import Frozen, LeagueState, Lineup, Player, Team
from ffmcp.domain.optimizer import optimize, projected_points
from ffmcp.domain.variance import TEAM_SD_FACTOR, player_sd

PlayerSds = Mapping[int, float]
"""``{player_id: weekly sd}`` from ``domain.variance.player_sds``."""

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

ProgressCallback = Callable[[int, int], None]
"""Called with ``(sims_completed, sims_total)``. The MCP layer forwards this to
``ctx.report_progress()``; ``domain/`` stays unaware that MCP exists."""

DEFAULT_SEED = 20260728
"""The default simulation seed is **fixed**, deliberately. Playoff odds that wobble by half a
point between two identical questions read as noise to a user and make a transcript
irreproducible; a caller who wants fresh draws passes their own Generator."""

DEFAULT_SIMS = 10_000
DEFAULT_DELTA_SIMS = 2_000
"""Fewer sims suffice for a delta than for a level, because common random numbers cancel most
of the variance between the two scenarios."""

_MIN_TEAM_SD = 1.0

_PROGRESS_STEPS = 10
_MAX_CHUNK = 2_500
"""Sims are run in chunks so progress can be reported about every 10% and so the noise array
stays a few megabytes instead of scaling with ``n_sims``."""


class ScoreModel(Frozen):
    """A team's weekly scoring distribution: Normal(mean, sd), clipped at zero."""

    team_id: int
    mean: float
    sd: float


class LineupScenario(Frozen):
    """One team fielding one lineup in one week: the unit ``win_prob_delta`` compares."""

    team_id: int
    week: int
    lineup: Lineup


class TeamOutlook(Frozen):
    """Where one team lands across the simulated seasons. All odds are percentages."""

    team_id: int
    name: str
    wins: int
    losses: int
    ties: int
    mean_final_wins: float
    playoff_odds: float
    title_odds: float
    seed_distribution: tuple[float, ...]
    """Probability of finishing in each seed, index 0 being the top seed."""


class SeasonOutcome(Frozen):
    n_sims: int
    playoff_team_count: int
    weeks_simulated: tuple[int, ...]
    teams: tuple[TeamOutlook, ...]

    def team(self, team_id: int) -> TeamOutlook:
        for outlook in self.teams:
            if outlook.team_id == team_id:
                return outlook
        raise KeyError(f"team {team_id} is not in this league")


def team_score_model(
    team: Team,
    slots: Sequence[str],
    projections: Mapping[int, float] | None = None,
    sds: PlayerSds | None = None,
) -> ScoreModel:
    """The scoring distribution for a team assumed to start its optimal lineup."""
    lineup = optimize(team.roster.players, slots, projections)
    return lineup_score_model(team.team_id, lineup, projections, sds)


def player_score_sd(player: Player, points: float, sds: PlayerSds | None = None) -> float:
    """One player's weekly scoring sd: from ``sds`` when it knows him, otherwise his position's
    measured spread (``domain.variance``).

    Broken out of ``lineup_score_model`` so a caller that wants to name the single
    highest-variance player on a roster (``analyze_matchup``) uses the identical assumption
    rather than re-deriving it.
    """
    if sds is not None and player.player_id in sds:
        return sds[player.player_id]
    return player_sd(player, points)


def lineup_score_model(
    team_id: int,
    lineup: Lineup,
    projections: Mapping[int, float] | None = None,
    sds: PlayerSds | None = None,
) -> ScoreModel:
    """The scoring distribution implied by one specific lineup."""
    variance = 0.0
    for player in lineup.started:
        points = projected_points(player, projections) or 0.0
        variance += player_score_sd(player, points, sds) ** 2
    sd = max(math.sqrt(variance) * TEAM_SD_FACTOR, _MIN_TEAM_SD)
    return ScoreModel(team_id=team_id, mean=lineup.projected_points, sd=sd)


def matchup_win_probability(home: ScoreModel, away: ScoreModel) -> float:
    """``home``'s win probability for a single head-to-head week, in percent.

    Closed-form rather than simulated: the difference of two independent Normals is itself
    Normal, so ``P(home > away)`` is one call to the standard normal CDF. Ties are split evenly,
    matching how ``_simulate_block`` scores them elsewhere in this module.
    """
    spread = home.mean - away.mean
    sd = math.sqrt(home.sd**2 + away.sd**2)
    if sd == 0.0:
        return 100.0 if spread > 0 else (0.0 if spread < 0 else 50.0)
    return 100.0 * 0.5 * (1.0 + math.erf(spread / (sd * math.sqrt(2.0))))


@dataclass(frozen=True)
class _Season:
    """Everything the vectorized inner loop needs, laid out as arrays once."""

    team_ids: tuple[int, ...]
    weeks: tuple[int, ...]
    game_week: IntArray
    game_home: IntArray
    game_away: IntArray
    base_wins: FloatArray
    base_points_for: FloatArray
    means: FloatArray  # (weeks, teams), per-week so a scenario can override a single cell
    sds: FloatArray
    team_means: FloatArray  # (teams,), used for playoff games, which have no schedule entry
    team_sds: FloatArray
    playoff_spots: int
    bracket_seeds: IntArray
    """Seed index occupying each first-round bracket position; -1 is a bye."""

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)

    @property
    def n_rounds(self) -> int:
        return max(int(self.bracket_seeds.size).bit_length() - 1, 0)


def _bracket_seeds(playoff_spots: int) -> IntArray:
    """Standard single-elimination seeding, padded to a power of two with byes.

    Four spots give ``[1, 4, 2, 3]``; six give ``[1, bye, 4, 5, 2, bye, 3, 6]``, which is the
    familiar shape where the top two seeds sit out the first round.
    """
    seeds = [0]
    while len(seeds) < playoff_spots:
        size = len(seeds) * 2
        seeds = [s for seed in seeds for s in (seed, size - 1 - seed)]
    return np.array([seed if seed < playoff_spots else -1 for seed in seeds], dtype=np.intp)


def _build_season(state: LeagueState, sds: PlayerSds | None = None) -> _Season:
    settings = state.settings
    slots = settings.starting_slots
    team_ids = tuple(team.team_id for team in state.teams)
    index_of = {team_id: index for index, team_id in enumerate(team_ids)}

    # Remaining schedule: this week onward. Games already played are in the teams' records, and
    # playoff matchups are simulated from seeding rather than from ESPN's bracket placeholders.
    # A matchup with a missing side is a bye and simply is not a game.
    remaining: list[tuple[int, int, int]] = []
    for matchup in state.matchups:
        if (
            matchup.is_playoff
            or not state.current_week <= matchup.week <= settings.reg_season_weeks
        ):
            continue
        home, away = (
            index_of.get(matchup.home_team_id or -1),
            index_of.get(matchup.away_team_id or -1),
        )
        if home is None or away is None:
            continue
        remaining.append((matchup.week, home, away))
    weeks = tuple(sorted({week for week, _, _ in remaining}))
    week_index = {week: index for index, week in enumerate(weeks)}

    models = [team_score_model(team, slots, sds=sds) for team in state.teams]
    team_means = np.array([model.mean for model in models], dtype=np.float64)
    team_sds = np.array([model.sd for model in models], dtype=np.float64)

    return _Season(
        team_ids=team_ids,
        weeks=weeks,
        game_week=np.array([week_index[week] for week, _, _ in remaining], dtype=np.intp),
        game_home=np.array([home for _, home, _ in remaining], dtype=np.intp),
        game_away=np.array([away for _, _, away in remaining], dtype=np.intp),
        base_wins=np.array([t.wins + 0.5 * t.ties for t in state.teams], dtype=np.float64),
        base_points_for=np.array([t.points_for for t in state.teams], dtype=np.float64),
        means=np.tile(team_means, (len(weeks), 1)),
        sds=np.tile(team_sds, (len(weeks), 1)),
        team_means=team_means,
        team_sds=team_sds,
        playoff_spots=max(0, min(settings.playoff_team_count, len(team_ids))),
        bracket_seeds=_bracket_seeds(max(1, min(settings.playoff_team_count, len(team_ids)))),
    )


@dataclass
class _Tally:
    """Counts accumulated across chunks. Everything the outcome needs is additive."""

    wins: FloatArray
    playoffs: FloatArray
    titles: FloatArray
    seeds: FloatArray

    @classmethod
    def empty(cls, n_teams: int) -> _Tally:
        return cls(
            wins=np.zeros(n_teams),
            playoffs=np.zeros(n_teams),
            titles=np.zeros(n_teams),
            seeds=np.zeros((n_teams, n_teams)),
        )

    def add(self, other: _Tally) -> None:
        self.wins += other.wins
        self.playoffs += other.playoffs
        self.titles += other.titles
        self.seeds += other.seeds


def _simulate_block(
    season: _Season, noise: FloatArray, means: FloatArray, sds: FloatArray
) -> _Tally:
    """One vectorized block of simulated seasons.

    ``noise`` is ``(n_sims, n_weeks + n_rounds, n_teams)`` of standard normals; the caller owns
    it, which is what makes common random numbers possible.
    """
    n_teams = season.n_teams
    n_weeks = season.n_weeks
    wins, _, order, seed_of_team = _regular_season(season, noise, means, sds)

    tally = _Tally.empty(n_teams)
    tally.wins = wins.sum(axis=0)
    tally.playoffs = (seed_of_team < season.playoff_spots).sum(axis=0).astype(np.float64)
    flat = (np.arange(n_teams)[None, :] * n_teams + seed_of_team).ravel()
    tally.seeds = (
        np.bincount(flat, minlength=n_teams * n_teams).reshape(n_teams, n_teams).astype(np.float64)
    )

    champions = _simulate_bracket(season, noise[:, n_weeks:, :], order)
    tally.titles = np.bincount(champions, minlength=n_teams).astype(np.float64)
    return tally


def _regular_season(
    season: _Season, noise: FloatArray, means: FloatArray, sds: FloatArray
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Play out the remaining regular season: ``(wins, week_results, order, seed_of_team)``.

    ``week_results[s, w, t]`` is team ``t``'s result in week ``w`` of simulation ``s``: 1 for a
    win, 0 for a loss, 0.5 for a tie, NaN for a bye. ``order[s]`` lists teams from the top seed
    down and ``seed_of_team`` is its inverse.
    """
    n_sims = noise.shape[0]
    n_weeks = season.n_weeks

    scores = means[None, :, :] + sds[None, :, :] * noise[:, :n_weeks, :]
    np.clip(scores, 0.0, None, out=scores)

    wins = np.tile(season.base_wins, (n_sims, 1))
    points_for = np.tile(season.base_points_for, (n_sims, 1))
    week_results = np.full((n_sims, n_weeks, season.n_teams), np.nan)
    for week, home, away in zip(season.game_week, season.game_home, season.game_away, strict=True):
        home_score = scores[:, week, home]
        away_score = scores[:, week, away]
        home_win = np.where(
            home_score > away_score, 1.0, np.where(home_score < away_score, 0.0, 0.5)
        )
        wins[:, home] += home_win
        wins[:, away] += 1.0 - home_win
        week_results[:, week, home] = home_win
        week_results[:, week, away] = 1.0 - home_win
        points_for[:, home] += home_score
        points_for[:, away] += away_score

    # Seeding: wins first, total points for as the tiebreak, the standard ESPN ordering.
    # lexsort takes its primary key last, and both keys are negated to sort descending.
    order: IntArray = np.lexsort((-points_for, -wins), axis=1)
    seed_of_team: IntArray = np.argsort(order, axis=1)  # inverse permutation
    return wins, week_results, order, seed_of_team


def _simulate_bracket(season: _Season, noise: FloatArray, order: IntArray) -> IntArray:
    """Single-elimination playoff, one fresh score per team per round.

    Playoff games use each team's own weekly distribution rather than a per-week mean: they fall
    outside the regular-season schedule, so any single-week lineup scenario has long since
    stopped applying.
    """
    n_sims = order.shape[0]
    bracket = np.full((n_sims, season.bracket_seeds.size), -1, dtype=np.intp)
    for position, seed in enumerate(season.bracket_seeds):
        if seed >= 0:
            bracket[:, position] = order[:, seed]

    for round_index in range(season.n_rounds):
        scores = season.team_means[None, :] + season.team_sds[None, :] * noise[:, round_index, :]
        upper, lower = bracket[:, 0::2], bracket[:, 1::2]
        upper_score = np.take_along_axis(scores, np.maximum(upper, 0), axis=1)
        lower_score = np.take_along_axis(scores, np.maximum(lower, 0), axis=1)
        winner = np.where(upper_score >= lower_score, upper, lower)
        bracket = np.where(lower < 0, upper, np.where(upper < 0, lower, winner))

    champions: IntArray = bracket[:, 0]
    return champions


def _run(
    season: _Season,
    means: FloatArray,
    sds: FloatArray,
    n_sims: int,
    rng: np.random.Generator,
    progress: ProgressCallback | None = None,
    shared_noise: list[FloatArray] | None = None,
) -> _Tally:
    """Run ``n_sims`` seasons in chunks, optionally recording or replaying the noise.

    ``shared_noise`` is the common-random-numbers hook: pass an empty list to record each
    chunk's draws, or the recorded list to replay them against a different scenario.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    chunk_size = max(1, min(math.ceil(n_sims / _PROGRESS_STEPS), _MAX_CHUNK))
    shape_tail = (season.n_weeks + season.n_rounds, season.n_teams)

    tally = _Tally.empty(season.n_teams)
    completed = 0
    chunk_index = 0
    while completed < n_sims:
        size = min(chunk_size, n_sims - completed)
        if shared_noise is not None and chunk_index < len(shared_noise):
            noise = shared_noise[chunk_index]
        else:
            noise = rng.standard_normal((size, *shape_tail))
            if shared_noise is not None:
                shared_noise.append(noise)
        tally.add(_simulate_block(season, noise, means, sds))
        completed += size
        chunk_index += 1
        if progress is not None:
            progress(completed, n_sims)
    return tally


def simulate_rest_of_season(
    state: LeagueState,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
    *,
    progress: ProgressCallback | None = None,
    player_sds: PlayerSds | None = None,
) -> SeasonOutcome:
    """Playoff odds, mean final wins, seed distribution and title odds for every team.

    The league's own playoff format is respected: ``playoff_team_count`` teams advance, seeded
    by wins then points for, into a standard single-elimination bracket with byes for the top
    seeds when the field is not a power of two.
    """
    generator = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    season = _build_season(state, player_sds)
    tally = _run(season, season.means, season.sds, n_sims, generator, progress)

    percent = 100.0 / n_sims
    teams = tuple(
        TeamOutlook(
            team_id=team.team_id,
            name=team.name,
            wins=team.wins,
            losses=team.losses,
            ties=team.ties,
            mean_final_wins=float(tally.wins[index] / n_sims),
            playoff_odds=float(tally.playoffs[index] * percent),
            title_odds=float(tally.titles[index] * percent),
            seed_distribution=tuple(float(value * percent) for value in tally.seeds[index]),
        )
        for index, team in enumerate(state.teams)
    )
    return SeasonOutcome(
        n_sims=n_sims,
        playoff_team_count=season.playoff_spots,
        weeks_simulated=season.weeks,
        teams=teams,
    )


def win_prob_delta(
    state: LeagueState,
    baseline: LineupScenario,
    counterfactual: LineupScenario,
    rng: np.random.Generator | None = None,
    *,
    n_sims: int = DEFAULT_DELTA_SIMS,
    common_random_numbers: bool = True,
    player_sds: PlayerSds | None = None,
) -> float:
    """Change in playoff odds, in percentage points, from fielding ``counterfactual`` instead.

    Why common random numbers: a lineup swap is worth a point or two out of ~115, and its
    effect on playoff odds is a fraction of a percentage point. Two independent simulations
    have a sampling error of roughly ±1 point at these sizes, so their difference would be
    mostly noise, and would change sign from one call to the next. Instead both scenarios are
    evaluated against *the same* draws of the noise array: every team's luck in every week is
    held identical, and the only difference left between the two runs is the thing being
    measured. The variance of the difference collapses, and a genuinely better lineup comes out
    positive essentially every time.

    ``common_random_numbers=False`` exists **only** to demonstrate that, in
    ``tests/domain/test_simulate.py``. No caller should pass it.
    """
    if baseline.team_id != counterfactual.team_id:
        raise ValueError("both scenarios must describe the same team")
    if baseline.week != counterfactual.week:
        raise ValueError("both scenarios must describe the same week")

    season = _build_season(state, player_sds)
    if baseline.week not in season.weeks:
        raise ValueError(f"week {baseline.week} is not in the remaining schedule")
    week = season.weeks.index(baseline.week)

    def arrays(scenario: LineupScenario) -> tuple[FloatArray, FloatArray]:
        model = lineup_score_model(scenario.team_id, scenario.lineup, sds=player_sds)
        return _scenario_arrays(season, [model], week_index=week)

    return _odds_delta(
        season,
        _team_index(season, baseline.team_id),
        arrays(baseline),
        arrays(counterfactual),
        n_sims=n_sims,
        rng=np.random.default_rng(DEFAULT_SEED) if rng is None else rng,
        common_random_numbers=common_random_numbers,
    )


def playoff_odds_delta(
    state: LeagueState,
    team_id: int,
    baseline: Sequence[ScoreModel],
    counterfactual: Sequence[ScoreModel],
    rng: np.random.Generator | None = None,
    *,
    n_sims: int = DEFAULT_DELTA_SIMS,
    common_random_numbers: bool = True,
    player_sds: PlayerSds | None = None,
) -> float:
    """Change in ``team_id``'s playoff odds, in percentage points, between two whole-season
    scenarios.

    Where ``win_prob_delta`` prices a single week's lineup, this prices a change that lasts the
    rest of the season, such as a trade or a waiver claim, by overriding the scoring models of
    every team it touches (each ``ScoreModel`` names its own team) for every remaining week.
    Both scenarios share one set of draws, for the reason ``win_prob_delta`` sets out at length.
    """
    season = _build_season(state, player_sds)
    return _odds_delta(
        season,
        _team_index(season, team_id),
        _scenario_arrays(season, baseline, week_index=None),
        _scenario_arrays(season, counterfactual, week_index=None),
        n_sims=n_sims,
        rng=np.random.default_rng(DEFAULT_SEED) if rng is None else rng,
        common_random_numbers=common_random_numbers,
    )


class WeekStakes(Frozen):
    """What one week's result is worth to one team's playoff chances, in percent."""

    week: int
    playoff_odds_if_win: float
    playoff_odds_if_loss: float

    @property
    def leverage(self) -> float:
        """Percentage points of playoff probability riding on this one game."""
        return self.playoff_odds_if_win - self.playoff_odds_if_loss


def week_stakes(
    state: LeagueState,
    team_id: int,
    week: int,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
    *,
    player_sds: PlayerSds | None = None,
) -> WeekStakes | None:
    """Playoff odds conditional on winning, and on losing, one specific week.

    Not a second simulation per outcome: one run of the season, split afterwards by how that
    week went in each simulated season. Every other week's randomness is shared between the two
    halves, so the gap between them is the week's own leverage and not simulation noise. ``None``
    when the week is not in the remaining schedule or the team has a bye in it.
    """
    season = _build_season(state, player_sds)
    if week not in season.weeks:
        return None
    team = _team_index(season, team_id)
    week_index = season.weeks.index(week)
    generator = np.random.default_rng(DEFAULT_SEED) if rng is None else rng

    won = made_after_win = lost = made_after_loss = 0.0
    remaining = n_sims
    while remaining > 0:
        size = min(_MAX_CHUNK, remaining)
        noise = generator.standard_normal((size, season.n_weeks + season.n_rounds, season.n_teams))
        _, results, _, seed_of_team = _regular_season(season, noise, season.means, season.sds)
        result = results[:, week_index, team]
        if np.all(np.isnan(result)):
            return None
        made = seed_of_team[:, team] < season.playoff_spots
        won += float(np.sum(result == 1.0))
        lost += float(np.sum(result == 0.0))
        made_after_win += float(np.sum(made & (result == 1.0)))
        made_after_loss += float(np.sum(made & (result == 0.0)))
        remaining -= size
    if won == 0.0 or lost == 0.0:
        return None
    return WeekStakes(
        week=week,
        playoff_odds_if_win=100.0 * made_after_win / won,
        playoff_odds_if_loss=100.0 * made_after_loss / lost,
    )


def _team_index(season: _Season, team_id: int) -> int:
    try:
        return season.team_ids.index(team_id)
    except ValueError:
        raise KeyError(f"team {team_id} is not in this league") from None


def _scenario_arrays(
    season: _Season, models: Sequence[ScoreModel], *, week_index: int | None
) -> tuple[FloatArray, FloatArray]:
    """The season's mean/sd grids with each model overlaid, on one week or on all of them."""
    means, sds = season.means.copy(), season.sds.copy()
    for model in models:
        team = _team_index(season, model.team_id)
        weeks = slice(None) if week_index is None else slice(week_index, week_index + 1)
        means[weeks, team] = model.mean
        sds[weeks, team] = model.sd
    return means, sds


def _odds_delta(
    season: _Season,
    team: int,
    baseline: tuple[FloatArray, FloatArray],
    counterfactual: tuple[FloatArray, FloatArray],
    *,
    n_sims: int,
    rng: np.random.Generator,
    common_random_numbers: bool,
) -> float:
    """Run both scenarios and difference one team's playoff odds.

    The shared noise list is the whole trick: recorded on the first run, replayed on the second.
    """
    noise: list[FloatArray] | None = [] if common_random_numbers else None
    base = _run(season, *baseline, n_sims, rng, shared_noise=noise)
    alt = _run(season, *counterfactual, n_sims, rng, shared_noise=noise)
    return float((alt.playoffs[team] - base.playoffs[team]) * 100.0 / n_sims)

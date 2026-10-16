"""Two-sided trade valuation and search.

What a player is worth is a property of the roster, not of the player. A second elite
quarterback in a one-QB league scores zero points for you every week he sits, while the same
player is a season-changing upgrade for the manager starting a replacement-level one. That
asymmetry is the only reason trades happen at all, and measuring it is what ``marginal_value``
does: the change in a roster's *optimal lineup* from adding or removing a player, which is
exactly why ``domain.optimizer`` had to be both correct and fast.

Search is a funnel, not an enumeration. Every pairing of two rosters is thousands of packages,
and simulating each would take a minute. So screen everything with additive marginal values at
each player's *season* rate (arithmetic, no simulation), evaluate the survivors exactly through
the optimizer week by week, and simulate playoff odds for only the handful still standing.

Only offers a real manager might accept survive, and that takes two filters, not one:

* **Both sides gain.** An offer only one side gains from is a fleece, and will be declined by a
  manager who does the same lineup arithmetic.
* **The market agrees it is fair.** Managers do not do that arithmetic. They anchor on names and
  on what players trade for, so an offer that gains both lineups a few points but hands over a
  backup quarterback for a star receiver is declined on sight, however good the math. Where
  market values are available (``providers/fantasycalc.py``), an offer is dropped when the
  partner gives up more than ``MARKET_RATIO_LIMIT`` times the market value they receive.

Fleeces and lopsided offers are filtered out rather than ranked low: the filters are the feature.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import numpy as np

from ffmcp.domain.models import Frozen, LeagueState, Player, Roster, TradeOffer
from ffmcp.domain.optimizer import optimize, projected_points
from ffmcp.domain.schedule import (
    WeekWeight,
    healthy,
    season_lineup_points,
    season_positional_points,
    weekly_rate,
)
from ffmcp.domain.simulate import DEFAULT_DELTA_SIMS, ProgressCallback, playoff_odds_delta

MIN_GAIN_POINTS = 0.5
"""Both sides must gain at least this many projected points over the remaining schedule for an
offer to count as mutually beneficial. Zero would surface trades whose entire case is
floating-point dust, and every such suggestion spends the reader's patience."""

MARKET_RATIO_LIMIT = 1.25
"""The most market value, as a multiple of what they receive, a partner is assumed willing to
give up. A judgement, not a fit: there is no record of offers declined. 1.25 leaves room for a
manager who needs the position, and rules out the 3,190-for-267 kind of offer outright."""

UNTRADED_POSITIONS = frozenset({"K", "D/ST"})
"""Left out of trade search unless asked for by name: nobody trades for a kicker, and both are
streamed off waivers in any league deep enough to need them."""

SCREENED_CANDIDATES = 60
"""Packages that survive the arithmetic screen and get an exact two-sided evaluation."""

SIMULATED_CANDIDATES = 20
"""Evaluated packages that get a playoff-odds simulation. Each one costs two simulation runs,
so this number is the knob that keeps ``search`` responsive."""

_PACKAGE_BREADTH = 8
"""Players considered from each side per partner: the cheapest to give, the best to get."""


class TradeEvaluation(Frozen):
    """One offer, priced from both sides. Deltas are projected points over the remaining
    schedule; ``my_playoff_odds_delta`` is in percentage points and is ``None`` until the offer
    survives far enough down the funnel to be worth simulating."""

    offer: TradeOffer
    my_value_delta: float
    partner_value_delta: float
    my_playoff_odds_delta: float | None = None
    my_drops: tuple[Player, ...] = ()
    partner_drops: tuple[Player, ...] = ()
    """Players each side would have to cut to stay at roster size. See ``apply_offer``."""
    market_given: float | None = None
    market_received: float | None = None
    """Market value of what I give and what I get, when market values are known."""

    @property
    def mutually_beneficial(self) -> bool:
        return (
            self.my_value_delta >= MIN_GAIN_POINTS and self.partner_value_delta >= MIN_GAIN_POINTS
        )

    @property
    def market_plausible(self) -> bool:
        """Whether the partner gives up no more than ``MARKET_RATIO_LIMIT`` times the market
        value they get back. True when values are unknown: no evidence is not a veto."""
        if self.market_given is None or self.market_received is None:
            return True
        return self.market_received <= MARKET_RATIO_LIMIT * self.market_given


def marginal_value(
    roster: Sequence[Player],
    player: Player,
    slots: Sequence[str],
    projections: Mapping[int, float] | None = None,
) -> float:
    """What ``player`` is worth to this roster, in optimal-lineup points per week.

    For a player already on the roster this is what losing him would cost; for anyone else it is
    what adding him would gain. Both are the same question asked from either side, so they are
    the same function.
    """
    on_roster = [other for other in roster if other.player_id != player.player_id]
    with_player = [*on_roster, player]
    return round(
        optimize(with_player, slots, projections).projected_points
        - optimize(on_roster, slots, projections).projected_points,
        4,
    )


def positional_impact(
    roster: Sequence[Player],
    give: Sequence[Player],
    get: Sequence[Player],
    slots: Sequence[str],
    weights: Sequence[WeekWeight] | None = None,
) -> dict[str, float]:
    """Change in *started* points per position, per week, if this offer is accepted.

    Measured through the lineup rather than over the players changing hands, because those are
    different numbers and only one of them is true. Trading away the backup quarterback on a
    one-QB roster costs nothing at quarterback however many points he is projected for: he
    was never going to be started. Summing the traded players' projections would report that
    non-loss as the headline of a trade worth making.

    With ``weights``, it is the average over the rest of the season, byes included, rather than
    this week alone: a trade is for the season, and this week may be someone's bye.
    """
    after_roster, _ = apply_offer(roster, give, get, slots, weights=weights)
    if weights:
        total_weight = sum(weight.weight for weight in weights) or 1.0
        before_ros = season_positional_points(roster, slots, weights)
        after_ros = season_positional_points(after_roster, slots, weights)
        deltas = {
            position: (after_ros.get(position, 0.0) - before_ros.get(position, 0.0)) / total_weight
            for position in {*before_ros, *after_ros}
        }
        return {position: round(d, 4) for position, d in deltas.items() if abs(d) >= 0.05}

    before = optimize(roster, slots)
    after = optimize(after_roster, slots)
    totals: dict[str, float] = {}
    for lineup, sign in ((after, 1.0), (before, -1.0)):
        for player in lineup.started:
            points = projected_points(player) or 0.0
            totals[player.position] = totals.get(player.position, 0.0) + sign * points
    return {position: round(delta, 4) for position, delta in totals.items() if abs(delta) >= 0.05}


def apply_offer(
    roster: Sequence[Player],
    give: Sequence[Player],
    get: Sequence[Player],
    slots: Sequence[str],
    projections: Mapping[int, float] | None = None,
    *,
    weights: Sequence[WeekWeight] | None = None,
) -> tuple[list[Player], list[Player]]:
    """The roster after a trade (or a waiver claim: ``give`` empty), cut back to its original
    size, plus whoever was cut.

    A 2-for-1 leaves one side a body over the limit, and every league makes that side drop
    someone. Ignoring the drop would make "get more players" look free and would quietly bias
    the search toward packages that no league would let you accept. Cuts come off the bench,
    never this week's lineup unless nobody else is left.

    Which bench player: with ``weights``, whoever the rest of the season misses least, tried
    one at a time through ``season_lineup_points``. That is the only rule that gets both of
    these right: a backup kicker who covers one bye is worth one week of kicking, and a second
    defense is worth nothing at all once the first one's bye has passed. The cheaper rule it
    replaced, lowest season rate across positions, cut the backup defense for every pickup,
    because defenses score least, whatever the roster needed. Without ``weights`` (a single
    week's question), the lowest rest-of-season rate goes, so a star on his bye is never cut.
    """
    given = {player.player_id for player in give}
    kept = [player for player in roster if player.player_id not in given]
    kept.extend(get)

    drops: list[Player] = []
    while len(kept) > len(roster):
        starting = {p.player_id for p in optimize(kept, slots, projections).started}
        droppable = [p for p in kept if p.player_id not in starting] or kept
        if weights:
            victim = max(
                droppable,
                key=lambda p: (
                    round(
                        season_lineup_points(
                            [q for q in kept if q.player_id != p.player_id], slots, weights
                        ),
                        4,
                    ),
                    -_keep_value(p, projections),
                    -p.player_id,
                ),
            )
        else:
            victim = min(droppable, key=lambda p: (_keep_value(p, projections), p.player_id))
        kept = [p for p in kept if p.player_id != victim.player_id]
        drops.append(victim)
    return kept, drops


def evaluate(
    offer: TradeOffer,
    my_roster: Roster | Sequence[Player],
    their_roster: Roster | Sequence[Player],
    slots: Sequence[str],
    *,
    weeks_remaining: int = 1,
    projections: Mapping[int, float] | None = None,
    my_weeks: Sequence[WeekWeight] | None = None,
    their_weeks: Sequence[WeekWeight] | None = None,
    market_values: Mapping[int, float] | None = None,
) -> TradeEvaluation:
    """Price an offer from both sides, exactly, through each roster's optimal lineup.

    The partner is assumed to evaluate the trade the same way: by what it does to the lineup
    they would actually field. ``market_values`` adds what the market thinks of each side, which
    is how a partner who does not do that arithmetic decides.

    With ``my_weeks``/``their_weeks`` (``domain.schedule.remaining_week_weights``), each side is
    priced week by week over the rest of the season, byes and playoff odds included, and
    ``weeks_remaining`` is ignored for that side. Without them, one week's delta is scaled by
    ``weeks_remaining``: the calendar-blind approximation.
    """
    mine, theirs = _players(my_roster), _players(their_roster)
    my_delta, my_drops = _side_delta(mine, offer.give, offer.get, slots, projections, my_weeks)
    their_delta, their_drops = _side_delta(
        theirs, offer.get, offer.give, slots, projections, their_weeks
    )
    my_scale = 1 if my_weeks is not None else weeks_remaining
    their_scale = 1 if their_weeks is not None else weeks_remaining
    given = received = None
    if market_values is not None:
        given = sum(market_values.get(p.player_id, 0.0) for p in offer.give)
        received = sum(market_values.get(p.player_id, 0.0) for p in offer.get)
    return TradeEvaluation(
        offer=offer,
        my_value_delta=round(my_delta * my_scale, 4),
        partner_value_delta=round(their_delta * their_scale, 4),
        my_drops=tuple(my_drops),
        partner_drops=tuple(their_drops),
        market_given=given,
        market_received=received,
    )


def search(
    state: LeagueState,
    my_team_id: int,
    max_results: int = 5,
    *,
    partner_team_id: int | None = None,
    positions_wanted: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
    n_sims: int = DEFAULT_DELTA_SIMS,
    progress: ProgressCallback | None = None,
    week_weights: Mapping[int, Sequence[WeekWeight]] | None = None,
    market_values: Mapping[int, float] | None = None,
) -> list[TradeEvaluation]:
    """Offers both lineups gain from and the market would call fair, ranked by what they do to
    *my* playoff odds.

    Ranking by odds rather than by points is the difference between "this trade is worth 4
    points a week" and "this trade is worth 6 points of playoff probability". Only the second
    one answers the question the manager actually asked.

    ``week_weights`` maps each team id to its ``remaining_week_weights``; teams missing from it
    (or all of them, when it is ``None``) are priced with the calendar-blind approximation.
    Kickers and defenses are left out unless ``positions_wanted`` names them.
    """
    slots = state.settings.starting_slots
    weeks_remaining = max(1, state.weeks_remaining)
    mine = list(state.team(my_team_id).roster.players)
    partners = [
        team
        for team in state.teams
        if team.team_id != my_team_id
        and (partner_team_id is None or team.team_id == partner_team_id)
    ]
    wanted = {position.upper() for position in positions_wanted} if positions_wanted else None
    excluded = UNTRADED_POSITIONS - (wanted or set())

    screened: list[tuple[float, TradeOffer]] = []
    for partner in partners:
        theirs = list(partner.roster.players)
        screened.extend(_screen_partner(mine, theirs, partner.team_id, slots, wanted, excluded))
    screened.sort(key=lambda scored: (-scored[0], _offer_key(scored[1])))

    # Stage two: the additive screen above double-counts players who compete for the same slot,
    # and knows nothing about byes, so every survivor is re-priced exactly before anything is
    # believed.
    evaluated: list[TradeEvaluation] = []
    for _, offer in screened[:SCREENED_CANDIDATES]:
        partner_roster = state.team(offer.partner_team_id).roster
        evaluation = evaluate(
            offer,
            mine,
            partner_roster,
            slots,
            weeks_remaining=weeks_remaining,
            my_weeks=week_weights.get(my_team_id) if week_weights else None,
            their_weeks=week_weights.get(offer.partner_team_id) if week_weights else None,
            market_values=market_values,
        )
        if evaluation.mutually_beneficial and evaluation.market_plausible:
            evaluated.append(evaluation)
    evaluated = _drop_padded_duplicates(evaluated)

    # Stage three: simulate only the finalists, and rank on odds rather than points.
    finalists = evaluated[:SIMULATED_CANDIDATES]
    simulated: list[TradeEvaluation] = []
    for index, evaluation in enumerate(finalists, start=1):
        simulated.append(with_playoff_odds(state, my_team_id, evaluation, rng, n_sims=n_sims))
        if progress is not None:
            progress(index, len(finalists))
    # A trade that lowers my playoff odds is not a suggestion, however many points it adds:
    # it makes the team I am chasing better by more than it makes me better.
    simulated = [e for e in simulated if (e.my_playoff_odds_delta or 0.0) >= 0.0]
    simulated.sort(key=lambda e: (-(e.my_playoff_odds_delta or 0.0), -e.my_value_delta))
    return _distinct(simulated, max_results)


def _distinct(evaluations: Sequence[TradeEvaluation], limit: int) -> list[TradeEvaluation]:
    """The best offers that are genuinely different trades, in the order given.

    An offer that shares a player on *both* sides with one already chosen, with the same
    partner, is a variant of it (the same core swap with a different sweetener), and a list of
    five variants of one idea is one suggestion wearing five hats. Once season value counts
    byes, a sweetener that covers a bye week is worth a few real points, so variants no longer
    tie exactly and ``_drop_padded_duplicates`` cannot catch them; this does.
    """
    chosen: list[TradeEvaluation] = []
    for evaluation in evaluations:
        offer = evaluation.offer
        give = {p.player_id for p in offer.give}
        get = {p.player_id for p in offer.get}
        is_variant = any(
            other.offer.partner_team_id == offer.partner_team_id
            and give & {p.player_id for p in other.offer.give}
            and get & {p.player_id for p in other.offer.get}
            for other in chosen
        )
        if not is_variant:
            chosen.append(evaluation)
        if len(chosen) == limit:
            break
    return chosen


def _package_size(evaluation: TradeEvaluation) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    offer = evaluation.offer
    return (len(offer.give) + len(offer.get), *_offer_key(offer)[1:])


def _drop_padded_duplicates(evaluations: Sequence[TradeEvaluation]) -> list[TradeEvaluation]:
    """Collapse offers that are the same trade with filler attached.

    A bench player worth nothing to either roster moves neither side's numbers, so the package
    builder cheerfully emits "give A / get B", "give A / get B + scrub" and "give A + scrub /
    get B" as three separate suggestions carrying identical deltas. They are one suggestion,
    and printing all three spends the reader's attention to say nothing. Keep the smallest
    package for each (partner, my delta, their delta), which is also the offer a manager
    would actually send, since every extra body is another thing for the other side to refuse.
    """
    best: dict[tuple[int, float, float], TradeEvaluation] = {}
    for evaluation in evaluations:
        key = (
            evaluation.offer.partner_team_id,
            round(evaluation.my_value_delta, 2),
            round(evaluation.partner_value_delta, 2),
        )
        incumbent = best.get(key)
        if incumbent is None or _package_size(evaluation) < _package_size(incumbent):
            best[key] = evaluation
    return sorted(best.values(), key=lambda e: (-e.my_value_delta, _offer_key(e.offer)))


def _keep_value(player: Player, projections: Mapping[int, float] | None) -> float:
    """How much a roster should want to keep ``player``: his rest-of-season rate."""
    if projections is not None:
        return projections.get(player.player_id) or 0.0
    return weekly_rate(player) or 0.0


def _players(roster: Roster | Sequence[Player]) -> list[Player]:
    return list(roster.players if isinstance(roster, Roster) else roster)


def _side_delta(
    roster: Sequence[Player],
    give: Sequence[Player],
    get: Sequence[Player],
    slots: Sequence[str],
    projections: Mapping[int, float] | None,
    weeks: Sequence[WeekWeight] | None = None,
) -> tuple[float, list[Player]]:
    after, drops = apply_offer(roster, give, get, slots, projections, weights=weeks)
    if weeks is not None:
        before_points = season_lineup_points(roster, slots, weeks)
        return season_lineup_points(after, slots, weeks) - before_points, drops
    before = optimize(roster, slots, projections).projected_points
    return optimize(after, slots, projections).projected_points - before, drops


def _offer_key(offer: TradeOffer) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Total order over offers, so ties rank identically on every run."""
    return (
        offer.partner_team_id,
        tuple(sorted(p.player_id for p in offer.give)),
        tuple(sorted(p.player_id for p in offer.get)),
    )


def season_rates(players: Iterable[Player]) -> dict[int, float]:
    """``{player_id: rest-of-season rate}``: one ordinary week of the season, nobody on bye and
    nobody hurt. The trade screen's projections."""
    return {p.player_id: rate for p in players if (rate := weekly_rate(p)) is not None}


def _screen_partner(
    mine: Sequence[Player],
    theirs: Sequence[Player],
    partner_team_id: int,
    slots: Sequence[str],
    wanted: set[str] | None,
    excluded: frozenset[str] | set[str],
) -> list[tuple[float, TradeOffer]]:
    """Cheap pass over one partner: build packages out of my surplus and their surplus.

    Each player is priced once against each roster: four marginal values per player pair
    instead of one optimization per package, and packages are scored by summing those. The sum
    is only an estimate (two receivers who would fill the same slot cannot both be worth their
    marginal value), which is why nothing here is trusted further than choosing what to evaluate
    properly.

    Prices are at each player's season rate, everyone healthy. By this week's projection, a
    star on his bye week is worth nothing to anyone, so he could never be a target, and my own
    bye-week starters looked like free surplus to give away.
    """
    mine = [healthy(p) for p in mine]
    theirs = [healthy(p) for p in theirs]
    rates = season_rates([*mine, *theirs])
    my_cost = {p.player_id: marginal_value(mine, p, slots, rates) for p in mine}
    my_gain = {p.player_id: marginal_value(mine, p, slots, rates) for p in theirs}
    their_cost = {p.player_id: marginal_value(theirs, p, slots, rates) for p in theirs}
    their_gain = {p.player_id: marginal_value(theirs, p, slots, rates) for p in mine}

    # My surplus is whoever costs me least to lose but is worth most to them, and vice versa.
    givable = sorted(
        (p for p in mine if p.position not in excluded),
        key=lambda p: (my_cost[p.player_id] - their_gain[p.player_id], p.player_id),
    )
    gettable = sorted(
        (
            p
            for p in theirs
            if p.position not in excluded and (wanted is None or p.position.upper() in wanted)
        ),
        key=lambda p: (their_cost[p.player_id] - my_gain[p.player_id], p.player_id),
    )
    givable, gettable = givable[:_PACKAGE_BREADTH], gettable[:_PACKAGE_BREADTH]

    scored: list[tuple[float, TradeOffer]] = []
    for give in _packages(givable):
        for get in _packages(gettable):
            if len(give) == len(get) == 2:  # 2-for-2 is a different, much larger search
                continue
            my_estimate = sum(my_gain[p.player_id] for p in get) - sum(
                my_cost[p.player_id] for p in give
            )
            their_estimate = sum(their_gain[p.player_id] for p in give) - sum(
                their_cost[p.player_id] for p in get
            )
            if my_estimate <= 0.0 or their_estimate <= 0.0:
                continue
            scored.append(
                (
                    my_estimate,
                    TradeOffer(partner_team_id=partner_team_id, give=tuple(give), get=tuple(get)),
                )
            )
    return scored


def _packages(players: Sequence[Player]) -> Iterable[tuple[Player, ...]]:
    """Singles and pairs: the 1-for-1 and 2-for-1 shapes managers actually propose."""
    yield from ((player,) for player in players)
    yield from combinations(players, 2)


def roster_after(
    roster: Sequence[Player],
    give: Sequence[Player],
    get: Sequence[Player],
    drops: Sequence[Player],
) -> list[Player]:
    """``roster`` with a trade's players swapped and its forced cuts made."""
    leaving = {p.player_id for p in (*give, *drops)}
    return [p for p in [*roster, *get] if p.player_id not in leaving]


def with_playoff_odds(
    state: LeagueState,
    my_team_id: int,
    evaluation: TradeEvaluation,
    rng: np.random.Generator | None = None,
    *,
    n_sims: int = DEFAULT_DELTA_SIMS,
) -> TradeEvaluation:
    """Attach the playoff-odds delta, simulating both rosters as they would stand after the
    trade (the cuts ``evaluate`` chose included) for every remaining week.

    The partner's roster changes too: a trade that makes the team you are chasing better can
    cost you more odds than the points it gains you, and only simulating both sides can show
    that. Public (rather than a ``search``-only helper) because ``evaluate_trade`` prices a
    single named offer the same way ``search`` prices its finalists.
    """
    offer = evaluation.offer
    mine = list(state.team(my_team_id).roster.players)
    theirs = list(state.team(offer.partner_team_id).roster.players)
    odds = playoff_odds_delta(
        state,
        my_team_id,
        {
            my_team_id: roster_after(mine, offer.give, offer.get, evaluation.my_drops),
            offer.partner_team_id: roster_after(
                theirs, offer.get, offer.give, evaluation.partner_drops
            ),
        },
        rng,
        n_sims=n_sims,
    )
    return evaluation.model_copy(update={"my_playoff_odds_delta": odds})

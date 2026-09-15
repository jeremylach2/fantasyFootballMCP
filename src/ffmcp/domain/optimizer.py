"""Exact lineup optimization.

Assign rostered players to starting slots so that total projected points is maximal, where each
slot admits a set of positions (``RB/WR/TE`` takes RB, WR or TE, and ``OP`` also takes QB), each
player fills at most one slot and each slot holds at most one player.

Why greedy is wrong: walking the slot list and dropping the best eligible player into each one
strands position-locked slots. ESPN declares its ``RB/WR`` flex *before* ``WR``, so a greedy
pass hands the flex the receiver the locked slot needed and backfills ``WR`` with whatever is
left. ``tests/domain/test_optimizer.py`` holds the concrete counterexample, and ``greedy()``
below exists only to keep that gap measurable.

Solved exactly as a maximum-weight bipartite matching, by the Hungarian method in its
successive-shortest-augmenting-path form with dual potentials. O(n^2 m) for n slots and m
candidates, which at roster scale (n ~ 10, m ~ 16) is microseconds. Hand-written, no scipy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ffmcp.domain.models import Lineup, Player, RosterSlot, Swap

SLOT_POSITIONS: Mapping[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "D/ST": frozenset({"D/ST"}),
    "RB/WR": frozenset({"RB", "WR"}),
    "WR/TE": frozenset({"WR", "TE"}),
    "RB/WR/TE": frozenset({"RB", "WR", "TE"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "OP": frozenset({"QB", "RB", "WR", "TE"}),
    "SUPERFLEX": frozenset({"QB", "RB", "WR", "TE"}),
}
"""Fallback slot-to-position table. Eligibility is normally read off the player's own
``eligible_slots``, which ESPN publishes per player and is authoritative (including for the odd
slot names a custom league invents). This table is the second opinion, so a provider that
reports a thin ``eligible_slots`` still gets a sensible flex."""

OUT_INJURY_STATUSES = frozenset(
    {"OUT", "INJURY_RESERVE", "IR", "SUSPENSION", "NOT_ACTIVE", "NA", "DOUBTFUL"}
)
"""Statuses that rule a player out. ``QUESTIONABLE`` is deliberately absent: a questionable
player is startable, and that uncertainty belongs in a caveat rather than in a silent benching.
"""

_FORBIDDEN = 1e9
"""Cost of an ineligible (slot, player) pair. Finite rather than infinite so the matching
arithmetic stays ordinary floats. It is never chosen, because every slot also has a zero-cost
"leave it empty" option."""


def slot_accepts(slot: str, player: Player) -> bool:
    """Whether ``player`` may legally be started in ``slot``."""
    if slot in player.eligible_slots:
        return True
    return player.position in SLOT_POSITIONS.get(slot, frozenset())


def projected_points(
    player: Player, projections: Mapping[int, float] | None = None
) -> float | None:
    """Points to optimize against, or ``None`` when the player has no game this week.

    An explicit ``projections`` map is authoritative: a player missing from it has no game,
    which is how a bye week reaches this module without the domain layer needing a calendar.
    With no map, the player's own attached projection is used on the same terms.
    """
    if projections is not None:
        return projections.get(player.player_id)
    return player.projection.points if player.projection is not None else None


def is_ruled_out(player: Player) -> bool:
    """Whether the player is ruled out for the week by injury, IR or suspension."""
    if player.injury_status is not None:
        return player.injury_status.upper() in OUT_INJURY_STATUSES
    return player.injured


def _candidates(
    players: Iterable[Player], projections: Mapping[int, float] | None
) -> tuple[list[tuple[Player, float]], list[Player]]:
    """Split a roster into startable ``(player, points)`` pairs and the unavailable rest.

    Sorted by descending points then player id, so every downstream tie is broken the same way
    on every run.
    """
    startable: list[tuple[Player, float]] = []
    unavailable: list[Player] = []
    seen: set[int] = set()
    for player in players:
        if player.player_id in seen:
            continue
        seen.add(player.player_id)
        points = projected_points(player, projections)
        if points is None or is_ruled_out(player):
            unavailable.append(player)
        else:
            startable.append((player, points))
    startable.sort(key=lambda pair: (-pair[1], pair[0].player_id))
    unavailable.sort(key=lambda player: player.player_id)
    return startable, unavailable


def _build_lineup(
    slots: Sequence[str],
    assigned: Sequence[tuple[Player, float] | None],
    startable: Sequence[tuple[Player, float]],
    unavailable: Sequence[Player],
) -> Lineup:
    used = {pair[0].player_id for pair in assigned if pair is not None}
    return Lineup(
        slots=tuple(
            RosterSlot(slot=slot, player=None if pair is None else pair[0])
            for slot, pair in zip(slots, assigned, strict=True)
        ),
        projected_points=round(sum(pair[1] for pair in assigned if pair is not None), 4),
        bench=tuple(player for player, _ in startable if player.player_id not in used),
        unavailable=tuple(unavailable),
    )


def optimize(
    players: Iterable[Player],
    slots: Sequence[str],
    projections: Mapping[int, float] | None = None,
) -> Lineup:
    """The highest-scoring legal lineup, exactly.

    Every slot is also offered a zero-weight "leave it empty" option, so a roster too thin (or
    too injured) to fill the lineup yields a valid partial lineup rather than an error.
    """
    startable, unavailable = _candidates(players, projections)
    n_slots, n_players = len(slots), len(startable)
    if n_slots == 0:
        return _build_lineup(slots, [], startable, unavailable)

    # Columns: one per candidate, then one empty-slot column per slot at cost 0. Those columns
    # guarantee a feasible assignment always exists, which is what lets an illegal pairing be
    # priced out of the matrix instead of needing special-case handling below.
    cost = [[0.0] * (n_players + n_slots) for _ in range(n_slots)]
    for index, slot in enumerate(slots):
        row = cost[index]
        for column, (player, points) in enumerate(startable):
            row[column] = -points if slot_accepts(slot, player) else _FORBIDDEN

    assignment = _min_cost_assignment(cost)
    assigned: list[tuple[Player, float] | None] = [
        startable[column] if 0 <= column < n_players and cost[row][column] < _FORBIDDEN else None
        for row, column in enumerate(assignment)
    ]
    return _build_lineup(slots, assigned, startable, unavailable)


def greedy(
    players: Iterable[Player],
    slots: Sequence[str],
    projections: Mapping[int, float] | None = None,
) -> Lineup:
    """The obvious wrong answer, kept **only** as a test baseline and for the README.

    Walks the league's declared slot order and takes the best eligible player left for each.
    Never call this from a tool: ``optimize`` is exact and no slower at roster scale.
    """
    startable, unavailable = _candidates(players, projections)
    assigned: list[tuple[Player, float] | None] = []
    used: set[int] = set()
    for slot in slots:
        pick = next(
            (
                pair
                for pair in startable
                if pair[0].player_id not in used and slot_accepts(slot, pair[0])
            ),
            None,
        )
        if pick is not None:
            used.add(pick[0].player_id)
        assigned.append(pick)
    return _build_lineup(slots, assigned, startable, unavailable)


def diff(
    current: Lineup,
    optimal: Lineup,
    projections: Mapping[int, float] | None = None,
) -> list[Swap]:
    """The minimal set of moves that turns ``current`` into ``optimal``.

    Compares the *set* of players started, not the slot-by-slot assignment: shuffling the same
    players between a locked slot and a flex changes nothing the manager has to do, so it is not
    a swap. Benchings are paired with promotions worst-first, which puts the largest gain in the
    headline move and makes the individual gains sum to the lineup's total point gain.
    """
    started_now = {player.player_id: player for player in current.started}
    started_optimal = {player.player_id: player for player in optimal.started}
    slot_of = {
        slot.player.player_id: slot.slot
        for slot in list(optimal.slots) + list(current.slots)
        if slot.player is not None
    }

    def points(player: Player | None) -> float:
        if player is None:
            return 0.0
        return projected_points(player, projections) or 0.0

    promoted = sorted(
        (player for pid, player in started_optimal.items() if pid not in started_now),
        key=lambda player: (-points(player), player.player_id),
    )
    benched = sorted(
        (player for pid, player in started_now.items() if pid not in started_optimal),
        key=lambda player: (points(player), player.player_id),
    )

    swaps: list[Swap] = []
    for index in range(max(len(promoted), len(benched))):
        player_in = promoted[index] if index < len(promoted) else None
        player_out = benched[index] if index < len(benched) else None
        anchor = player_in if player_in is not None else player_out
        assert anchor is not None
        swaps.append(
            Swap(
                slot=slot_of.get(anchor.player_id, ""),
                player_in=player_in,
                player_out=player_out,
                gain=round(points(player_in) - points(player_out), 4),
            )
        )
    swaps.sort(key=lambda swap: (-swap.gain, swap.player_in.player_id if swap.player_in else 0))
    return swaps


def _min_cost_assignment(cost: Sequence[Sequence[float]]) -> list[int]:
    """Hungarian algorithm, as successive shortest augmenting paths with dual potentials.

    ``cost`` is rectangular with rows <= columns; returns the column chosen for each row. Rows
    enter one at a time, each along a shortest augmenting path measured in reduced costs (kept
    non-negative by the potentials ``u`` and ``v``), so every intermediate matching is optimal
    for the rows seen so far and the final one is optimal overall.
    """
    n_rows = len(cost)
    if n_rows == 0:
        return []
    n_cols = len(cost[0])

    # One-indexed internally; index 0 is the sentinel each augmenting path starts from.
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    row_of_col = [0] * (n_cols + 1)
    previous_col = [0] * (n_cols + 1)

    for row in range(1, n_rows + 1):
        row_of_col[0] = row
        col = 0
        min_reduced = [float("inf")] * (n_cols + 1)
        visited = [False] * (n_cols + 1)
        while True:
            visited[col] = True
            current_row = row_of_col[col]
            weights = cost[current_row - 1]
            delta = float("inf")
            next_col = 0
            for candidate in range(1, n_cols + 1):
                if visited[candidate]:
                    continue
                reduced = weights[candidate - 1] - u[current_row] - v[candidate]
                if reduced < min_reduced[candidate]:
                    min_reduced[candidate] = reduced
                    previous_col[candidate] = col
                if min_reduced[candidate] < delta:
                    delta = min_reduced[candidate]
                    next_col = candidate
            for candidate in range(n_cols + 1):
                if visited[candidate]:
                    u[row_of_col[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_reduced[candidate] -= delta
            col = next_col
            if row_of_col[col] == 0:
                break
        while col:
            previous = previous_col[col]
            row_of_col[col] = row_of_col[previous]
            col = previous

    assignment = [-1] * n_rows
    for candidate in range(1, n_cols + 1):
        if row_of_col[candidate]:
            assignment[row_of_col[candidate] - 1] = candidate - 1
    return assignment

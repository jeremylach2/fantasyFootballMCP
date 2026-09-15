"""The lineup optimizer is exact, legal, and deterministic.

The first test is the whole justification for the module: it is the case where the obvious
greedy fill is strictly worse than the optimum. Everything after it is guardrail.
"""

from __future__ import annotations

import random

from ffmcp.domain.models import Lineup, Player, Projection, RosterSlot
from ffmcp.domain.optimizer import diff, greedy, optimize, projected_points, slot_accepts

# ESPN's own slot ordering puts the RB/WR flex ahead of the locked WR slot, so "walk the slots
# in order and take the best eligible player" really does meet this case in the wild.
ESPN_SLOTS = ("QB", "RB", "RB/WR", "WR", "TE", "D/ST", "K")


def make_player(
    player_id: int,
    position: str,
    points: float | None,
    *,
    name: str | None = None,
    eligible_slots: tuple[str, ...] | None = None,
    injury_status: str | None = None,
) -> Player:
    """A player with just enough on it to be optimized. ``points=None`` means no game."""
    default_slots = {
        "QB": ("QB", "OP"),
        "RB": ("RB", "RB/WR", "RB/WR/TE"),
        "WR": ("WR", "RB/WR", "RB/WR/TE"),
        "TE": ("TE", "RB/WR/TE"),
        "K": ("K",),
        "D/ST": ("D/ST",),
    }
    return Player(
        player_id=player_id,
        name=name or f"{position}{player_id}",
        position=position,
        eligible_slots=eligible_slots or default_slots[position],
        pro_team="FA",
        injured=injury_status is not None,
        injury_status=injury_status,
        projection=None if points is None else Projection(week=1, points=points),
    )


def assert_legal(lineup: Lineup, slots: tuple[str, ...]) -> None:
    assert tuple(s.slot for s in lineup.slots) == slots
    started = [s for s in lineup.slots if s.player is not None]
    ids = [s.player.player_id for s in started if s.player is not None]
    assert len(ids) == len(set(ids)), "a player was started twice"
    for filled in started:
        assert filled.player is not None
        assert slot_accepts(filled.slot, filled.player), (
            f"{filled.player.name} is not eligible for {filled.slot}"
        )


def test_greedy_strands_a_locked_slot_and_optimize_does_not() -> None:
    """The counterexample that justifies exact matching.

    Two slots, in ESPN's declared order: a RB/WR flex followed by a locked WR. Greedy fills the
    flex first with the best player it can see (the 17-point receiver), which leaves the WR slot
    nothing but a 3-point scrub. Playing the 12-point back in the flex instead frees the
    receiver for the slot only he can fill, and is worth 9 more points.
    """
    slots = ("RB/WR", "WR")
    receiver = make_player(1, "WR", 17.0, name="Top Receiver")
    back = make_player(2, "RB", 12.0, name="Solid Back")
    scrub = make_player(3, "WR", 3.0, name="Scrub Receiver")
    roster = [receiver, back, scrub]

    greedy_lineup = greedy(roster, slots)
    optimal = optimize(roster, slots)

    assert greedy_lineup.projected_points == 20.0
    assert optimal.projected_points == 29.0
    assert optimal.projected_points - greedy_lineup.projected_points == 9.0

    # Greedy strands the locked slot on the scrub; the optimum never does.
    assert greedy_lineup.slots[1].player == scrub
    assert optimal.slots[0].player == back
    assert optimal.slots[1].player == receiver

    assert_legal(greedy_lineup, slots)
    assert_legal(optimal, slots)


def random_roster(rng: random.Random) -> list[Player]:
    counts = {"QB": 2, "RB": 5, "WR": 5, "TE": 2, "K": 1, "D/ST": 1}
    roster: list[Player] = []
    player_id = 1
    for position, count in counts.items():
        for _ in range(count):
            roll = rng.random()
            points = None if roll < 0.08 else round(rng.uniform(0.0, 25.0), 1)
            status = "OUT" if 0.08 <= roll < 0.14 else None
            roster.append(make_player(player_id, position, points, injury_status=status))
            player_id += 1
    return roster


def test_optimal_is_never_worse_than_greedy_over_random_rosters() -> None:
    rng = random.Random(20260728)
    strictly_better = 0
    for _ in range(1000):
        roster = random_roster(rng)
        optimal = optimize(roster, ESPN_SLOTS)
        baseline = greedy(roster, ESPN_SLOTS)
        assert optimal.projected_points >= baseline.projected_points - 1e-9
        strictly_better += optimal.projected_points > baseline.projected_points + 1e-9
    # Not an assertion about the gap's size, just proof the sample actually exercises the case.
    assert strictly_better > 0


def test_every_optimized_lineup_is_legal_over_random_rosters() -> None:
    rng = random.Random(11)
    for _ in range(200):
        assert_legal(optimize(random_roster(rng), ESPN_SLOTS), ESPN_SLOTS)


def test_bye_week_and_ruled_out_players_are_never_started() -> None:
    slots = ("QB", "RB")
    on_bye = make_player(1, "QB", None, name="Bye QB")
    ruled_out = make_player(2, "RB", 30.0, name="Out RB", injury_status="OUT")
    playable_qb = make_player(3, "QB", 4.0)
    playable_rb = make_player(4, "RB", 2.0)

    lineup = optimize([on_bye, ruled_out, playable_qb, playable_rb], slots)

    assert [s.player for s in lineup.slots] == [playable_qb, playable_rb]
    assert set(lineup.unavailable) == {on_bye, ruled_out}
    assert lineup.projected_points == 6.0


def test_questionable_players_are_still_startable() -> None:
    questionable = make_player(1, "QB", 22.0, injury_status="QUESTIONABLE")
    healthy = make_player(2, "QB", 8.0)
    lineup = optimize([questionable, healthy], ("QB",))
    assert lineup.slots[0].player == questionable


def test_short_roster_yields_a_partial_lineup_rather_than_a_crash() -> None:
    lineup = optimize([make_player(1, "QB", 18.0)], ESPN_SLOTS)
    assert_legal(lineup, ESPN_SLOTS)
    assert lineup.projected_points == 18.0
    assert [s.slot for s in lineup.slots if s.player is None] == list(ESPN_SLOTS[1:])


def test_empty_roster_and_empty_slot_list_are_both_fine() -> None:
    assert optimize([], ESPN_SLOTS).projected_points == 0.0
    assert optimize([make_player(1, "QB", 10.0)], ()).slots == ()


def test_ties_produce_identical_output_across_runs() -> None:
    slots = ("RB", "WR", "RB/WR")
    # Every player projects the same, so the result is decided entirely by tie-breaking.
    roster = [make_player(i, "RB" if i % 2 else "WR", 10.0) for i in range(1, 7)]
    results = [optimize(list(reversed(roster)) if run % 2 else roster, slots) for run in range(8)]
    assert all(result == results[0] for result in results)


def test_diff_reports_only_real_moves_and_gains_that_sum_to_the_total() -> None:
    bench_star = make_player(1, "WR", 20.0, name="Bench Star")
    starter = make_player(2, "WR", 4.0, name="Weak Starter")
    back = make_player(3, "RB", 11.0, name="The Back")

    current = Lineup(
        slots=(
            RosterSlot(slot="RB", player=back),
            RosterSlot(slot="WR", player=starter),
        ),
        projected_points=15.0,
    )
    optimal = optimize([bench_star, starter, back], ("RB", "WR"))

    swaps = diff(current, optimal)
    assert len(swaps) == 1
    assert swaps[0].player_in == bench_star
    assert swaps[0].player_out == starter
    assert swaps[0].slot == "WR"
    assert sum(s.gain for s in swaps) == optimal.projected_points - current.projected_points


def test_diff_ignores_a_pure_slot_shuffle() -> None:
    """Moving the same two players between a locked slot and the flex is not a move to make."""
    back = make_player(1, "RB", 12.0)
    receiver = make_player(2, "WR", 9.0)
    shuffled = Lineup(
        slots=(
            RosterSlot(slot="RB/WR", player=back),
            RosterSlot(slot="WR", player=receiver),
        ),
        projected_points=21.0,
    )
    optimal = optimize([back, receiver], ("RB/WR", "WR"))
    assert optimal.projected_points == 21.0
    assert diff(shuffled, optimal) == []


def brute_force_best(players: list[Player], slots: tuple[str, ...]) -> float:
    """Exhaustive search, for small cases only. The ground truth `optimize` must match."""
    best = 0.0

    def search(slot_index: int, used: frozenset[int], total: float) -> None:
        nonlocal best
        if slot_index == len(slots):
            best = max(best, total)
            return
        search(slot_index + 1, used, total)  # leaving a slot empty is legal
        for player in players:
            if player.player_id in used or not slot_accepts(slots[slot_index], player):
                continue
            if player.projection is None or player.injury_status == "OUT":
                continue
            search(slot_index + 1, used | {player.player_id}, total + player.projection.points)

    search(0, frozenset(), 0.0)
    return best


def test_a_live_points_projections_map_can_yield_different_swaps_than_pregame() -> None:
    """``optimize()`` and ``diff()`` don't know or care where a
    ``projections`` map came from. A live-points map built from Monday's actual results should
    therefore be able to disagree with one built from Thursday's pregame projections on the
    same roster. This is the whole mechanism ``tools_lineup.py`` leans on for the recap.
    """
    slots = ("WR",)
    cold_starter = make_player(1, "WR", 18.0, name="Cold Starter")  # projected well, busts live
    hot_bench = make_player(2, "WR", 4.0, name="Hot Bench")  # barely projected, explodes live

    pregame_map = {1: 18.0, 2: 4.0}
    live_map = {1: 2.1, 2: 22.0}

    current = Lineup(slots=(RosterSlot(slot="WR", player=cold_starter),), projected_points=18.0)
    pregame_optimal = optimize([cold_starter, hot_bench], slots, pregame_map)
    live_optimal = optimize([cold_starter, hot_bench], slots, live_map)

    assert pregame_optimal.slots[0].player == cold_starter
    assert live_optimal.slots[0].player == hot_bench

    pregame_swaps = diff(current, pregame_optimal, pregame_map)
    live_swaps = diff(current, live_optimal, live_map)
    assert pregame_swaps == []
    assert len(live_swaps) == 1
    assert live_swaps[0].player_in == hot_bench
    assert live_swaps[0].player_out == cold_starter
    assert projected_points(hot_bench, live_map) == 22.0


def test_optimize_matches_exhaustive_search_on_small_random_instances() -> None:
    rng = random.Random(404)
    slot_pool = ("QB", "RB", "WR", "TE", "RB/WR", "RB/WR/TE", "WR/TE")
    for _ in range(300):
        slots = tuple(rng.choice(slot_pool) for _ in range(rng.randint(1, 5)))
        roster = [
            make_player(
                i,
                rng.choice(["QB", "RB", "WR", "TE"]),
                round(rng.uniform(0.0, 20.0), 1),
            )
            for i in range(1, rng.randint(2, 7) + 1)
        ]
        assert optimize(roster, slots).projected_points == round(brute_force_best(roster, slots), 4)

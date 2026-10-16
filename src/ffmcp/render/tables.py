"""Fixed-width text tables: no JSON punctuation, no repeated keys, headers abbreviated once
(defined in full at ``ffmcp://glossary``), null columns dropped rather than emitted as ``-``,
points rounded to one decimal.

Every renderer here takes domain objects and returns ``str`` for a tool declared
``structured_output=False``. Nothing in this module does I/O or arithmetic beyond formatting;
the numbers it prints were computed in ``domain/``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Literal

from ffmcp.domain.calibration import SourceAccuracy
from ffmcp.domain.league_intel import ManagerReport
from ffmcp.domain.models import (
    GameLine,
    LeagueState,
    MarketSignal,
    Player,
    PlayerWeek,
    Team,
    Transaction,
)
from ffmcp.domain.optimizer import is_game_time_decision, is_ruled_out, projected_points
from ffmcp.domain.roster import ByeWeek, DropCandidate, Pickup
from ffmcp.domain.simulate import SeasonOutcome, WeekStakes
from ffmcp.domain.usage import UsageProfile
from ffmcp.domain.valuation import remaining_strength_of_schedule
from ffmcp.domain.variance import (
    disagreement,
    expected_out_by_primary,
    floor_ceiling,
    sharp_disagreement,
)

Detail = Literal["compact", "standard", "full"]

_TRUNCATION_ELLIPSIS = "…"
_DISPUTE_THRESHOLD = 0.25
"""Relative ESPN/Sleeper gap worth a sentence: in the 2025 calibration data, projections
disputed by more than about a quarter missed by roughly 40% more than typical ones."""


def fmt_points(points: float | None) -> str:
    """One decimal place. The extra digits a raw float carries are noise a reader could
    mistake for precision the projection does not have."""
    return "" if points is None else f"{points:.1f}"


def fmt_pct(value: float | None, *, decimals: int = 1) -> str:
    return "" if value is None else f"{value:.{decimals}f}%"


def fmt_count(count: int | None) -> str:
    """``126385`` -> ``"126k"``: a column of six-digit counts is unreadable, and the precision
    is not the point (it is a measure of competition for a player)."""
    if count is None:
        return ""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.0f}k"
    return f"{count / 1_000_000:.1f}M"


def short_name(name: str, width: int = 15) -> str:
    """``"Josh Allen"`` -> ``"J. Allen"``. Truncated with an ellipsis if still too wide.

    A team defense (``"Commanders D/ST"``) keeps its first word: initialising it would produce
    ``"C. D/ST"``, which identifies nothing. Defense names are short enough to fit anyway.
    For *players* only: a fantasy team's name is not a first and last name (see
    ``team_label``).
    """
    parts = name.split()
    initialisable = len(parts) > 1 and parts[-1] != "D/ST"
    candidate = f"{parts[0][0]}. {' '.join(parts[1:])}" if initialisable else name
    return _truncate(candidate, width)


def team_label(name: str, width: int = 16) -> str:
    """A fantasy team's name, trimmed and truncated with an ellipsis. Never initialised:
    run through ``short_name``, "Best Team Ever" became "B. Team Ever" and "0 out of 5" became
    "0. out of 5"."""
    return _truncate(" ".join(name.split()), width)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(1, width - 1)] + _TRUNCATION_ELLIPSIS


def status_flag(
    player: Player, points: float | None, week: int | None = None, *, locked: bool = False
) -> str:
    """A short flag: ruled out, questionable, on bye, or locked. Blank when none apply.

    A bye is read off ``Player.bye_week`` when the week is known. ESPN projects a player on bye
    for ``0.0`` points rather than nothing, so "no projection" alone never saw a live bye; it
    remains the fallback for a provider that does not know byes.
    """
    if is_ruled_out(player):
        return (player.injury_status or "O")[:1]
    if player.injury_status and player.injury_status.upper() == "QUESTIONABLE":
        return "Q"
    if (week is not None and player.bye_week == week) or points is None:
        return "BYE"
    if locked:
        return "LCK"
    return ""


def render_table(
    headers: Sequence[str], widths: Sequence[int], rows: Sequence[Sequence[str]]
) -> str:
    """One header line plus one line per row, space-padded to ``widths`` and right-trimmed.

    A cell wider than its column is truncated rather than allowed to push the rest of the row
    sideways. A fixed-width table that silently stops being fixed-width is worse than a
    slightly clipped value: the model is being asked to read this positionally.
    """

    def cell_text(cell: str, width: int) -> str:
        if len(cell) <= width:
            return cell.ljust(width)
        return cell[: max(1, width - 1)] + _TRUNCATION_ELLIPSIS

    def line(cells: Sequence[str]) -> str:
        return " ".join(
            cell_text(cell, width) for cell, width in zip(cells, widths, strict=True)
        ).rstrip()

    return "\n".join([line(headers), *(line(row) for row in rows)])


_SLOT_LABELS = {
    "RB/WR/TE": "FLEX",
    "RB/WR": "FLEX",
    "WR/TE": "FLEX",
    "SUPERFLEX": "SFLEX",
    "OP": "SFLEX",
}
"""Display names for the slots whose ESPN identifiers do not fit a column. ``FLEX`` is what
every fantasy interface (ESPN's included) calls a ``RB/WR/TE`` slot, so it is the shorter *and*
the more familiar label. The eligibility rules behind it live in ``domain.optimizer``, which
never sees these strings."""

_SLOT_ORDER = ("QB", "RB", "WR", "TE", "FLEX", "SFLEX", "D/ST", "K")
"""Display order for a lineup, the order every fantasy app uses. ESPN's own slot order is an
internal one (WR, RB, RB, WR, TE, QB, FLEX ...) that reads as noise."""


def slot_label(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot)


def slot_rank(slot: str) -> int:
    label = slot_label(slot)
    return _SLOT_ORDER.index(label) if label in _SLOT_ORDER else len(_SLOT_ORDER)


# ---------------------------------------------------------------------------
# get_my_team
# ---------------------------------------------------------------------------

_ROSTER_HEADERS: dict[Detail, tuple[str, ...]] = {
    "compact": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "ST"),
    "standard": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "LIVE", "ST", "OWN%"),
    "full": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "LIVE", "ST", "OWN%", "STRT%", "ADD/24H"),
}
_ROSTER_WIDTHS: dict[Detail, tuple[int, ...]] = {
    "compact": (5, 16, 4, 3, 5, 3),
    "standard": (5, 16, 4, 3, 5, 5, 3, 5),
    "full": (5, 16, 4, 3, 5, 5, 3, 5, 6, 7),
}


def _roster_row(
    slot: str, player: Player | None, detail: Detail, week: int, locked: Collection[int]
) -> list[str]:
    slot = slot_label(slot)
    n_cols = len(_ROSTER_HEADERS[detail])
    if player is None:
        return [slot, "(empty)", *([""] * (n_cols - 2))]
    points = projected_points(player)
    market: MarketSignal | None = player.market
    row = [
        slot,
        short_name(player.name),
        player.position,
        player.pro_team,
        fmt_points(points),
    ]
    if detail in ("standard", "full"):
        row.append(fmt_points(player.live_points))
    row.append(status_flag(player, points, week, locked=player.player_id in locked))
    if detail in ("standard", "full"):
        row.append(fmt_pct(market.percent_owned if market else None, decimals=0))
    if detail == "full":
        row.append(fmt_pct(market.percent_started if market else None, decimals=0))
        row.append(fmt_count(market.trending_adds if market else None))
    return row


def render_injury_summary(players: Sequence[Player], week: int | None = None) -> str:
    """``"Injuries: none"``, or one flag per player who is out, doubtful, questionable or on
    IR/suspension, using the same ``status_flag`` logic as the ``ST`` column, so the two can
    never disagree. Byes and locks are not injuries and are left out: they have their own
    column."""
    flags = [
        f"{short_name(player.name)} ({flag})"
        for player in players
        if (flag := status_flag(player, projected_points(player), week)) and flag != "BYE"
    ]
    return "Injuries: none" if not flags else "Injuries: " + ", ".join(flags)


def render_roster(
    team: Team,
    week: int,
    detail: Detail,
    *,
    note: str | None = None,
    locked: Collection[int] = (),
) -> str:
    """``get_my_team``: the team's current lineup, bench and IR as ESPN has them set, not the
    optimizer's suggestion. Use ``optimize_lineup`` for that."""
    headers = _ROSTER_HEADERS[detail]
    widths = _ROSTER_WIDTHS[detail]
    ordered = sorted(team.roster.slots, key=lambda s: slot_rank(s.slot))
    rows = [_roster_row(slot.slot, slot.player, detail, week, locked) for slot in ordered]
    rows.extend(_roster_row("BE", player, detail, week, locked) for player in team.roster.bench)
    rows.extend(_roster_row("IR", player, detail, week, locked) for player in team.roster.ir)
    title = f"Week {week} — {team.name} ({team.wins}-{team.losses}-{team.ties})"
    roster_players = [
        *(slot.player for slot in team.roster.slots if slot.player is not None),
        *team.roster.bench,
        *team.roster.ir,
    ]
    lines = [title]
    if note:
        lines.append(note)
    lines.append(render_injury_summary(roster_players, week))
    lines.append(render_table(headers, widths, rows))
    return "\n".join(lines)


def render_week_recap(
    team: Team, week: int, rows: Sequence[PlayerWeek], *, best_projected: float, best_actual: float
) -> str:
    """``get_my_team`` for a completed week: who started, what each was projected, and what
    each actually scored, with the best lineups by the projections of the day and in
    hindsight."""
    lineup = sorted(
        rows,
        key=lambda r: (not r.started, slot_rank(r.slot or "BE") if r.started else 99, -r.actual),
    )
    started = [r for r in rows if r.started]
    scored = sum(r.actual for r in started)
    projected = sum(r.projected or 0.0 for r in started)
    table = render_table(
        ("SLOT", "PLAYER", "POS", "TM", "PROJ", "PTS"),
        (5, 16, 4, 3, 5, 5),
        [
            [
                slot_label(r.slot or "BE"),
                short_name(r.name),
                r.position,
                r.pro_team,
                fmt_points(r.projected),
                fmt_points(r.actual) if r.played else "BYE",
            ]
            for r in lineup
            if r.slot != "IR"
        ],
    )
    return (
        f"Week {week} recap — {team.name}: started {scored:.1f} (projected {projected:.1f}). "
        f"Best lineup by that week's projections: {best_projected:.1f} projected; "
        f"in hindsight: {best_actual:.1f}.\n{table}"
    )


# ---------------------------------------------------------------------------
# find_waiver_targets
# ---------------------------------------------------------------------------

_WAIVER_HEADERS: dict[Detail, tuple[str, ...]] = {
    "compact": ("PLAYER", "POS", "TM", "PROJ", "ROS", "VAL", "DROP"),
    "standard": ("PLAYER", "POS", "TM", "PROJ", "ROS", "VAL", "DROP", "ADD/24H"),
    "full": ("PLAYER", "POS", "TM", "PROJ", "ROS", "VAL", "DROP", "ADD/24H", "OWN%"),
}
_WAIVER_WIDTHS: dict[Detail, tuple[int, ...]] = {
    "compact": (16, 4, 3, 5, 5, 5, 16),
    "standard": (16, 4, 3, 5, 5, 5, 16, 7),
    "full": (16, 4, 3, 5, 5, 5, 16, 7, 5),
}


def _waiver_row(pickup: Pickup, detail: Detail) -> list[str]:
    player = pickup.player
    points = projected_points(player)
    on_bye = player.projection is not None and player.projection.week == player.bye_week
    row = [
        short_name(player.name),
        player.position,
        player.pro_team,
        "BYE" if points is None and on_bye else fmt_points(points),
        f"{pickup.season_value:+.0f}",
        f"{pickup.week_value:+.1f}",
        short_name(pickup.drop.name, 16) if pickup.drop is not None else "(no drop needed)",
    ]
    if detail in ("standard", "full"):
        row.append(fmt_count(player.market.trending_adds if player.market else None))
    if detail == "full":
        owned = player.market.percent_owned if player.market else None
        row.append(fmt_pct(owned, decimals=0))
    return row


def return_week(player: Player) -> int | None:
    """The first week a player's outlook projects him to score, or ``None``."""
    return next((p.week for p in player.outlook if p.points > 0.0), None)


def _stash_text(pickup: Pickup) -> str:
    player = pickup.player
    owned = player.market.percent_owned if player.market else None
    parts = [f"{short_name(player.name)} {player.position} {player.pro_team}"]
    if owned is not None:
        parts.append(f"{owned:.0f}% owned")
    back = return_week(player)
    if back is not None:
        parts.append(f"back wk {back}, ROS {pickup.season_value:+.0f}")
    else:
        parts.append("no return projected")
    return " ".join(parts[:1]) + " (" + ", ".join(parts[1:]) + ")"


def defense_matchups(
    defenses: Sequence[Player], lines: Mapping[str, Sequence[GameLine]], *, games: int = 2
) -> str | None:
    """``"Matchups (opponent implied pts, lower is better): Bucs D/ST NYG 18.0, ATL 21.5"``:
    what the betting market expects each defense's next opponents to score, the best public
    read on a defense's week. Context beside the ranking, never an input to it: this server has
    no history of lines to test how much it adds to a projection."""
    parts = []
    for defense in defenses:
        upcoming = list(lines.get(defense.pro_team, ()))[:games]
        if upcoming:
            opponents = ", ".join(
                f"{line.opponent} {line.total - line.implied_total:.1f}" for line in upcoming
            )
            parts.append(f"{short_name(defense.name)} {opponents}")
    if not parts:
        return None
    return "Matchups (opponent implied pts, lower is better): " + "; ".join(parts)


def render_waiver_targets(
    pickups: Sequence[Pickup],
    detail: Detail,
    *,
    limit: int,
    near_misses: Sequence[Pickup] = (),
    stashes: Sequence[Pickup] = (),
    open_ir_slots: int = 0,
    handcuffs: Sequence[str] = (),
    matchups: str | None = None,
) -> str:
    """``find_waiver_targets``: complete moves (add and drop) ranked by what they do to the
    rest of the season (ROS), this week's part shown beside it (VAL), the drop always named,
    because a pickup recommendation with no drop is useless advice.

    When nothing helps the season, ``near_misses`` show how close the best moves came, so "the
    wire is empty" reads differently from "everything was a point short". Injured players who
    could sit in an open IR slot are listed separately: no drop now, value once back.
    """
    headers = _WAIVER_HEADERS[detail]
    widths = _WAIVER_WIDTHS[detail]
    lines: list[str] = []
    if pickups:
        lines.append(render_table(headers, widths, [_waiver_row(p, detail) for p in pickups]))
        if len(pickups) >= limit:
            lines.append("(raise limit to see more)")
    elif near_misses:
        lines.append("No move helps your season. Closest:")
        lines.append(render_table(headers, widths, [_waiver_row(p, detail) for p in near_misses]))
    else:
        lines.append("No waiver targets clear the bar.")
    if stashes:
        slot = (
            f"{open_ir_slots} open IR slot{'s' if open_ir_slots != 1 else ''}"
            if open_ir_slots
            else "no open IR slot, so a drop is needed"
        )
        lines.append(f"Stash ({slot}): " + "; ".join(_stash_text(p) for p in stashes))
    if matchups:
        lines.append(matchups)
    if handcuffs:
        lines.append("Handcuffs: " + "; ".join(handcuffs))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# league_standings
# ---------------------------------------------------------------------------

_STANDINGS_HEADERS = ("RK", "ID", "TEAM", "W-L", "PF", "PA", "PLAYOFF%", "SOS")
_STANDINGS_WIDTHS = (3, 3, 16, 6, 7, 7, 9, 6)


def _week_in_progress_note(state: LeagueState) -> str | None:
    """ESPN does not update ``Team.wins``/``points_for`` until a week is finalized. Observed:
    still 0-0-0 the Monday morning after week 1's early games, even though that same week's
    box-score-derived live scores are already nonzero. When this happens, the standings table
    displays a note about the week still being in progress rather than reading as the season
    not having started.
    """
    if any(team.wins or team.losses or team.ties for team in state.teams):
        return None
    current_week_matchups = [m for m in state.matchups if m.week == state.current_week]
    if not any(m.home_score > 0.0 or m.away_score > 0.0 for m in current_week_matchups):
        return None
    return (
        f"Week {state.current_week} is still in progress — standings reflect ESPN's last "
        "finalized week."
    )


def render_standings(
    state: LeagueState, detail: Detail, playoff_odds: Mapping[int, float] | None = None
) -> str:
    """``league_standings``. ``PLAYOFF%`` is this server's own simulation (the same number
    ``simulate_season`` and ``analyze_matchup`` show), passed in as ``{team_id: percent}``;
    ESPN's own estimate is never shown, so a manager sees one number per team, not two.

    ``ID`` is ``Team.team_id``, distinct from ``RK`` (rank by record). It is the only place a
    caller can read a team's id from prose rather than guess it from rank position: tools like
    ``evaluate_trade`` take a numeric ``partner_team_id``, and rank and id agree only when the
    standings happen to be sorted by id.
    """
    teams = sorted(state.teams, key=lambda team: team.standing)
    rows = []
    for team in teams:
        sos = remaining_strength_of_schedule(team.team_id, state)
        odds = playoff_odds.get(team.team_id) if playoff_odds is not None else None
        rows.append(
            [
                str(team.standing),
                str(team.team_id),
                team_label(team.name),
                f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else ""),
                f"{team.points_for:.1f}",
                f"{team.points_against:.1f}",
                fmt_pct(odds),
                fmt_pct(sos) if sos is not None else "",
            ]
        )
    lines = [render_table(_STANDINGS_HEADERS, _STANDINGS_WIDTHS, rows)]
    note = _week_in_progress_note(state)
    if note is not None:
        lines.append(note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# simulate_season
# ---------------------------------------------------------------------------

_SEASON_HEADERS = ("TEAM", "REC", "MEANW", "PLAYOFF%", "TITLE%")
_SEASON_WIDTHS = (18, 7, 6, 9, 7)


def render_season_outcome(
    outcome: SeasonOutcome,
    *,
    highlight_team_id: int | None = None,
    caveats: Sequence[str] = (),
) -> str:
    ordered = sorted(outcome.teams, key=lambda t: (t.team_id != highlight_team_id, -t.playoff_odds))
    rows = []
    for team in ordered:
        record = f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else "")
        marker = "*" if team.team_id == highlight_team_id else ""
        rows.append(
            [
                marker + team_label(team.name, 18 - len(marker)),
                record,
                f"{team.mean_final_wins:.1f}",
                fmt_pct(team.playoff_odds),
                fmt_pct(team.title_odds),
            ]
        )
    if outcome.weeks_simulated:
        weeks = f"weeks {outcome.weeks_simulated[0]}-{outcome.weeks_simulated[-1]}"
        header = f"{outcome.n_sims:,} sims, {weeks}:"
    else:
        header = f"{outcome.n_sims:,} sims:"
    return "\n".join([header, render_table(_SEASON_HEADERS, _SEASON_WIDTHS, rows), *caveats])


# ---------------------------------------------------------------------------
# analyze_matchup
# ---------------------------------------------------------------------------


def render_matchup(
    *,
    week: int,
    my_team: Team,
    opp_team: Team,
    my_projected: float,
    opp_projected: float,
    my_live_score: float,
    opp_live_score: float,
    win_probability: float,
    edges: Sequence[tuple[str, float]],
    swing_player: Player | None,
    swing_sd: float,
    stakes: WeekStakes | None = None,
    note: str | None = None,
    live: tuple[float, float, float] | None = None,
) -> str:
    """``analyze_matchup``: totals, win probability, the biggest positional edges, and the
    single highest-variance swing player. ``live`` is ``(win probability, my projected points
    still to play, theirs)`` once games have started (``domain.live``).

    ``my_live_score``/``opp_live_score`` are this week's actual in-game score so far. Omitted
    when both are still 0.0, which means the games haven't kicked off yet and there is nothing
    live to show.
    """
    lines = []
    if my_live_score > 0.0 or opp_live_score > 0.0:
        lines.append(
            f"Live: {my_team.name} {my_live_score:.1f} vs {opp_team.name} {opp_live_score:.1f}"
        )
    if live is not None:
        lines.append(
            f"Live win prob: {live[0]:.0f}% ({live[1]:.1f} vs {live[2]:.1f} projected still "
            "to play)"
        )
    lines.append(
        f"Week {week} (proj): {my_team.name} {my_projected:.1f} vs {opp_team.name} "
        f"{opp_projected:.1f} — {win_probability:.0f}% win prob"
    )
    if note:
        lines.append(note)
    lines.append("Biggest edges:")
    if not edges:
        lines.append("  none — rosters are projected evenly by position")
    for position, delta in edges:
        who = my_team.name if delta >= 0 else opp_team.name
        lines.append(f"  {position}: {who} +{abs(delta):.1f}")
    if swing_player is not None:
        lines.append(
            f"Swing player: {short_name(swing_player.name, 20)} "
            f"({swing_player.position} {swing_player.pro_team}, ±{swing_sd:.1f} pts)"
        )
    if stakes is not None:
        lines.append(
            f"Stakes: win -> {stakes.playoff_odds_if_win:.0f}% playoff odds, "
            f"loss -> {stakes.playoff_odds_if_loss:.0f}% ({stakes.leverage:.0f}-pt swing)"
        )
    return "\n".join(lines)


def render_matchup_result(
    *,
    week: int,
    my_team: Team,
    opp_team: Team,
    my_score: float,
    opp_score: float,
    my_projected: float | None,
    opp_projected: float | None,
) -> str:
    """``analyze_matchup`` for a completed week: the final, and what was projected."""
    result = "won" if my_score > opp_score else "lost" if my_score < opp_score else "tied"
    line = (
        f"Week {week} final: {my_team.name} {my_score:.1f} vs {opp_team.name} "
        f"{opp_score:.1f} ({result})"
    )
    if my_projected is not None and opp_projected is not None:
        line += f"\nProjected at the time: {my_projected:.1f} vs {opp_projected:.1f}"
    return line


# ---------------------------------------------------------------------------
# find_trades
# ---------------------------------------------------------------------------


def render_trade_line(
    partner_name: str,
    partner_team_id: int,
    give: Sequence[Player],
    get: Sequence[Player],
    my_value_delta: float,
    partner_value_delta: float,
    my_odds_delta: float | None,
    rationale: str,
    market: tuple[float, float] | None = None,
) -> str:
    """One trade, both sides' gains in the same unit (points). Win-probability is real signal
    too, so it stays, as a secondary figure, not the only one shown for my side. ``market`` is
    ``(given, received)`` in market value, so a reader sees why an offer is or is not realistic.

    ``partner_team_id`` is printed alongside the name so a suggested offer can be handed
    straight to ``evaluate_trade`` (which takes ``partner_team_id``, not a name) without a
    separate ``league_standings`` lookup to find it.
    """
    give_names = ", ".join(short_name(p.name, 14) for p in give)
    get_names = ", ".join(short_name(p.name, 14) for p in get)
    odds = f", {my_odds_delta:+.1f}% odds" if my_odds_delta is not None else ""
    value = f" — mkt {market[0]:,.0f} vs {market[1]:,.0f}" if market is not None else ""
    return (
        f"{partner_name} (id {partner_team_id}): give {give_names} / get {get_names} — "
        f"me {my_value_delta:+.1f}pts, them {partner_value_delta:+.1f}pts{odds}{value} — "
        f"{rationale}"
    )


def render_trades(lines: Sequence[str], notes: Sequence[str] = ()) -> str:
    body = "\n".join(lines) if lines else "No trades found that would help both sides right now."
    return "\n".join([body, *notes])


def deadline_note(deadline: datetime | None, at: datetime) -> str | None:
    """``"Trade deadline in 48 days (Dec 2)."`` or ``"Trade deadline passed (Dec 2)."``"""
    if deadline is None:
        return None
    day = f"{deadline:%b} {deadline.day}"
    if deadline <= at:
        return f"Trade deadline passed ({day})."
    days = (deadline - at).days
    when = "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}"
    return f"Trade deadline {when} ({day})."


# ---------------------------------------------------------------------------
# player_report / compare_players
# ---------------------------------------------------------------------------


def _status_line(player: Player, points: float | None, week: int) -> str:
    """A ``Status:`` line that is always present, using the same signals as ``status_flag``,
    with what the injury is where a source reports it."""
    detail = player.market.injury_detail if player.market else None
    what = f" ({detail})" if detail and player.injury_status else ""
    if is_ruled_out(player):
        label = (player.injury_status or "OUT").replace("_", " ").title()
        return f"Status: {label}{what}"
    if player.injury_status and player.injury_status.upper() == "QUESTIONABLE":
        return f"Status: Questionable{what}"
    if player.bye_week == week or points is None:
        return f"Status: Bye (week {week})"
    return "Status: healthy"


_SIGNAL_LABELS = {"buy_low": "buy low", "sell_high": "sell high"}


def render_usage(profile: UsageProfile) -> str:
    """One line on workload and luck: the evidence behind a buy-low or sell-high call."""
    parts = []
    if profile.snap_pct is not None:
        recent = profile.recent_snap_pct
        last = f" (last {recent:.0f}%)" if recent is not None else ""
        parts.append(f"{profile.snap_pct:.0f}% snaps{last}")
    if profile.targets_per_game >= 1.0:
        share = f" ({profile.target_share:.0f}% share)" if profile.target_share else ""
        parts.append(f"{profile.targets_per_game:.1f} tgt/g{share}")
    if profile.carries_per_game >= 1.0 and profile.position != "QB":
        parts.append(f"{profile.carries_per_game:.1f} car/g")
    verdict = _SIGNAL_LABELS.get(profile.signal or "", "")
    luck = f"expected {profile.expected_ppg:.1f} vs actual {profile.actual_ppg:.1f} pts/g" + (
        f" -> {verdict}" if verdict else ""
    )
    return f"Usage ({profile.games} g): " + ", ".join([*parts, luck])


def render_game_line(line: GameLine) -> str:
    return (
        f"Game: {line.pro_team} implied {line.implied_total:.1f} pts "
        f"({line.spread:+.1f} vs {line.opponent}, total {line.total:.1f})"
    )


def second_opinion(points: float | None, alternative: float, *, undecided: bool = True) -> str:
    """The second source's number, and what the gap means. Three cases, because a percentage
    stops meaning anything past a point: ESPN zeroing a game-time decision (``undecided``)
    whom Sleeper expects to play is about *availability*, and is said in words."""
    if expected_out_by_primary(points, alternative):
        if not undecided:
            return f"Second opinion: Sleeper {alternative:.1f}"
        return (
            f"ESPN projects 0 (expects him out); Sleeper {alternative:.1f}. Check final "
            "inactives 90 min before kickoff."
        )
    gap = disagreement(points, alternative)
    if sharp_disagreement(gap):
        note = "; sources disagree sharply, so the range is wider"
    elif gap is not None and gap >= _DISPUTE_THRESHOLD:
        note = f"; sources disagree by {gap * 100:.0f}%, so the range is wider"
    else:
        note = ""
    return f"Second opinion: Sleeper {alternative:.1f}{note}"


def render_player_report(
    player: Player,
    *,
    week: int,
    value_over_replacement: float,
    effective_weeks: float,
    sd: float | None = None,
    alternative: float | None = None,
    usage: UsageProfile | None = None,
    game_line: GameLine | None = None,
    trade_value: float | None = None,
    note: str | None = None,
) -> str:
    points = projected_points(player)
    market = player.market
    out = is_ruled_out(player)
    if points is None:
        on_bye = player.bye_week == week
        rate = f", season rate {player.season_rate:.1f} pts/g" if player.season_rate else ""
        projected = f"Projected: no game this week{' (bye' + rate + ')' if on_bye else ''}"
    else:
        projected = f"Projected: {fmt_points(points)} pts"
        # A range for a player who will not play is a range of zeros; leave it out.
        if sd is not None and not out:
            low, high = floor_ceiling(points, sd)
            projected += f" (range {low:.1f}-{high:.1f}, p10-p90)"
    lines = [f"{player.name} — {player.position} {player.pro_team}, week {week}", projected]
    if note:
        lines.append(note)
    if alternative is not None:
        lines.append(second_opinion(points, alternative, undecided=is_game_time_decision(player)))
    status = _status_line(player, points, week)
    if player.bye_week is not None and player.bye_week > week:
        status += f" · bye week {player.bye_week}"
    lines.append(status)
    if player.outlook or (is_ruled_out(player) and player.season_rate is None):
        back = return_week(player)
        if back is None:
            lines.append("Outlook: no return projected this season")
        else:
            after = [p.points for p in player.outlook if p.week >= back]
            lines.append(
                f"Outlook: back week {back} (Sleeper), {sum(after) / len(after):.1f} pts/g after"
            )
    if value_over_replacement > 0.0:
        lines.append(
            f"Value over replacement: {value_over_replacement:+.1f} pts/wk, "
            f"{value_over_replacement * effective_weeks:+.1f} pts rest of season "
            f"({effective_weeks:.1f} wks, byes and playoff odds counted)"
        )
    else:
        # At or below replacement, a player adds nothing over a season (he would not start),
        # which is a different fact from costing points: multiplying a negative rate by the
        # weeks left printed the latter.
        lines.append(
            f"Value over replacement: {value_over_replacement:+.1f} pts/wk: no rest-of-season "
            "value as things stand"
        )
    if usage is not None:
        lines.append(render_usage(usage))
    if game_line is not None:
        lines.append(render_game_line(game_line))
    pieces = []
    if market is not None and market.percent_owned is not None:
        pieces.append(f"{market.percent_owned:.0f}% owned")
    if market is not None and market.trending_adds is not None:
        pieces.append(f"{market.trending_adds:,} adds (24h)")
    if trade_value is not None:
        pieces.append(f"trade value {trade_value:,.0f} (FantasyCalc)")
    if pieces:
        lines.append("Market: " + ", ".join(pieces))
    return "\n".join(lines)


def start_sit_sentence(starts: Sequence[tuple[Player, str]], sits: Sequence[Player]) -> str:
    """``"Start D. Swift (RB), G. Kittle (TE); sit S. Barkley."``"""
    parts = []
    if starts:
        parts.append(
            "Start " + ", ".join(f"{short_name(p.name)} ({slot_label(slot)})" for p, slot in starts)
        )
    if sits:
        parts.append("sit " + ", ".join(short_name(p.name) for p in sits))
    if not parts:
        return "None of them makes your lineup."
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def render_compare_players(
    recommendation: str,
    rows: Sequence[tuple[Player, float, float]],
    sds: dict[int, float] | None = None,
    game_lines: dict[str, GameLine] | None = None,
    note: str | None = None,
) -> str:
    """``rows`` are ``(player, projected_points, value_over_replacement)``. FLOOR and CEIL are
    the 10th and 90th percentile outcomes; IMPL is the player's team's implied points from the
    betting line."""
    headers = ("PLAYER", "POS", "TM", "PROJ", "FLOOR", "CEIL", "IMPL", "VOR")
    widths = (16, 4, 3, 5, 5, 5, 5, 6)
    table_rows = []
    for p, points, vor in rows:
        sd = (sds or {}).get(p.player_id)
        low, high = floor_ceiling(points, sd) if sd is not None else (None, None)
        line = (game_lines or {}).get(p.pro_team)
        table_rows.append(
            [
                short_name(p.name),
                p.position,
                p.pro_team,
                fmt_points(points),
                fmt_points(low),
                fmt_points(high),
                "" if line is None else f"{line.implied_total:.1f}",
                f"{vor:+.1f}",
            ]
        )
    head = recommendation if note is None else f"{recommendation}\n{note}"
    return f"{head}\n{render_table(headers, widths, table_rows)}"


# ---------------------------------------------------------------------------
# roster_planner
# ---------------------------------------------------------------------------


def _drop_reason(candidate: DropCandidate) -> str:
    weeks = candidate.weeks_started
    if not weeks:
        return "never starts"
    covered: dict[int, str] = {}
    for week, starter in candidate.covers:
        covered.setdefault(week, short_name(starter.name))
    if covered and len(covered) == len(weeks):
        cover = ", ".join(f"{name}'s wk-{week} bye" for week, name in sorted(covered.items()))
        return f"only starts to cover {cover}"
    if len(weeks) <= 3:
        return "only starts wk " + ", ".join(str(w) for w in weeks)
    return f"starts {len(weeks)} of the remaining weeks"


def render_roster_plan(
    byes: Sequence[ByeWeek], drops: Sequence[DropCandidate], *, typical: float | None
) -> str:
    """``roster_planner``: the weeks byes cost the lineup, and the bench players the season
    would miss least."""
    lines = []
    if byes:
        base = f" (typical week {typical:.1f})" if typical is not None else ""
        lines.append(f"Bye weeks ahead{base}:")
        for bye in byes:
            names = ", ".join(short_name(p.name) for p in bye.missing)
            lines.append(f"  Wk {bye.week}: {names} out -> {bye.projected:.1f} ({-bye.gap:+.1f})")
    else:
        lines.append("Bye weeks ahead: none cost a regular starter.")
    if drops:
        lines.append("Drop candidates (rest-of-season cost, cheapest first):")
        lines.append(
            render_table(
                ("PLAYER", "POS", "TM", "ROS", "WHY"),
                (16, 4, 3, 6, 44),
                [
                    [
                        short_name(c.player.name),
                        c.player.position,
                        c.player.pro_team,
                        f"{-c.season_value:+.1f}" if c.season_value >= 0.05 else "0.0",
                        _drop_reason(c),
                    ]
                    for c in drops
                ],
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# league_activity
# ---------------------------------------------------------------------------


def render_activity(
    transactions: Sequence[Transaction], team_names: Mapping[int, str], *, limit: int
) -> str:
    """``league_activity``: one line per team per moment, newest first."""
    groups: dict[tuple[datetime, int | None], list[Transaction]] = {}
    for transaction in transactions:
        groups.setdefault((transaction.when, transaction.team_id), []).append(transaction)
    lines = []
    for (when, team_id), moves in list(groups.items())[:limit]:
        team = team_label(team_names.get(team_id or -1, "?"), 18) if team_id is not None else "?"
        actions = ", ".join(f"{m.action} {m.player_name}" for m in moves)
        lines.append(f"{when:%b} {when.day:<2} {team} (id {team_id}): {actions}")
    return "\n".join(lines) if lines else "No recent activity."


# ---------------------------------------------------------------------------
# power_rankings
# ---------------------------------------------------------------------------

_POWER_HEADERS = ("RK", "ID", "TEAM", "W-L", "ALLPLAY", "LUCK", "PPG", "LINEUP%", "BENCH")
_POWER_WIDTHS = (3, 3, 16, 5, 7, 5, 6, 7, 5)


def render_power_rankings(reports: Sequence[ManagerReport], weeks: int) -> str:
    """``power_rankings``: teams by all-play record, with the luck gap and lineup accuracy.

    ``LUCK`` is actual wins minus all-play expected wins. ``LINEUP%`` is projected points
    started over the best lineup available by the projections of the day: decisions, not luck.
    ``BENCH`` is points per game a perfect-hindsight lineup would have added, which is mostly
    luck and is shown as context only."""
    rows = []
    notes = []
    for rank, report in enumerate(reports, start=1):
        losses = report.games - report.wins
        record = f"{report.wins:g}-{losses:g}"
        rows.append(
            [
                str(rank),
                str(report.team_id),
                team_label(report.name),
                record,
                f"{report.all_play_wins}-{report.all_play_losses}",
                f"{report.luck:+.1f}",
                f"{report.points_per_game:.1f}",
                fmt_pct(report.lineup_accuracy, decimals=0),
                fmt_points(report.bench_points_per_game),
            ]
        )
        if report.tags:
            notes.append(f"{report.name}: {', '.join(report.tags)}")
    title = f"Power rankings after {weeks} week{'s' if weeks != 1 else ''} (by all-play record):"
    body = render_table(_POWER_HEADERS, _POWER_WIDTHS, rows)
    return "\n".join([title, body, *notes])


# ---------------------------------------------------------------------------
# buy_low_sell_high
# ---------------------------------------------------------------------------

_LUCK_HEADERS = ("PLAYER", "POS", "TM", "OWNER", "G", "SNAP", "XPPG", "PPG", "LUCK")
_LUCK_WIDTHS = (16, 4, 3, 14, 2, 4, 5, 5, 5)


def _luck_row(profile: UsageProfile, owner: str) -> list[str]:
    return [
        short_name(profile.name),
        profile.position,
        profile.pro_team,
        team_label(owner, 14),
        str(profile.games),
        "" if profile.snap_pct is None else f"{profile.snap_pct:.0f}%",
        f"{profile.expected_ppg:.1f}",
        f"{profile.actual_ppg:.1f}",
        f"{profile.luck_ppg:+.1f}",
    ]


def render_buy_low_sell_high(
    sections: Sequence[tuple[str, Sequence[tuple[UsageProfile, str]]]],
    *,
    labelled: bool,
    min_games: int,
) -> str:
    """``buy_low_sell_high``. XPPG is expected points per game from volume alone; LUCK is
    actual minus expected. ``labelled`` is ``False`` before any player has ``min_games`` games,
    when the tables are a watch list rather than a verdict."""
    out = []
    if not labelled:
        out.append(
            f"Early season: labels need {min_games}+ games, so these are the biggest gaps to "
            "watch, not yet calls."
        )
    for title, rows in sections:
        out.append(f"{title}:")
        if not rows:
            out.append("  none")
            continue
        out.append(render_table(_LUCK_HEADERS, _LUCK_WIDTHS, [_luck_row(p, o) for p, o in rows]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# projection_accuracy
# ---------------------------------------------------------------------------

_ACCURACY_HEADERS = ("POS", "N", "ESPN", "SLEEPER", "AVG", "BIAS")
_ACCURACY_WIDTHS = (4, 5, 5, 7, 5, 6)


def render_projection_accuracy(
    rows: Sequence[SourceAccuracy], weeks: int, disputes: Sequence[str] = ()
) -> str:
    """``projection_accuracy``: mean absolute error per source, on identical player-weeks where
    both sources exist, plus ESPN's bias (actual minus projected; negative = over-projects)."""
    table_rows = []
    for row in rows:
        espn = row.primary_mae_matched if row.primary_mae_matched is not None else row.primary_mae
        table_rows.append(
            [
                row.position,
                str(row.secondary_n or row.n),
                f"{espn:.2f}",
                "" if row.secondary_mae is None else f"{row.secondary_mae:.2f}",
                "" if row.blend_mae is None else f"{row.blend_mae:.2f}",
                f"{row.primary_bias:+.2f}",
            ]
        )
    lines = [
        f"Projection error in this league, weeks 1-{weeks} (mean absolute error, pts):",
        render_table(_ACCURACY_HEADERS, _ACCURACY_WIDTHS, table_rows),
    ]
    overall = next((row for row in rows if row.position == "ALL"), None)
    if overall is not None and overall.secondary_mae is not None:
        espn = overall.primary_mae_matched or overall.primary_mae
        better = "ESPN" if espn < overall.secondary_mae else "Sleeper"
        margin = abs(espn - overall.secondary_mae)
        lines.append(
            f"{better} is closer by {margin:.2f} pts/player-week. Differences under ~0.3 are "
            "noise at this sample size; where the sources disagree, trust both less."
        )
    if disputes:
        lines.append("Biggest disagreements on your roster this week: " + "; ".join(disputes))
    return "\n".join(lines)

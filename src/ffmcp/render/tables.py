"""Fixed-width text tables: no JSON punctuation, no repeated keys, headers abbreviated once
(defined in full at ``ffmcp://glossary``), null columns dropped rather than emitted as ``-``,
points rounded to one decimal.

Every renderer here takes domain objects and returns ``str`` for a tool declared
``structured_output=False``. Nothing in this module does I/O or arithmetic beyond formatting;
the numbers it prints were computed in ``domain/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ffmcp.domain.calibration import SourceAccuracy
from ffmcp.domain.league_intel import ManagerReport
from ffmcp.domain.models import GameLine, LeagueState, MarketSignal, Player, Team
from ffmcp.domain.optimizer import is_ruled_out, projected_points
from ffmcp.domain.simulate import SeasonOutcome, WeekStakes
from ffmcp.domain.usage import UsageProfile
from ffmcp.domain.valuation import remaining_strength_of_schedule
from ffmcp.domain.variance import disagreement, floor_ceiling

Detail = Literal["compact", "standard", "full"]

_TRUNCATION_ELLIPSIS = "…"


def fmt_points(points: float | None) -> str:
    """One decimal place. The extra digits a raw float carries are noise a reader could
    mistake for precision the projection does not have."""
    return "" if points is None else f"{points:.1f}"


def fmt_pct(value: float | None, *, decimals: int = 1) -> str:
    return "" if value is None else f"{value:.{decimals}f}%"


def short_name(name: str, width: int = 15) -> str:
    """``"Josh Allen"`` -> ``"J. Allen"``. Truncated with an ellipsis if still too wide.

    A team defense (``"Commanders D/ST"``) keeps its first word: initialising it would produce
    ``"C. D/ST"``, which identifies nothing. Defense names are short enough to fit anyway.
    """
    parts = name.split()
    initialisable = len(parts) > 1 and parts[-1] != "D/ST"
    candidate = f"{parts[0][0]}. {' '.join(parts[1:])}" if initialisable else name
    if len(candidate) <= width:
        return candidate
    return candidate[: max(1, width - 1)] + _TRUNCATION_ELLIPSIS


def status_flag(player: Player, points: float | None) -> str:
    """A one-to-two letter flag: ruled out, questionable, or on bye. Blank when none apply.

    "Bye" is inferred the same way ``domain.optimizer`` infers it: no projection and not
    injured. There is no separate bye-week field on ``Player`` (docs/architecture.md's models
    don't carry an NFL schedule) so this is the only signal available.
    """
    if is_ruled_out(player):
        status = (player.injury_status or "O")[:1]
        return status
    if player.injury_status and player.injury_status.upper() == "QUESTIONABLE":
        return "Q"
    if points is None:
        return "BYE"
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


def slot_label(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot)


# ---------------------------------------------------------------------------
# get_my_team
# ---------------------------------------------------------------------------

_ROSTER_HEADERS: dict[Detail, tuple[str, ...]] = {
    "compact": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "ST"),
    "standard": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "LIVE", "ST", "OWN%"),
    "full": ("SLOT", "PLAYER", "POS", "TM", "PROJ", "LIVE", "ST", "OWN%", "STRT%", "TREND"),
}
_ROSTER_WIDTHS: dict[Detail, tuple[int, ...]] = {
    "compact": (5, 16, 4, 3, 5, 3),
    "standard": (5, 16, 4, 3, 5, 5, 3, 5),
    "full": (5, 16, 4, 3, 5, 5, 3, 5, 6, 6),
}


def _roster_row(slot: str, player: Player | None, detail: Detail) -> list[str]:
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
    row.append(status_flag(player, points))
    if detail in ("standard", "full"):
        row.append(fmt_pct(market.percent_owned if market else None, decimals=0))
    if detail == "full":
        row.append(fmt_pct(market.percent_started if market else None, decimals=0))
        row.append("" if not market or market.trending_adds is None else str(market.trending_adds))
    return row


def render_injury_summary(players: Sequence[Player]) -> str:
    """``"Injuries: none"``, or one flag per player who is out, doubtful, questionable or on
    IR/suspension, using the same ``is_ruled_out``/``status_flag`` logic as the ``ST`` column,
    so the two can never disagree. A bye is not an injury and is left out: it already has its
    own column."""
    flags = [
        f"{short_name(player.name)} ({flag})"
        for player in players
        if (flag := status_flag(player, projected_points(player))) and flag != "BYE"
    ]
    return "Injuries: none" if not flags else "Injuries: " + ", ".join(flags)


def render_roster(team: Team, week: int, detail: Detail) -> str:
    """``get_my_team``: the team's current lineup, bench and IR as ESPN has them set, not the
    optimizer's suggestion. Use ``optimize_lineup`` for that."""
    headers = _ROSTER_HEADERS[detail]
    widths = _ROSTER_WIDTHS[detail]
    rows = [_roster_row(slot.slot, slot.player, detail) for slot in team.roster.slots]
    rows.extend(_roster_row("BE", player, detail) for player in team.roster.bench)
    rows.extend(_roster_row("IR", player, detail) for player in team.roster.ir)
    title = f"Week {week} — {team.name} ({team.wins}-{team.losses}-{team.ties})"
    roster_players = [
        *(slot.player for slot in team.roster.slots if slot.player is not None),
        *team.roster.bench,
        *team.roster.ir,
    ]
    injuries = render_injury_summary(roster_players)
    return f"{title}\n{injuries}\n{render_table(headers, widths, rows)}"


# ---------------------------------------------------------------------------
# find_waiver_targets
# ---------------------------------------------------------------------------

_WAIVER_HEADERS: dict[Detail, tuple[str, ...]] = {
    "compact": ("PLAYER", "POS", "TM", "PROJ", "VAL", "ROS", "DROP"),
    "standard": ("PLAYER", "POS", "TM", "PROJ", "VAL", "ROS", "DROP", "TREND"),
    "full": ("PLAYER", "POS", "TM", "PROJ", "VAL", "ROS", "DROP", "TREND", "OWN%"),
}
_WAIVER_WIDTHS: dict[Detail, tuple[int, ...]] = {
    "compact": (16, 4, 3, 5, 6, 6, 16),
    "standard": (16, 4, 3, 5, 6, 6, 16, 6),
    "full": (16, 4, 3, 5, 6, 6, 16, 6, 5),
}


class WaiverTarget:
    """One ranked pickup: the add, its marginal value to this roster this week and over the
    rest of the season, and who to drop for it."""

    __slots__ = ("drop", "marginal_value", "player", "season_value")

    def __init__(
        self,
        player: Player,
        marginal_value: float,
        drop: Player | None,
        season_value: float | None = None,
    ) -> None:
        self.player = player
        self.marginal_value = marginal_value
        self.drop = drop
        self.season_value = season_value


def _waiver_row(
    target: WaiverTarget, detail: Detail, *, no_drop_label: str = "(no drop needed)"
) -> list[str]:
    player = target.player
    row = [
        short_name(player.name),
        player.position,
        player.pro_team,
        fmt_points(projected_points(player)),
        f"{target.marginal_value:+.1f}",
        "" if target.season_value is None else f"{target.season_value:+.0f}",
        short_name(target.drop.name, 16) if target.drop is not None else no_drop_label,
    ]
    if detail in ("standard", "full"):
        trending = player.market.trending_adds if player.market else None
        row.append("" if trending is None else str(trending))
    if detail == "full":
        owned = player.market.percent_owned if player.market else None
        row.append(fmt_pct(owned, decimals=0))
    return row


def render_waiver_targets(
    targets: Sequence[WaiverTarget],
    detail: Detail,
    *,
    limit: int,
    total_considered: int,
    near_misses: Sequence[WaiverTarget] = (),
    handcuffs: Sequence[str] = (),
) -> str:
    """``find_waiver_targets``: ranked by marginal value, drop always named, because a pickup
    recommendation with no drop is useless advice.

    When nothing clears the positive-value bar, ``near_misses`` are shown: the top few
    candidates by raw (negative) marginal value, so a manager can tell "the wire is genuinely
    empty" from "everything on it scored -0.1", neither of which a bare "no targets" line
    can distinguish.
    """
    headers = _WAIVER_HEADERS[detail]
    widths = _WAIVER_WIDTHS[detail]
    footer = "" if not handcuffs else "\nHandcuffs: " + "; ".join(handcuffs)
    if not targets:
        if not near_misses:
            return "No waiver targets clear the bar this week." + footer
        rows = [_waiver_row(target, detail, no_drop_label="(no upgrade)") for target in near_misses]
        table = render_table(headers, widths, rows)
        return f"Closest misses (would not improve your lineup):\n{table}{footer}"
    rows = [_waiver_row(target, detail) for target in targets[:limit]]
    table = render_table(headers, widths, rows)
    if total_considered > limit:
        table += f"\n… {total_considered - limit} more (raise limit to see)"
    return table + footer


# ---------------------------------------------------------------------------
# league_standings
# ---------------------------------------------------------------------------

_STANDINGS_HEADERS = ("RK", "ID", "TEAM", "W-L", "PF", "PA", "PLAYOFF%", "SOS")
_STANDINGS_WIDTHS = (3, 3, 14, 6, 7, 7, 9, 6)


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


def render_standings(state: LeagueState, detail: Detail) -> str:
    """``league_standings``. ``PLAYOFF%`` stands in for ESPN's "power ranking": it is the one
    forward-looking team-strength number this server's domain models actually carry. There is
    no ``power_score`` field; that column, as originally specified, was dropped rather than
    faked.

    ``ID`` is ``Team.team_id``, distinct from ``RK`` (rank by record). It is the only place a
    caller can read a team's id from prose rather than guess it from rank position: tools like
    ``evaluate_trade`` take a numeric ``partner_team_id``, and rank and id agree only when the
    standings happen to be sorted by id.
    """
    teams = sorted(state.teams, key=lambda team: team.standing)
    rows = []
    for team in teams:
        sos = remaining_strength_of_schedule(team.team_id, state)
        rows.append(
            [
                str(team.standing),
                str(team.team_id),
                team.name[:14],
                f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else ""),
                f"{team.points_for:.1f}",
                f"{team.points_against:.1f}",
                fmt_pct(team.playoff_pct),
                fmt_pct(sos) if sos is not None else "",
            ]
        )
    table = render_table(_STANDINGS_HEADERS, _STANDINGS_WIDTHS, rows)
    note = _week_in_progress_note(state)
    return table if note is None else f"{table}\n{note}"


# ---------------------------------------------------------------------------
# simulate_season
# ---------------------------------------------------------------------------

_SEASON_HEADERS = ("TEAM", "REC", "MEANW", "PLAYOFF%", "TITLE%")
_SEASON_WIDTHS = (16, 7, 6, 9, 7)


def render_season_outcome(outcome: SeasonOutcome, *, highlight_team_id: int | None = None) -> str:
    ordered = sorted(outcome.teams, key=lambda t: (t.team_id != highlight_team_id, -t.playoff_odds))
    rows = []
    for team in ordered:
        record = f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else "")
        marker = "*" if team.team_id == highlight_team_id else ""
        rows.append(
            [
                marker + short_name(team.name, 16 - len(marker)),
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
    return f"{header}\n{render_table(_SEASON_HEADERS, _SEASON_WIDTHS, rows)}"


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
) -> str:
    """``analyze_matchup``: totals, win probability, the biggest positional edges, and the
    single highest-variance swing player.

    ``my_live_score``/``opp_live_score`` are this week's actual in-game score so far. Omitted
    when both are still 0.0, which means the games haven't kicked off yet and there is nothing
    live to show.
    """
    lines = []
    if my_live_score > 0.0 or opp_live_score > 0.0:
        lines.append(
            f"Live: {my_team.name} {my_live_score:.1f} vs {opp_team.name} {opp_live_score:.1f}"
        )
    lines.append(
        f"Week {week} (proj): {my_team.name} {my_projected:.1f} vs {opp_team.name} "
        f"{opp_projected:.1f} — {win_probability:.0f}% win prob"
    )
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
) -> str:
    """One trade, both sides' gains in the same unit (points). Win-probability is real signal
    too, so it stays, as a secondary figure, not the only one shown for my side.

    ``partner_team_id`` is printed alongside the name so a suggested offer can be handed
    straight to ``evaluate_trade`` (which takes ``partner_team_id``, not a name) without a
    separate ``league_standings`` lookup to find it.
    """
    give_names = ", ".join(short_name(p.name, 14) for p in give)
    get_names = ", ".join(short_name(p.name, 14) for p in get)
    odds = f", {my_odds_delta:+.1f}% odds" if my_odds_delta is not None else ""
    return (
        f"{partner_name} (id {partner_team_id}): give {give_names} / get {get_names} — "
        f"me +{my_value_delta:.1f}pts, them +{partner_value_delta:.1f}pts{odds} — {rationale}"
    )


def render_trades(lines: Sequence[str]) -> str:
    if not lines:
        return "No trades found that would help both sides right now."
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# player_report / compare_players
# ---------------------------------------------------------------------------


def _status_line(player: Player, points: float | None) -> str:
    """A ``Status:`` line that is always present, using the same ``is_ruled_out`` signal
    ``status_flag`` does."""
    if is_ruled_out(player):
        label = (player.injury_status or "OUT").replace("_", " ").title()
        return f"Status: {label}"
    if player.injury_status and player.injury_status.upper() == "QUESTIONABLE":
        return "Status: Questionable"
    if points is None:
        return "Status: Bye"
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
) -> str:
    points = projected_points(player)
    market = player.market
    rest_of_season = value_over_replacement * effective_weeks
    projected = f"Projected: {fmt_points(points)} pts"
    if points is not None and sd is not None:
        low, high = floor_ceiling(points, sd)
        projected += f" (range {low:.1f}-{high:.1f}, p10-p90)"
    if points is None and player.season_rate is not None and not is_ruled_out(player):
        projected = f"Projected: no game this week (season rate {player.season_rate:.1f} pts/g)"
    lines = [f"{player.name} — {player.position} {player.pro_team}, week {week}", projected]
    if alternative is not None:
        gap = disagreement(points, alternative)
        note = (
            f"; sources disagree by {gap * 100:.0f}%, so the range is wider"
            if gap is not None and gap >= 0.25
            else ""
        )
        lines.append(f"Second opinion: Sleeper {alternative:.1f}{note}")
    status = _status_line(player, points)
    if player.bye_week is not None and player.bye_week != week:
        status += f" · bye week {player.bye_week}"
    lines.append(status)
    lines.append(
        f"Value over replacement: {value_over_replacement:+.1f} pts/wk, "
        f"{rest_of_season:+.1f} pts rest of season ({effective_weeks:.1f} wks, byes and "
        "playoff odds counted)"
    )
    if usage is not None:
        lines.append(render_usage(usage))
    if game_line is not None:
        lines.append(render_game_line(game_line))
    if market is not None:
        pieces = []
        if market.percent_owned is not None:
            pieces.append(f"{market.percent_owned:.0f}% owned")
        if market.trending_adds is not None:
            pieces.append(f"{market.trending_adds:,} adds (24h)")
        if pieces:
            lines.append("Market: " + ", ".join(pieces))
    return "\n".join(lines)


def render_compare_players(
    recommendation: str,
    rows: Sequence[tuple[Player, float, float]],
    sds: dict[int, float] | None = None,
    game_lines: dict[str, GameLine] | None = None,
) -> str:
    """``rows`` are ``(player, projected_points, value_over_replacement)``, already ordered
    best-first. The recommendation names ``rows[0]``. FLOOR and CEIL are the 10th and 90th
    percentile outcomes; IMPL is the player's team's implied points from the betting line."""
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
    return f"{recommendation}\n{render_table(headers, widths, table_rows)}"


# ---------------------------------------------------------------------------
# power_rankings
# ---------------------------------------------------------------------------

_POWER_HEADERS = ("RK", "ID", "TEAM", "W-L", "ALLPLAY", "LUCK", "PPG", "LINEUP%", "BENCH")
_POWER_WIDTHS = (3, 3, 14, 5, 7, 5, 6, 7, 5)


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
                report.name[:14],
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
        owner[:14],
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

"""Is ESPN's projection any good in *this* league, and is a second source any better?

Scores every completed, played player-week in the league's history against what actually
happened, for ESPN's projection, a second source's (Sleeper's), and their average. Reported by
position, as mean absolute error and bias (actual minus projected: negative means the source
over-projects).

What this is for, and what it is not for: it tells a manager how much weight a projection
deserves and where the sources part ways. It deliberately does *not* feed a per-player bias
correction back into projections. That was tested against a full season
(``scripts/calibrate.py``), and correcting a player by his own recent misses made the next
weeks' predictions worse at every shrinkage strength tried: last month's miss is noise, not a
trait. The honest use of this table is the one it is put to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ffmcp.domain.models import Frozen, PlayerWeek

POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "D/ST")


class SourceAccuracy(Frozen):
    position: str
    """A position, or ``"ALL"``."""
    n: int
    primary_mae: float
    primary_bias: float
    secondary_n: int = 0
    """Player-weeks where the second source also had a projection. The comparison columns
    below cover only these, so the two sources are always measured on identical games."""
    primary_mae_matched: float | None = None
    secondary_mae: float | None = None
    blend_mae: float | None = None


def source_accuracy(
    history: Sequence[PlayerWeek],
    secondary: Mapping[tuple[int, int], float] | None = None,
) -> list[SourceAccuracy]:
    """One row per position with data, then ``"ALL"``. ``secondary`` is keyed
    ``(week, player_id)``."""
    rows = [
        row for row in history if row.played and row.projected is not None and row.projected > 0.0
    ]
    result = []
    for position in (*POSITIONS, "ALL"):
        subset = [row for row in rows if position == "ALL" or row.position == position]
        if not subset:
            continue
        errors = [row.actual - (row.projected or 0.0) for row in subset]
        matched = [
            (row, secondary[(row.week, row.player_id)])
            for row in subset
            if secondary is not None and (row.week, row.player_id) in secondary
        ]
        entry = SourceAccuracy(
            position=position,
            n=len(subset),
            primary_mae=sum(abs(e) for e in errors) / len(errors),
            primary_bias=sum(errors) / len(errors),
        )
        if matched:
            m = len(matched)
            entry = entry.model_copy(
                update={
                    "secondary_n": m,
                    "primary_mae_matched": sum(
                        abs(r.actual - (r.projected or 0.0)) for r, _ in matched
                    )
                    / m,
                    "secondary_mae": sum(abs(r.actual - s) for r, s in matched) / m,
                    "blend_mae": sum(
                        abs(r.actual - ((r.projected or 0.0) + s) / 2) for r, s in matched
                    )
                    / m,
                }
            )
        result.append(entry)
    return result

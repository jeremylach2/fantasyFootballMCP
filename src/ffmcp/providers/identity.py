"""Resolve player identity between ESPN and Sleeper.

The two platforms disagree about IDs, so this matches on a normalized ``(name, position,
pro_team)`` key, with a small committed override table for the handful of players that don't
normalize to the same key on both sides (suffix handling, etc.; see docs/architecture.md §3). A
miss is counted in ``unresolved``, never raised: the caller degrades to "no market signal for
this player" rather than failing the tool.

Team defenses are the one case that never round-trips through that name key at all, for every
league, not just a handful of players: ESPN calls the position ``D/ST`` and names the player
``"<Nickname> D/ST"`` (e.g. ``"Falcons D/ST"``); Sleeper calls the position ``DEF`` and names it
the full team name (``"Atlanta Falcons"``). Neither the position token nor the name token could
ever agree. Both providers do agree on the team's own pro-team abbreviation, though (Sleeper's
``player_id`` for a defense literally *is* that abbreviation), so defenses are matched on that
instead, via ``_TEAM_ABBR_ALIASES`` for the one abbreviation the two disagree on (Washington:
ESPN's ``WSH`` vs Sleeper's ``WAS``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_STRIP_RE = re.compile(r"[^a-z0-9 ]")

IdentityKey = tuple[str, str, str]

_DEFENSE_POSITIONS = frozenset({"D/ST", "DEF", "DST"})

_TEAM_ABBR_ALIASES = {"WSH": "WAS"}
"""The one pro-team abbreviation ESPN and Sleeper spell differently. Keyed on ESPN's spelling,
since ``resolve()`` is always called with ESPN's ``pro_team``."""

# Committed overrides for ESPN player ids that don't normalize to their Sleeper counterpart.
# Populate as `scripts/record_fixtures.py` surfaces real misses against a live league.
OVERRIDES: dict[int, str] = {}


def normalize(name: str, position: str, pro_team: str) -> IdentityKey:
    tokens = _STRIP_RE.sub("", name.lower()).split()
    tokens = [tok for tok in tokens if tok not in _SUFFIXES]
    return (" ".join(tokens), position.upper(), pro_team.upper())


def _canonical_team(pro_team: str) -> str:
    abbr = pro_team.upper()
    return _TEAM_ABBR_ALIASES.get(abbr, abbr)


@dataclass
class IdentityIndex:
    """Built once from a Sleeper ``id -> {name, pos, team}`` index."""

    by_key: dict[IdentityKey, str] = field(default_factory=dict)
    by_defense_team: dict[str, str] = field(default_factory=dict)
    unresolved: int = 0

    @classmethod
    def build(cls, sleeper_players: dict[str, dict[str, str]]) -> IdentityIndex:
        by_key: dict[IdentityKey, str] = {}
        by_defense_team: dict[str, str] = {}
        for sleeper_id, info in sleeper_players.items():
            if info["pos"].upper() in _DEFENSE_POSITIONS:
                by_defense_team.setdefault(_canonical_team(info["team"]), sleeper_id)
                continue
            key = normalize(info["name"], info["pos"], info["team"])
            by_key.setdefault(key, sleeper_id)
        return cls(by_key=by_key, by_defense_team=by_defense_team)

    def resolve(self, espn_player_id: int, name: str, position: str, pro_team: str) -> str | None:
        override = OVERRIDES.get(espn_player_id)
        if override is not None:
            return override
        if position.upper() in _DEFENSE_POSITIONS:
            sleeper_id = self.by_defense_team.get(_canonical_team(pro_team))
        else:
            sleeper_id = self.by_key.get(normalize(name, position, pro_team))
        if sleeper_id is None:
            self.unresolved += 1
        return sleeper_id

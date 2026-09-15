"""Resolve player identity between ESPN and Sleeper.

The two platforms disagree about IDs, so this matches on a normalized ``(name, position,
pro_team)`` key, with a small committed override table for the handful of players that don't
normalize to the same key on both sides (defense/special-teams naming, suffix handling, etc.;
see docs/architecture.md §3). A miss is counted in ``unresolved``, never raised: the caller
degrades to "no market signal for this player" rather than failing the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_STRIP_RE = re.compile(r"[^a-z0-9 ]")

IdentityKey = tuple[str, str, str]

# Committed overrides for ESPN player ids that don't normalize to their Sleeper counterpart.
# Populate as `scripts/record_fixtures.py` surfaces real misses against a live league.
OVERRIDES: dict[int, str] = {}


def normalize(name: str, position: str, pro_team: str) -> IdentityKey:
    tokens = _STRIP_RE.sub("", name.lower()).split()
    tokens = [tok for tok in tokens if tok not in _SUFFIXES]
    return (" ".join(tokens), position.upper(), pro_team.upper())


@dataclass
class IdentityIndex:
    """Built once from a Sleeper ``id -> {name, pos, team}`` index."""

    by_key: dict[IdentityKey, str] = field(default_factory=dict)
    unresolved: int = 0

    @classmethod
    def build(cls, sleeper_players: dict[str, dict[str, str]]) -> IdentityIndex:
        by_key: dict[IdentityKey, str] = {}
        for sleeper_id, info in sleeper_players.items():
            key = normalize(info["name"], info["pos"], info["team"])
            by_key.setdefault(key, sleeper_id)
        return cls(by_key=by_key)

    def resolve(self, espn_player_id: int, name: str, position: str, pro_team: str) -> str | None:
        override = OVERRIDES.get(espn_player_id)
        if override is not None:
            return override
        sleeper_id = self.by_key.get(normalize(name, position, pro_team))
        if sleeper_id is None:
            self.unresolved += 1
        return sleeper_id

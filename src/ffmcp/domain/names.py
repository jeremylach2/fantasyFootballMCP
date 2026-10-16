"""Finding a player from his name as a person types it.

People type "AJ Brown" for ESPN's "A.J. Brown", "Marvin Harrison" for "Marvin Harrison Jr.",
and misspell the rest. So names are compared after normalizing away case, punctuation and
generational suffixes, and when nothing matches, the closest names are offered back rather
than guessed: a tool that silently picked the wrong Brown would be worse than one that asks.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

from ffmcp.domain.models import Player

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_PUNCTUATION = re.compile("[.'`" + chr(0x2019) + "]")
"""Periods, straight and curly apostrophes, backticks: "A.J." and "D'Andre" as typed."""
_SEPARATORS = re.compile(r"[-\s]+")
SUGGESTION_CUTOFF = 0.75
"""``difflib`` similarity a name needs to be offered as "did you mean". Low enough for a dropped
letter ("Barkly"), high enough that a different player with a similar surname is not."""


def normalize_name(name: str) -> str:
    """``"A.J. Brown"`` and ``"aj brown"`` -> ``"aj brown"``; ``"Kenneth Walker III"`` ->
    ``"kenneth walker"``. Hyphens become spaces, so "Amon-Ra" and "Amon Ra" agree."""
    words = _SEPARATORS.split(_PUNCTUATION.sub("", name.lower()).strip())
    return " ".join(word for word in words if word and word not in _SUFFIXES)


def match_players(pool: Sequence[Player], query: str) -> list[Player]:
    """Every player ``query`` names: exact normalized matches if there are any, otherwise every
    player whose normalized name contains it. Empty on no match, several on ambiguity; the
    caller decides what to do with either."""
    needle = normalize_name(query)
    if not needle:
        return []
    names = [(normalize_name(player.name), player) for player in pool]
    exact = [player for name, player in names if name == needle]
    if exact:
        return exact
    return [player for name, player in names if needle in name]


def suggestions(pool: Sequence[Player], query: str, limit: int = 3) -> list[str]:
    """Up to ``limit`` real names closest to ``query``, best first, for a "did you mean"."""
    needle = normalize_name(query)
    by_name: dict[str, str] = {}
    for player in pool:
        key = normalize_name(player.name)
        by_name.setdefault(key, player.name)
        surname = key.rsplit(" ", 1)[-1]
        by_name.setdefault(surname, player.name)
    close = difflib.get_close_matches(needle, list(by_name), n=limit * 2, cutoff=SUGGESTION_CUTOFF)
    seen: list[str] = []
    for key in close:
        name = by_name[key]
        if name not in seen:
            seen.append(name)
    return seen[:limit]

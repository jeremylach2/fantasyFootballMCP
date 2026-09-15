"""Error taxonomy and secret redaction. See docs/architecture.md §4.

An error raised here becomes a tool response the model reads and pays tokens for. It must be
short, actionable, and never a stack trace. ``redact()`` is applied at construction so a secret
can never reach the model or a log line through an error message, regardless of call site.
"""

from __future__ import annotations

import re

_SWID_RE = re.compile(
    r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}"
)
_ESPN_S2_RE = re.compile(r"(espn_s2=)[^\s&;\"']+", re.IGNORECASE)
_COOKIE_HEADER_RE = re.compile(r"(?im)^Cookie:.*$")


def redact(text: str) -> str:
    """Strip ``espn_s2``, ``SWID``, and ``Cookie`` headers out of a string."""
    text = _COOKIE_HEADER_RE.sub("Cookie: [REDACTED]", text)
    text = _ESPN_S2_RE.sub(r"\1[REDACTED]", text)
    text = _SWID_RE.sub("[REDACTED]", text)
    return text


class FFMCPError(Exception):
    """Base for errors surfaced to the model. ``str(e)`` is the entire tool-facing message."""


class CredentialsMissing(FFMCPError):
    """No cookies were configured for a private league."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            redact(
                message
                or "This is a private league; set FFMCP_ESPN_S2 and FFMCP_SWID (see .env.example)."
            )
        )


class LeagueNotAccessible(FFMCPError):
    """Wrong league id, or cookies that were rejected or have expired."""

    def __init__(self, reason: str) -> None:
        super().__init__(redact(f"League not accessible: {reason}"))


class UpstreamUnavailable(FFMCPError):
    """ESPN or Sleeper returned a server error, or the request timed out."""

    def __init__(self, source: str, *, stale_served: bool = False) -> None:
        note = " (serving stale cached data)" if stale_served else ""
        super().__init__(redact(f"{source} is temporarily unavailable{note}."))


class PlayerNotFound(FFMCPError):
    """A player name matched zero or more than one player. Offer candidates instead of
    guessing, in either direction."""

    def __init__(self, query: str, candidates: list[str] | None = None) -> None:
        if candidates:
            hint = f" Multiple players match: {', '.join(candidates[:3])}. Be more specific."
        else:
            hint = ""
        super().__init__(redact(f"No single player found matching '{query}'.{hint}"))


class TradePartnerAmbiguous(FFMCPError):
    """``evaluate_trade`` was called without ``partner_team_id`` and the ``get`` names do not
    trace to exactly one other roster."""

    def __init__(self) -> None:
        super().__init__(redact("Could not determine the trade partner; pass partner_team_id."))


class TeamNotConfigured(FFMCPError):
    """Which team is "mine" could not be resolved: no ``FFMCP_TEAM_ID`` and the connected
    client cannot elicit."""

    def __init__(self) -> None:
        super().__init__(
            redact(
                "Which team is yours is not configured. Set FFMCP_TEAM_ID (see .env.example), "
                "or connect with a client that supports elicitation."
            )
        )


class WeekOutOfRange(FFMCPError):
    """The requested week falls outside the season's valid range."""

    def __init__(self, week: int, *, valid_min: int, valid_max: int) -> None:
        super().__init__(
            redact(f"Week {week} is out of range. Valid weeks are {valid_min}-{valid_max}.")
        )

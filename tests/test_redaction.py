"""Secrets must never survive redact(), in any message shape."""

from __future__ import annotations

from ffmcp.errors import CredentialsMissing, LeagueNotAccessible, redact

FAKE_ESPN_S2 = "AEA1b2C3d4E5f6G7h8I9j0K%2FL1m2N3o4P5q6R7s8T9u0V1w2X3y4Z5"
FAKE_SWID = "{ABCDEF12-3456-7890-ABCD-EF1234567890}"


def test_redact_strips_espn_s2_and_swid() -> None:
    text = f"request failed with cookies espn_s2={FAKE_ESPN_S2}; SWID={FAKE_SWID}"
    result = redact(text)
    assert FAKE_ESPN_S2 not in result
    assert FAKE_SWID not in result


def test_redact_strips_cookie_header() -> None:
    text = f"Cookie: espn_s2={FAKE_ESPN_S2}; SWID={FAKE_SWID}\nGET /league HTTP/1.1"
    result = redact(text)
    assert FAKE_ESPN_S2 not in result
    assert FAKE_SWID not in result
    assert "[REDACTED]" in result


def test_error_messages_never_leak_a_swid_passed_in() -> None:
    err = LeagueNotAccessible(f"cookie SWID={FAKE_SWID} was rejected")
    assert FAKE_SWID not in str(err)


def test_credentials_missing_default_message_has_no_secret() -> None:
    err = CredentialsMissing()
    assert FAKE_ESPN_S2 not in str(err)
    assert FAKE_SWID not in str(err)
    assert "FFMCP_ESPN_S2" in str(err)

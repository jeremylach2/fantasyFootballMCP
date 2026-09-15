"""Token counting: two backends.

Preferred, when available: Anthropic's token-counting endpoint, used when the optional
``anthropic`` package is installed and ``ANTHROPIC_API_KEY`` is set. Exact for the target model
family, but it is real network I/O, so nothing in this project's test suite may depend on it.

Default, and what CI always uses: a deterministic local estimator, so the budget tests and the
benchmark script are hermetic. The estimate is ``len(text) / _CHARS_PER_TOKEN``, rounded to the
nearest integer with a floor of 1. ``_CHARS_PER_TOKEN = 4`` is the commonly cited rule of thumb
for English text under BPE-style tokenizers. It has not been calibrated against the real
Anthropic endpoint in this environment: no ``anthropic`` package and no API key were available
when this module was written, so this note states that gap plainly instead of printing a
fabricated error figure.
"""

from __future__ import annotations

import os

_CHARS_PER_TOKEN = 4.0

_MODEL = "claude-sonnet-5"


def estimate_tokens(text: str) -> int:
    """The deterministic local estimator. No network, no API key, always available."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def count_tokens(text: str) -> int:
    """Preferred backend when available, else the local estimator.

    Never raises on a missing package, a missing key, or a network failure. A benchmark or a
    budget test must not go red because an optional dependency is absent.
    """
    exact = _count_tokens_via_api(text)
    return exact if exact is not None else estimate_tokens(text)


def _count_tokens_via_api(text: str) -> int | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        result = client.messages.count_tokens(
            model=_MODEL,
            messages=[{"role": "user", "content": text}],
        )
        return int(result.input_tokens)
    except Exception:
        return None

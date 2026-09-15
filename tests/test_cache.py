"""Hit, miss, expiry, stale-read-after-expiry, corrupt-file recovery."""

from __future__ import annotations

from pathlib import Path

from ffmcp.providers.cache import Cache


async def test_miss_calls_fetch_and_caches(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        return "value"

    assert await cache.get_or_fetch("k", 60, fetch) == "value"
    assert calls == 1


async def test_hit_does_not_call_fetch_again(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        return "value"

    await cache.get_or_fetch("k", 60, fetch)
    await cache.get_or_fetch("k", 60, fetch)
    assert calls == 1


async def test_expiry_triggers_refetch(tmp_path: Path) -> None:
    now = [1000.0]
    cache = Cache(tmp_path, clock=lambda: now[0])
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        return f"value-{calls}"

    assert await cache.get_or_fetch("k", 10, fetch) == "value-1"
    now[0] += 11  # past the 10s ttl
    assert await cache.get_or_fetch("k", 10, fetch) == "value-2"
    assert calls == 2


async def test_stale_read_after_expiry(tmp_path: Path) -> None:
    now = [1000.0]
    cache = Cache(tmp_path, clock=lambda: now[0])

    async def fetch() -> str:
        return "value"

    await cache.get_or_fetch("k", 10, fetch)
    now[0] += 11  # expired, but get_stale ignores expiry

    assert cache.get_stale("k") == "value"


def test_stale_read_of_unknown_key_returns_none(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.get_stale("never-cached") is None


async def test_corrupt_cache_file_is_deleted_and_treated_as_miss(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    path = cache.key_path("k")
    path.write_text("{not valid json", encoding="utf-8")

    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        return "recovered"

    # A fresh Cache instance forces the read to go through disk, not memory.
    fresh = Cache(tmp_path)
    assert await fresh.get_or_fetch("k", 60, fetch) == "recovered"
    assert calls == 1
    assert path.read_text(encoding="utf-8")  # rewritten with a valid entry

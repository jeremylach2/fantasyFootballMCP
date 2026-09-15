"""Two-tier TTL cache: in-memory for the process lifetime, JSON on disk for warm restarts.

Keys are namespaced and versioned (e.g. ``v1:espn:roster:{league}:{season}:{week}``) so a
schema change invalidates cleanly. See docs/architecture.md §3. Only JSON-serializable
values may be cached; callers store plain dicts/lists, not SDK or domain objects.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger("ffmcp.cache")

T = TypeVar("T")


@dataclass
class _Entry:
    value: Any
    expires_at: float


class Cache:
    """``get_or_fetch`` serves a fresh value from memory, then disk, then calls ``fetch``.

    ``get_stale`` is the degraded-path read: it returns the last known value regardless of
    expiry, or ``None`` if the key was never cached, for the caller to serve alongside a note
    that upstream is unavailable (docs/architecture.md §4 rule 3).
    """

    def __init__(self, cache_dir: Path, *, clock: Callable[[], float] = time.time) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._mem: dict[str, _Entry] = {}

    def key_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self._dir / f"{safe}.json"

    async def get_or_fetch(self, key: str, ttl: float, fetch: Callable[[], Awaitable[T]]) -> T:
        now = self._clock()

        mem = self._mem.get(key)
        if mem is not None and mem.expires_at > now:
            return mem.value  # type: ignore[no-any-return]

        disk = self._read_disk(key)
        if disk is not None and disk.expires_at > now:
            self._mem[key] = disk
            return disk.value  # type: ignore[no-any-return]

        value = await fetch()
        self._write(key, value, now + ttl)
        return value

    def get_stale(self, key: str) -> Any | None:
        mem = self._mem.get(key)
        if mem is not None:
            return mem.value
        disk = self._read_disk(key)
        return disk.value if disk is not None else None

    def _write(self, key: str, value: Any, expires_at: float) -> None:
        self._mem[key] = _Entry(value=value, expires_at=expires_at)
        try:
            payload = json.dumps({"value": value, "expires_at": expires_at})
        except TypeError:
            logger.debug("cache value for %s is not JSON-serializable; memory-only", key)
            return
        self.key_path(key).write_text(payload, encoding="utf-8")

    def _read_disk(self, key: str) -> _Entry | None:
        path = self.key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _Entry(value=data["value"], expires_at=data["expires_at"])
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning("corrupt cache file %s; deleting and treating as a miss", path)
            path.unlink(missing_ok=True)
            return None

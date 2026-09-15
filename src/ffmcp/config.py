"""Application settings.

``pydantic-settings``, env prefix ``FFMCP_``, with ``.env`` support. The MCP SDK no longer reads
``.env`` or ``MCP_*`` for us, so this module is the only place configuration is read. See
docs/architecture.md §5 for the full variable table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(ValueError):
    """A single, readable settings misconfiguration. Never a pydantic ``ValidationError``."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FFMCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["live", "demo"] = "live"
    league_id: int | None = None
    season: int | None = None
    team_id: int | None = None
    espn_s2: SecretStr | None = None
    swid: SecretStr | None = None
    auth_token: SecretStr | None = None
    cache_dir: Path = Field(default=Path("~/.cache/ffmcp"))
    sims: int = 10_000

    @field_validator("cache_dir", mode="after")
    @classmethod
    def _expand_cache_dir(cls, value: Path) -> Path:
        return value.expanduser()


def load_settings() -> Settings:
    """Construct and validate ``Settings``, raising one readable line on misconfiguration.

    Deliberately validated outside pydantic's own error path: a ``ValidationError`` raised
    from a validator renders as a multi-line block, which is not acceptable for a message a
    model or a terminal user has to read.
    """
    settings = Settings()
    if settings.mode == "live" and settings.league_id is None:
        raise ConfigurationError(
            "FFMCP_LEAGUE_ID is required when FFMCP_MODE=live (see .env.example)."
        )
    return settings

"""Domain models: frozen, platform-agnostic. No ESPN or Sleeper vocabulary leaks in here.
``providers/`` is responsible for translating upstream shapes into these (docs/architecture.md
§1). Frozen because these are handed around and compared freely between the optimizer, the
simulator and the renderers, and nothing downstream should be able to mutate a shared instance.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

NON_STARTING_SLOTS = frozenset({"BE", "IR"})
"""Roster slots that are not part of the starting lineup. ESPN reports bench and injured
reserve in the same ``position_slot_counts`` map as real slots, so the split has to happen
somewhere. It happens here, once."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Projection(Frozen):
    week: int
    points: float
    source: str = "espn"


class MarketSignal(Frozen):
    percent_owned: float | None = None
    percent_started: float | None = None
    trending_adds: int | None = None


class Player(Frozen):
    player_id: int
    name: str
    position: str
    eligible_slots: tuple[str, ...]
    pro_team: str
    injured: bool = False
    injury_status: str | None = None
    projection: Projection | None = None
    market: MarketSignal | None = None
    live_points: float | None = None
    """Actual points scored so far this week, or ``None`` before the player's game has kicked
    off (including a bye). Distinct from ``projection``, which never updates once the week's
    games start."""


class RosterSlot(Frozen):
    """One starting lineup slot. ``player`` is ``None`` for an empty slot."""

    slot: str
    player: Player | None = None


class Roster(Frozen):
    slots: tuple[RosterSlot, ...]
    bench: tuple[Player, ...] = ()
    ir: tuple[Player, ...] = ()

    @property
    def players(self) -> tuple[Player, ...]:
        """Everyone eligible to be started: current starters plus the bench. IR is excluded:
        an IR player cannot be moved into a lineup without a corresponding roster move, which
        is a write operation this server does not perform."""
        started = tuple(s.player for s in self.slots if s.player is not None)
        return started + self.bench


class Lineup(Frozen):
    """A proposed assignment of players to starting slots. Produced by ``domain.optimizer``;
    ``slots`` preserves the league's declared slot order, and a slot with ``player=None`` is
    one nothing legal was available for."""

    slots: tuple[RosterSlot, ...]
    projected_points: float
    bench: tuple[Player, ...] = ()
    """Startable players this lineup leaves out."""
    unavailable: tuple[Player, ...] = ()
    """Players excluded from consideration entirely: on bye, or ruled out."""

    @property
    def started(self) -> tuple[Player, ...]:
        return tuple(s.player for s in self.slots if s.player is not None)


class Swap(Frozen):
    """One roster move a manager would actually make: sit ``player_out``, start ``player_in``.

    Either side may be ``None`` at the edges (starting into an empty slot, or benching someone
    with no legal replacement). ``render/`` maps this onto a wire-shaped ``Swap``; this one
    keeps whole ``Player`` objects so callers can explain themselves without a second lookup."""

    slot: str
    player_in: Player | None = None
    player_out: Player | None = None
    gain: float = 0.0


class Team(Frozen):
    team_id: int
    name: str
    abbrev: str
    wins: int
    losses: int
    ties: int = 0
    points_for: float
    points_against: float
    standing: int
    playoff_pct: float
    roster: Roster


class Matchup(Frozen):
    week: int
    home_team_id: int | None
    away_team_id: int | None
    home_score: float
    away_score: float
    home_projected: float
    away_projected: float
    is_playoff: bool = False


class LeagueSettings(Frozen):
    league_id: int
    season: int
    team_count: int
    playoff_team_count: int
    reg_season_weeks: int
    slot_counts: dict[str, int]
    scoring_type: str | None = None

    @property
    def starting_slots(self) -> tuple[str, ...]:
        """The starting lineup expanded to one entry per slot, in the league's declared order
        (``("QB", "RB", "RB", "WR", ...)``). This is the optimizer's input, and the declared
        order is load-bearing for the greedy baseline. See ``domain.optimizer``."""
        return tuple(
            slot
            for slot, count in self.slot_counts.items()
            if slot not in NON_STARTING_SLOTS and count > 0
            for _ in range(count)
        )


class LeagueState(Frozen):
    """The full snapshot a provider assembles for a given week."""

    settings: LeagueSettings
    teams: tuple[Team, ...]
    current_week: int
    matchups: tuple[Matchup, ...] = ()

    @property
    def weeks_remaining(self) -> int:
        """Regular-season weeks still to play, including the current one. Season-long value is
        per-week value multiplied by this, so it lives with the state rather than being
        recomputed at each call site."""
        return max(0, self.settings.reg_season_weeks - self.current_week + 1)

    def team(self, team_id: int) -> Team:
        for team in self.teams:
            if team.team_id == team_id:
                return team
        raise KeyError(f"team {team_id} is not in this league")


class TradeOffer(Frozen):
    """A proposed swap, always written from the owner's point of view: ``give`` leaves this
    roster and ``get`` arrives from ``partner_team_id``."""

    partner_team_id: int
    give: tuple[Player, ...]
    get: tuple[Player, ...]

    @property
    def mirrored(self) -> TradeOffer:
        """The same trade seen from the partner's side of the table."""
        return TradeOffer(partner_team_id=self.partner_team_id, give=self.get, get=self.give)

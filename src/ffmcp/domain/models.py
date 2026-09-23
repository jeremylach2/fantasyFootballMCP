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
    bye_week: int | None = None
    """The NFL week this player's pro team is off, or ``None`` when the provider does not know.
    Rest-of-season value needs it: a player on bye during your first playoff week is worth one
    fewer week than his projection times the weeks remaining suggests."""
    season_rate: float | None = None
    """Projected points per game over the season, where the provider publishes one. Stands in
    for a weekly projection the player does not have this week (a bye), so a player is not
    valued at zero for the rest of the season just because this week happens to be his bye."""


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
    reception_points: float = 1.0
    """Points per reception: 1.0 for PPR, 0.5 for half-PPR, 0.0 for standard. Picks which of a
    second projection source's scoring variants matches this league."""
    playoff_round_weeks: int = 1
    """NFL weeks per playoff round (ESPN lets a league play two-week rounds)."""

    @property
    def playoff_weeks(self) -> tuple[int, ...]:
        """The NFL weeks the fantasy playoffs occupy, derived from the bracket size: eight
        teams is three rounds, so a 14-week regular season puts the playoffs in weeks 15-17."""
        rounds = max(0, (max(1, self.playoff_team_count) - 1).bit_length())
        first = self.reg_season_weeks + 1
        return tuple(range(first, first + rounds * max(1, self.playoff_round_weeks)))

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


class PlayerWeek(Frozen):
    """One rostered player's completed week: what was projected, what happened, and where he
    sat. The raw material for everything this server learns from the league's own history:
    variance, projection accuracy, lineup efficiency."""

    season: int
    week: int
    player_id: int
    name: str
    position: str
    pro_team: str
    eligible_slots: tuple[str, ...] = ()
    fantasy_team_id: int | None = None
    slot: str | None = None
    """The lineup slot he was in (``"BE"`` for the bench). ``None`` for a free agent."""
    projected: float | None = None
    actual: float = 0.0
    played: bool = True
    """Whether his NFL game was played. A bye or an inactive week is not a scoring outcome."""

    @property
    def started(self) -> bool:
        return self.slot is not None and self.slot not in NON_STARTING_SLOTS

    def as_player(self, *, hindsight: bool = True) -> Player:
        """A bare ``Player`` carrying this week's actual points (``hindsight``) or its
        pre-game projection as his projection, so a completed week can be run back through
        ``domain.optimizer`` with either what happened or what was known at the time."""
        points = self.actual if hindsight else self.projected
        source = "actual" if hindsight else "espn"
        return Player(
            player_id=self.player_id,
            name=self.name,
            position=self.position,
            eligible_slots=self.eligible_slots,
            pro_team=self.pro_team,
            projection=None
            if points is None
            else Projection(week=self.week, points=points, source=source),
        )


class UsageWeek(Frozen):
    """One player's NFL workload in one game, from play-by-play derived stats. Opportunity is
    what a player's role *gives* him; points are what he did with it. Only the first is stable
    enough to project from."""

    season: int
    week: int
    player_id: int
    """ESPN player id, so usage joins straight onto rosters."""
    name: str
    position: str
    pro_team: str
    snap_pct: float | None = None
    """Share of the team's offensive snaps, 0-1."""
    attempts: float = 0.0
    passing_air_yards: float = 0.0
    carries: float = 0.0
    targets: float = 0.0
    receiving_air_yards: float = 0.0
    target_share: float | None = None
    """Share of the team's targets, 0-1."""
    air_yards_share: float | None = None
    fantasy_points: float = 0.0
    """Points under this league's reception scoring."""


class GameLine(Frozen):
    """The betting market's view of one team's upcoming game."""

    pro_team: str
    opponent: str
    implied_total: float
    """Points the market expects this team to score: half the total, adjusted by the spread."""
    spread: float
    """This team's point spread, negative when favoured."""
    total: float
    kickoff: str | None = None


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

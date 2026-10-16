"""Resources: stable reference data the model would otherwise ask for repeatedly. Every
resource here is served under the server-wide ``resources/read`` cache hint set in
``server.py``.

The installed SDK's ``cache_hints`` are keyed by *method* (``resources/read``), not by
individual resource URI, so each provider caches this data upstream on its own schedule rather
than through a per-URI wire-level cache hint. There is no such mechanism to attach one.
``ffmcp://glossary`` is public in spirit but is still served under the single
``resources/read`` hint, which is scoped ``private``. That is conservative (public data
withheld from a shared cache costs nothing but a cache hit) rather than wrong.
"""

from __future__ import annotations

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceError

from ffmcp.domain.models import LeagueSettings
from ffmcp.errors import FFMCPError
from ffmcp.mcp._shared import (
    Detail,
    adapt_errors,
    app_context,
    enrich_team_roster,
    resolve_week,
    team_as_of_week,
    week_note,
)
from ffmcp.render.tables import render_roster
from ffmcp.server import get_app_context

GLOSSARY = """\
Fantasy Football MCP — glossary

PROJ    Projected fantasy points for the week, one decimal place. In a later week's view:
        the player's season average, everyone healthy.
ST      Status flag: injury-status initial (O=out, D=doubtful, I=injured reserve, ...),
        Q=questionable, BYE=his team is off this week, LCK=his game has kicked off, so his
        lineup slot is locked.
OWN%    Percent of leagues (ESPN) rostering this player.
STRT%   Percent of leagues (ESPN) starting this player.
ADD/24H Sleeper adds in the last 24 hours (126k = 126,000): competition for the player, not a
        quality signal.
VOR     Value over replacement: this player's projection minus the best player at the same
        position who would not be starting anywhere in the league. Answers "what is this
        player worth," not "how many points will he score".
Marginal value   The change in a roster's optimal-lineup points from adding or removing one
        player. What a trade or waiver claim is actually worth to *this* roster.
ROS     Rest-of-season value: lineup points a move adds over every remaining week (this one
        included), each player absent in his bye week, later weeks at season averages, playoff
        weeks weighted by your playoff odds. Waiver moves are ranked by it, add and drop
        together.
VAL     This week's part of a waiver move: lineup points it adds this week.
Stash   An injured-reserve free agent who can sit in an open IR slot with no drop; valued from
        when he is projected back.
Delta odds (odds)   Change in playoff probability, in percentage points, from a lineup, trade
        or waiver move, simulated with common random numbers so small effects are not lost in
        simulation noise.
PLAYOFF%   This server's simulated probability of making the playoffs. Every tool shows the
        same simulation (simulate_season, league_standings, analyze_matchup's stakes); ESPN's
        own percentage is never shown.
mkt     Market value (FantasyCalc) of what you give vs what you get. Offers where the other
        manager gives up more than 1.25x what they get are not suggested: they get declined.
SOS     Remaining strength of schedule: the average current win percentage of a team's
        opponents over the rest of the regular season.
FLOOR / CEIL   10th / 90th percentile outcome for the week. The spread is measured on a real
        league's season and widened when ESPN and Sleeper disagree; a questionable player
        ESPN projects at zero gets Sleeper's spread, since he may yet play.
IMPL    The player's NFL team's implied points from the betting line (total / 2 - spread / 2).
        Context only: ESPN's projection already reflects the matchup, so it is never added in.
Matchups   For defenses: each next opponent's implied points. Lower is better. Context only.
Win probability   Chance your lineup outscores this week's opponent. optimize_lineup maximises
        it, which occasionally means a lower-projected, higher-ceiling start when you are the
        underdog (or a steadier one when you are the favourite).
Live win prob   Once games start: the score so far plus each starter's projection for the
        share of his game still to play (estimated from kickoff time).
Stakes  Playoff odds if you win this week versus if you lose it.
XPPG    Expected points per game from volume alone (snaps, targets, carries, air yards).
LUCK    Actual minus expected points per game. Large gaps regress: the basis for buy-low and
        sell-high.
ALLPLAY Record if a team had played every other team every week.
LINEUP% Projected points started over the best lineup by the projections available that week:
        a manager's decisions, net of luck.
BENCH   Points per game a perfect-hindsight lineup would have added (mostly luck).
"""


def register(mcp: MCPServer) -> None:
    @mcp.resource(
        "ffmcp://league/settings",
        name="league_settings",
        title="League Settings",
        description="Scoring rules, roster slots and playoff format for this league.",
        mime_type="application/json",
    )
    async def league_settings() -> LeagueSettings:
        return await get_app_context().league.get_settings()

    @mcp.resource(
        "ffmcp://league/teams",
        name="league_teams",
        title="League Teams",
        description="Team id -> name index, needed to use partner_team_id / team_id.",
        mime_type="application/json",
    )
    async def league_teams() -> dict[str, str]:
        teams = await get_app_context().league.get_teams()
        return {str(team.team_id): team.name for team in teams}

    @mcp.resource(
        "ffmcp://glossary",
        name="glossary",
        title="Glossary",
        description="Definitions of VOR, marginal value, delta-odds and every table column.",
        mime_type="text/plain",
    )
    def glossary() -> str:
        return GLOSSARY

    @mcp.resource(
        "ffmcp://team/{team_id}/roster{?week,detail}",
        name="team_roster",
        title="Team Roster",
        description="Any team's roster, with the same shape as get_my_team.",
        mime_type="text/plain",
    )
    async def team_roster(
        ctx: Context, team_id: int, week: int | None = None, detail: Detail = "compact"
    ) -> str:
        with adapt_errors(ResourceError):
            app = app_context(ctx)
            settings = await app.league.get_settings()
            resolved_week = await resolve_week(ctx, week, settings)
            teams = await app.league.get_teams()
            team = next((t for t in teams if t.team_id == team_id), None)
            if team is None:
                raise FFMCPError(f"team {team_id} is not in this league")
            if detail != "compact":
                team = await enrich_team_roster(ctx, team)
            current_week = await app.league.get_current_week()
            return render_roster(
                team_as_of_week(team, resolved_week, current_week),
                resolved_week,
                detail,
                note=week_note(resolved_week, current_week),
            )

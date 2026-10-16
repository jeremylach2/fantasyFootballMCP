# Architecture

## 1. Layering

Four layers, one direction of dependency. Arrows point the only way imports are allowed to go.

```
  mcp/        thin adapters: decode args → call domain → render → return
    │         (no business logic, no arithmetic, no I/O)
    ▼
  render/     domain objects → compact text / small structured models
    │
    ▼
  domain/     PURE: models, optimizer, simulation, valuation, trades, variance,
    ▲         risk, schedule, roster, live, names, usage, league_intel, calibration
    │         (no I/O, no network, no MCP imports, no clock, no randomness
    │          except through an injected Generator)
  providers/  ALL I/O: espn-api, Sleeper, nflverse, The Odds API and FantasyCalc over
              httpx2, disk cache
```

`providers/` builds domain objects and hands them upward. `domain/` never reaches down. The
purity of `domain/` is what makes the interesting code testable in milliseconds with no network,
and it is enforced by `tests/test_layering.py` (AST walk over `src/ffmcp/domain/**`, assert no
import of `mcp`, `httpx2`, `httpx`, `espn_api`, `requests`, or `ffmcp.providers`).

Why this matters for the portfolio: the optimizer and simulator, the parts worth reading, are
ordinary pure functions. A reviewer can open `domain/optimizer.py` and understand it without
knowing anything about MCP or ESPN.

## 2. Request lifecycle

```
tools/call optimize_lineup(week=3)
  └─ mcp/tools_lineup.py
       ├─ resolve settings + league handle from lifespan context
       ├─ providers.espn.get_roster(week)      ─┐ cached, TTL by data volatility
       ├─ providers.espn.get_projections(week) ─┤
       ├─ providers.sleeper.get_market()       ─┘
       ├─ domain.optimizer.optimize(roster, slots, projections)   # pure
       ├─ domain.simulate.win_prob_delta(current, optimal)        # pure, seeded
       └─ render.detail.lineup_delta(...) → compact text + small structured result
```

The adapter is the only place that knows about caching, credentials, or MCP. It should read like
a table of contents.

## 3. Providers and caching

A single `Cache` with two tiers: in-process dict for the life of the server, and a JSON/msgpack
disk tier under `${FFMCP_CACHE_DIR:-~/.cache/ffmcp}`. Keys are namespaced and versioned
(`v1:espn:roster:{league}:{season}:{week}`) so a schema change invalidates cleanly.

TTLs follow how fast the underlying truth actually moves:

| Data | Source | TTL | Rationale |
|---|---|---|---|
| League handle (settings, rosters, live scores) | ESPN | 10 min; 60 s while a game is in progress | rosters change on waivers and trades; scores and lineup locks move by the minute on game day |
| Season pairings | ESPN | 24 h | fixed for the season; one request instead of a full-schedule download per remaining week |
| Free agents | ESPN | 10 min | moves on waivers |
| Recent activity | ESPN | 5 min | the point is recency |
| Completed weeks' box scores | ESPN | 7 d (6 h for last week) | history never changes once stat corrections settle |
| Player universe and injury detail | Sleeper | 6 h | ~14.6 MB; reduced on write; injury news lands through the week |
| Trending adds/drops | Sleeper | 15 min | that is the point of the signal |
| Second-opinion projections | Sleeper | 1 h (7 d for past weeks) | ~2 MB filtered to fantasy positions, reduced to two numbers per player |
| Injured players' week-by-week outlook | Sleeper | 6 h | per player, ~2 KB; moves with injury news |
| ID crosswalk (ESPN, GSIS, PFR) | nflverse | 24 h | 7 MB CSV reduced to two id maps |
| Weekly usage, snap counts | nflverse | 6 h | rebuilt overnight upstream |
| Betting lines | The Odds API | 12 h | free tier is 500 requests a month; 2 per fetch |
| Trade market values | FantasyCalc | 24 h | one request per league shape, reduced to `{espn_id: value}` |
| Season simulation | computed | 6 h, keyed by a fingerprint of everything it reads | every tool's playoff odds come from one run |

Two rules for the Sleeper player dump: it is fetched **lazily** (nothing needs it at startup) and
it is **immediately reduced** to an id→`{name, pos, team, injury}` index on write. The raw
document never sits in memory as parsed JSON longer than the reduction takes.

Startup must never block on network. The `lifespan` handler creates the httpx2 client and the
cache but does not warm them. A server that hangs for twelve seconds before answering
`tools/list` is a bad demo.

The ESPN `League` handle is the one thing that cannot go through the disk `Cache`: it is a live
`espn-api` object, not JSON. It is held in memory instead, rebuilt on the schedule above, and when
a rebuild fails the last good handle keeps serving (rule 3 in §4), logged as a warning. What that
does not yet do is *say so* in the tool response, the second half of rule 3: the provider knows
it is serving stale data, but no tool output carries the note. Everything ESPN serves beyond the
handle goes through the shared `Cache` with the TTLs above.

### ID mapping
ESPN and Sleeper disagree about player identity. Build one `providers/identity.py` resolver:
match on normalized `(name, position, pro_team)`, keep a small committed override table for the
handful that fail, and expose `unresolved` as a diagnostic count rather than a crash. Never let
an identity miss take down a tool. Degrade to "no market signal for this player."

Team defenses are not "the handful that fail" — they are *all* of them, on every league, and the
override table is the wrong tool for it: ESPN's D/ST position (`"D/ST"`, name `"Falcons D/ST"`)
and Sleeper's (`"DEF"`, name `"Atlanta Falcons"`) share no token at all, so no normalization of
either field would ever line them up. `IdentityIndex` resolves a defense on the one thing both
providers do agree on instead: the pro-team abbreviation (Sleeper's `player_id` for a defense
literally *is* that abbreviation), aliasing the single case they spell differently
(`WSH` on ESPN, `WAS` on Sleeper).

## 4. Error model

`errors.py` defines a small taxonomy, each mapped to a short, *actionable* message:

| Error | Cause | What the model is told |
|---|---|---|
| `CredentialsMissing` | no cookies, private league | how to get `espn_s2`/`SWID`, in one sentence |
| `LeagueNotAccessible` | wrong id, or cookies expired | which of the two it is, if distinguishable |
| `UpstreamUnavailable` | ESPN/Sleeper 5xx or timeout | that it is transient; whether stale cache was served |
| `PlayerNotFound` | bad name | the candidates on ambiguity, or up to 3 "did you mean" names (`domain/names.py`) |
| `WeekOutOfRange` | week < 1 or > current | valid range |

Three rules:

1. Errors are short. An error is a tool response and costs tokens like any other: two lines,
   no stack traces into the context. Those go to stderr.
2. Errors never leak secrets. A single `redact()` helper strips `espn_s2`, `SWID`, and any
   `Cookie` header from every message and log line. Apply it at the boundary, not at call sites,
   and unit-test it against a string containing a fake cookie.
3. Stale beats absent. If upstream fails and the cache holds an expired entry, serve it and
   say so in one clause (`cached 14m ago; ESPN unreachable`). A stale roster is far more useful
   than an exception.
4. Optional beats fatal. Every source beyond ESPN (Sleeper projections, nflverse usage, betting
   lines, even the league's own history) only refines an answer ESPN alone can give. Its
   failure is caught in `mcp/_shared.py` (`OPTIONAL_FAILURES`) and becomes a one-line caveat,
   never a failed tool call.

## 5. Configuration

`config.py`, `pydantic-settings`, prefix `FFMCP_`. Recall that SDK v2 no longer reads `.env` or
`MCP_*` for you: this layer is now entirely yours.

| Setting | Default | Notes |
|---|---|---|
| `FFMCP_MODE` | `live` | `live` \| `demo` (fixtures, no network, no credentials) |
| `FFMCP_LEAGUE_ID` | — | required in live mode |
| `FFMCP_SEASON` | current | derived from Sleeper `/v1/state/nfl` when unset |
| `FFMCP_TEAM_ID` | — | which team is "mine"; if unset, elicit once and cache |
| `FFMCP_ESPN_S2` | — | secret; private leagues only |
| `FFMCP_SWID` | — | secret; private leagues only |
| `FFMCP_ODDS_API_KEY` | — | secret; optional, betting lines; `ODDS_API_KEY` also accepted |
| `FFMCP_CACHE_DIR` | `~/.cache/ffmcp` | |
| `FFMCP_SIMS` | `10000` | Monte Carlo iterations |

Validation happens at startup and produces one clear error, not a traceback. `.env.example`
documents every variable with a comment. `.env` is gitignored. Since the repo is public, add a
`git secrets`-style pre-commit hook that refuses any commit containing a string matching the
`SWID` `{8-4-4-4-12}` GUID shape or a long `espn_s2` value.

## 6. Demo mode

`FFMCP_MODE=demo` swaps the provider implementations for fixture-backed ones behind the same
protocols (`providers/base.py` defines them as `typing.Protocol`). No branching inside tools: the
swap happens once, in `lifespan`. Demo mode also replays the moment its fixture was recorded as
"now" (`AppContext.clock`), so lineup locks and trade deadlines read the same in a test next year
as they did on the day.

Two fixture sets are committed, selected by `FFMCP_FIXTURE`:

* `demo` (the default): a synthetic ten-team league, generated by `scripts/make_demo_fixture.py`
  and `scripts/make_demo_insights.py`. Invented players, so publishing it publishes nobody's
  roster. Every README transcript is generated from it by `uv run poe demo`, so a reader can
  reproduce it exactly.
* `snapshot`: a real 16-team league at week 6 of 2026, recorded by `scripts/record_fixtures.py`
  and anonymized (team names → `Team Alpha`…, league id → `1234567`; the domain models carry no
  owner data). Player names stay real: the analysis is meaningless without them and they are
  public facts. It exists for the regression tests (`tests/mcp/test_snapshot_regressions.py`),
  because the synthetic league differed from live ESPN in exactly the ways that had hidden real
  bugs. The recorder refuses to overwrite `demo`.

## 7. Concurrency

Tools are `async`. Independent upstream fetches within one tool run under `asyncio.gather`.
`espn-api` is synchronous, so wrap its calls in `asyncio.to_thread`. Do not block the event loop.
One shared `httpx2.AsyncClient` for the process lifetime, created in `lifespan`, with a connect
timeout of 5 s and a read timeout of 15 s.

Monte Carlo runs in `asyncio.to_thread` as well (it is CPU-bound numpy) and reports progress via
`ctx.report_progress()` at 10 % intervals.

## 8. Observability

Structured logging to **stderr** only (stdout is the stdio protocol channel: writing to it
corrupts the session, and this is the single most common way to break an MCP server). One line per
tool call: tool name, duration, cache hit/miss, bytes fetched upstream, and the approximate token
size of the response. That last field is what makes the token work measurable in normal operation
rather than only in the benchmark, and it costs nothing.

Optionally attach a `ServerMiddleware` to do this uniformly rather than decorating each tool.

## 9. A second projection source (implemented)

ESPN's own `projected_points` was the sole basis for every projection the server produced, and
an earlier version of this section proposed a cross-check against a second source, surfaced as a
disagreement flag and never blended into the number itself. It was shelved because the two
candidates did not fit (FantasyFootballCalculator's ADP is draft-time only; FantasyPros needs an
account and a key).

Sleeper's weekly projections turned out to fit: no key, the same identity machinery that already
reconciles ESPN ids against Sleeper's (`providers/identity.py`), and both standard and PPR figures,
so any reception scoring interpolates exactly. `SleeperMarketProvider.get_alt_projections` serves
them keyed by ESPN id.

The design principle survived, and then the data backed it. Measured over a real league's 2025
season (`scripts/calibrate.py`), Sleeper and ESPN are equally accurate (mean absolute error 5.68
each) and their average barely improves on either (5.66), so the number is never blended. But
the *disagreement* between them is strongly predictive of error: the most-disputed 5% of
projections missed by 2.5 times as much as the least-disputed half. So disagreement widens a
player's range (`domain/variance.py`), `player_report` shows the second opinion, and
`projection_accuracy` reports both sources' error in the user's own league.

## 10. Calibration

Every fitted constant in `domain/` (position spreads, the team correlation factor, the
disagreement sensitivity, the expected-points coefficients, the luck and lineup-accuracy
thresholds) comes from `scripts/calibrate.py`, which prints the evidence for each against real
data and is re-run by hand. Its output is copied into the code deliberately, so a refit is a
reviewed diff rather than a silent drift, and each constant's docstring says where it came from.
The script also records what was tested and rejected (per-player bias correction, per-player
spread from history), because a model that only lists what it kept cannot show it was tested.

Sections 8 and 9 work on the season model, from completed seasons of the same league
(`--backtest 2024 2025`, or `--only-season-model` to run just these). Section 8 fits
`TEAM_STRENGTH_SD` as a one-way random effect: across every (team, starting week), the variance
of a team's average miss over the rest of the season, less what weekly noise explains. Section 9
re-simulates each season from weeks 4, 6, 8 and 10 with the shipped model and the one it replaced,
and scores both against who actually made the playoffs. Its limits are stated with its numbers:
season rates are reconstructed from the projections known by each starting week, and ESPN's own
historical playoff percentages are not available, so models are compared with each other, not
with ESPN.

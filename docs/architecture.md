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
    ▲         risk, schedule, usage, league_intel, calibration
    │         (no I/O, no network, no MCP imports, no clock, no randomness
    │          except through an injected Generator)
  providers/  ALL I/O: espn-api, Sleeper, nflverse and The Odds API over httpx2,
              disk cache
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
| League settings, scoring, slots | ESPN | 24 h | effectively static in-season |
| Rosters | ESPN | 10 min | changes on waivers/trades |
| Live scores / box scores | ESPN | 60 s | moves during games |
| Weekly projections | ESPN | 30 min | updates through the week |
| Player universe | Sleeper | 24 h | ~14.6 MB; fetch once a day, never in full to the model |
| Trending adds/drops | Sleeper | 15 min | that is the point of the signal |
| Completed weeks' box scores | ESPN | 7 d (6 h for last week) | history never changes once stat corrections settle |
| Second-opinion projections | Sleeper | 1 h (7 d for past weeks) | ~2 MB filtered to fantasy positions, reduced to two numbers per player |
| ID crosswalk (ESPN, GSIS, PFR) | nflverse | 24 h | 7 MB CSV reduced to two id maps |
| Weekly usage, snap counts | nflverse | 6 h | rebuilt overnight upstream |
| Betting lines | The Odds API | 12 h | free tier is 500 requests a month; 2 per fetch |

Two rules for the Sleeper player dump: it is fetched **lazily** (nothing needs it at startup) and
it is **immediately reduced** to an id→`{name, pos, team}` index on write. The raw document never
sits in memory as parsed JSON longer than the reduction takes.

Startup must never block on network. The `lifespan` handler creates the httpx2 client and the
cache but does not warm them. A server that hangs for twelve seconds before answering
`tools/list` is a bad demo.

**Gap, stated rather than hidden:** the differentiated TTLs in the table above describe what
`SleeperMarketProvider`, `NflverseUsageProvider` and `OddsApiProvider` actually do, and what
`EspnLeagueProvider.get_player_history` does for completed weeks, but the rest of
`EspnLeagueProvider` does not use the shared `Cache`. It holds one `espn_api.League` handle behind a single blanket
`_LEAGUE_TTL_SECONDS` (600s), so league settings, rosters and live scores all refresh together
on one timer rather than on the four separate schedules the table above implies. Rule 3 in §4
below ("stale beats absent") is likewise unimplemented on the ESPN path: `Cache.get_stale()`
and `UpstreamUnavailable(stale_served=...)` both exist, but no call site in `providers/espn.py`
uses either, so an ESPN hiccup today is a hard error rather than a stale-and-say-so response.
Wiring the shared `Cache` into `EspnLeagueProvider` per-endpoint, with the TTLs the table already
documents, would close both gaps at once.

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
| `PlayerNotFound` | bad name | up to 3 near-matches, nothing more |
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
protocol (`providers/base.py` defines `LeagueProvider` and `MarketProvider` as `typing.Protocol`).
No branching inside tools: the swap happens once, in `lifespan`.

Fixtures are produced by `scripts/record_fixtures.py`, which pulls a real league and anonymizes
it: team names → `Team Alpha`…, owner names → removed, league id → `1234567`. Player names stay
real, because the analysis is meaningless without them and they are public facts.

This is what makes the README credible: every example in it is generated from the fixture league
by `uv run poe demo`, so a reader can reproduce it exactly.

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

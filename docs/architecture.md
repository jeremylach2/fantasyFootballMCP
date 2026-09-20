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
  domain/     PURE: models, optimizer, simulation, valuation, trades
    ▲         (no I/O, no network, no MCP imports, no clock, no randomness
    │          except through an injected Generator)
  providers/  ALL I/O: espn-api, Sleeper httpx2, disk cache
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

Two rules for the Sleeper player dump: it is fetched **lazily** (nothing needs it at startup) and
it is **immediately reduced** to an id→`{name, pos, team}` index on write. The raw document never
sits in memory as parsed JSON longer than the reduction takes.

Startup must never block on network. The `lifespan` handler creates the httpx2 client and the
cache but does not warm them. A server that hangs for twelve seconds before answering
`tools/list` is a bad demo.

**Gap, stated rather than hidden:** the differentiated TTLs in the table above describe what
`SleeperMarketProvider` actually does, but `EspnLeagueProvider` does not use the shared
`Cache` at all. It holds one `espn_api.League` handle behind a single blanket
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

## 9. Future work: consensus-rank cross-check (not implemented)

Chosen to not implement this to not require users to sign up for an account and create an API key.

ESPN's own `projected_points` is the sole basis for every projection and VOR figure the server
produces (`providers/espn.py`). For high-variance positions, such as D/ST and K, a
handful of sacks or a return TD swings a week. One site's model can lag what the broader analyst
community already knows (an O-line injury, a defensive coordinator change) faster than it can
propagate into ESPN's own number. A second, independent signal would let a tool say "ESPN
projects A over B, but expert consensus ranks B higher" instead of presenting ESPN's figure as
uncontested.

**Why this isn't built:** the design is straightforward, but a suitable data source is not. Two
free options exist and neither fits:

- **FantasyFootballCalculator ADP API** — genuinely free, no signup, attribution only
  (help.fantasyfootballcalculator.com/article/42-adp-rest-api). But ADP is a draft-time signal; it
  does not move week to week and cannot answer a weekly streaming question like D/ST.
- **FantasyPros Public API** — the real weekly expert-consensus rankings data, but gated behind
  account signup and an API key (fantasypros.com/api-data). Free at this scale, but not the
  zero-friction, no-auth shape `SleeperMarketProvider` enjoys.

Picking this up later means accepting the FantasyPros signup, or finding an equivalent weekly
consensus source with a comparably open API.

**Proposed shape, if built**, following the pattern `SleeperMarketProvider` already establishes
for a second upstream source:

- A `ConsensusSignal` domain model (`rank: int | None`, `source: str`) alongside the existing
  `MarketSignal` on `Player` (`domain/models.py`) — frozen, optional, vendor-agnostic.
- A `ConsensusRankProvider`, resolved through the same `providers/identity.py` machinery already
  reconciling ESPN IDs against Sleeper's, so a third vocabulary doesn't need its own matcher.
- An `enrich_with_consensus` step in `mcp/_shared.py`, mirroring `enrich_with_market`.
- Surfaced in `compare_players` and `find_waiver_targets` (`mcp/tools_roster.py`) as a disagreement
  flag only — shown when ESPN's ordering and consensus rank disagree — never blended into the VOR
  score itself. `valuation.py` already establishes the principle for the market-sentiment nudge:
  a soft signal may act as a tiebreaker within a few percent, never overturn a real projection
  gap. Consensus rank should follow the same rule rather than get its own weighting scheme.

# fantasyFootballMCP

I lost last year's fantasy league, and I am determined to never lose again.
Instead of studying draft tactics, doing deep dives on new players, obssesing over my lineup every week,,, I would rather create this :) 

One of my mistakes last year was trusting ESPN analytics, so I used Claude to help me discover metrics and algos I can use to maximize my team's point potential per week.

I created an MCP server that answers the four questions a fantasy football manager actually asks (*who
should I start, should I make this trade, who should I pick up, will I make the playoffs*)
by doing the analysis in Python and returning the conclusion, not the data.

### Let's jump in 
```
$ optimize_lineup()
{
  "week": 8,
  "current_projected": 95.3,
  "optimal_projected": 106.1,
  "point_gain": 10.8,
  "win_prob_delta": 0.8,
  "swaps": [
    { "bench": "Tariq Whitlock", "starter": "Marcus Moreau",     "slot": "WR",   "gain": 10.7, "reason": "+10.7 projected points" },
    { "bench": "Devin Obi",      "starter": "Elijah Villanueva", "slot": "FLEX", "gain":  3.5, "reason": "+3.5 projected points" },
    { "bench": "Soren Quintero", "starter": "Dominic Okafor",    "slot": "RB",   "gain": -3.4, "reason": "Soren Quintero is OUT" }
  ],
  "caveats": []
}
```

Three swaps, 159 tokens. The alternative, handing the model sixteen players, their
projections, their injury designations and the league's slot eligibility rules, and asking it
to solve an assignment problem in context, costs ~31,900 tokens and gets the answer wrong,
because the correct answer is a maximum-weight bipartite matching and not a sort.

Every transcript in this README is real output from `uv run poe demo`, which runs the whole
tool surface against a committed fixture league. None of it is written by hand.

## Quickstart

No ESPN account, no credentials, no network beyond PyPI:

```bash
git clone https://github.com/jeremylach2/fantasyFootballMCP
cd fantasyFootballMCP
uv sync
uv run poe demo
```

Connect Claude Desktop by adding this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fantasy-football": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/fantasyFootballMCP", "ffmcp"],
      "env": { "FFMCP_MODE": "demo" }
    }
  }
}
```

Or Claude Code, from inside the repo:

```bash
claude mcp add fantasy-football --env FFMCP_MODE=demo -- uv run ffmcp
```

Drop `FFMCP_MODE=demo`, copy `.env.example` to `.env`, and set `FFMCP_LEAGUE_ID` from your
league's URL. Private leagues also need `FFMCP_ESPN_S2` and `FFMCP_SWID`, the two session cookies
from a logged-in ESPN browser session (DevTools → Application → Cookies →
`fantasy.espn.com`). Those cookies are as sensitive as your ESPN password, so `.env` is
gitignored and every error message is redacted before it can reach a log or a model.
`FFMCP_TEAM_ID` picks which roster is "mine". Leave it unset and the server asks once through
MCP elicitation. For streamable HTTP instead of stdio, run `uv run poe serve` and point a client
at `http://localhost:8000/mcp`.

## What it does

Ten tools, four resources, three prompts. All output below is verbatim from the demo league.

| Tool | Does |
|---|---|
| `get_my_team` | Show my roster with projections, injuries and bye weeks. |
| `optimize_lineup` | Find the highest-scoring legal lineup and the swaps to get there. |
| `analyze_matchup` | Break down this week's head-to-head matchup and win probability. |
| `find_trades` | Suggest trades that help my roster and are plausibly accepted. |
| `evaluate_trade` | Evaluate a specific proposed trade from both sides. |
| `find_waiver_targets` | Rank available free agents by how much they would help my team. |
| `simulate_season` | Simulate the rest of the season for playoff odds and seeding. |
| `league_standings` | Show standings with playoff odds and strength of schedule. |
| `player_report` | Detailed outlook for one player: projection, value over replacement, market. |
| `compare_players` | Compare players head-to-head for a start/sit decision. |

Plus four resources (`ffmcp://league/settings`, `ffmcp://league/teams`, `ffmcp://glossary`,
`ffmcp://team/{team_id}/roster`) that carry stable reference data with long cache TTLs instead of
being re-fetched as tool calls, and three prompts (`weekly_checkin`, `trade_workshop`,
`playoff_push`) that chain several tools into one routine.

### Who should I start?

`get_my_team` shows the lineup as ESPN has it set. Note the running back ruled out and the
receiver on bye, both still in the lineup: it is Sunday morning and nobody has touched it.

```
Week 8 — Team Alpha (3-4-0)
SLOT  PLAYER           POS  TM  PROJ  ST
QB    L. Petrov        QB   LAC 22.6
RB    S. Quintero      RB   DAL 11.8  O
RB    I. Adeyemi       RB   IND 9.6
WR    T. Whitlock      WR   SEA       BYE
WR    D. Rios          WR   GB  15.9
TE    L. Santoro       TE   NO  13.0
FLEX  D. Obi           TE   DET 6.9
D/ST  WSH D/ST         D/ST WSH 6.7
K     T. Prescott      K    MIN 8.8
BE    Q. Hargrove      QB   IND 19.4
BE    T. Moreau        RB   KC  6.9
BE    D. Okafor        RB   TEN 8.4
BE    M. Moreau        WR   NYG 10.7
BE    E. Villanueva    WR   NE  10.4
BE    D. Holloway      WR   BAL 6.9
IR    L. Novak         RB   MIA 5.2   I
```

`optimize_lineup` is the response at the top of this README: +10.8 projected points, worth
+0.8 points of win probability this week. `analyze_matchup` prices the week:

```
Week 8: Team Alpha 106.1 vs Team Charlie 118.6 — 39% win prob
Biggest edges:
  RB: Team Charlie +25.9
  K: Team Alpha +8.8
  WR: Team Alpha +4.8
Swing player: D. Duval (WR MIA, ±9.4 pts)
```

### Should I make this trade?

`find_trades` enumerates 1-for-1, 2-for-1 and 1-for-2 packages across every other roster,
**keeps only the ones both sides gain from**, and ranks them by what they do to *my* playoff
odds:

```
Team Bravo: give Q. Hargrove / get B. Battaglia — me +4.2%odds, them +11.2pts — Strengthens RB by 8.0 started pts/wk.
Team Bravo: give L. Petrov / get L. Castellan, T. Obi — me +4.0%odds, them +0.7pts — Strengthens RB by 9.8 started pts/wk.
Team Bravo: give L. Petrov / get C. Castellan, E. Brennan — me +3.4%odds, them +6.3pts — Strengthens RB by 6.7 started pts/wk.
Team Bravo: give L. Petrov / get L. Castellan — me +3.4%odds, them +21.0pts — Strengthens RB by 9.8 started pts/wk.
Team Bravo: give Q. Hargrove / get C. Castellan — me +3.4%odds, them +20.3pts — Strengthens RB by 6.7 started pts/wk.
```

The first row is the whole thesis of the trade module in one line. Quentin Hargrove is a
19.4-point quarterback, on a roster that already starts a 22.6-point quarterback, in a league
with one QB slot. He is worth **zero** to Team Alpha, because he never plays. He is worth 6.5
points a week to Team Bravo, who start a 12.9-point quarterback. *What a player is worth is a
property of the roster, not of the player*, and that asymmetry is the only reason trades happen.
`marginal_value` measures it as the change in a roster's **optimal lineup** from adding or
removing him, which is exactly why the optimizer had to be both correct and fast.

`evaluate_trade` prices one named offer from both sides:

```
{
  "verdict": "accept",
  "my_value_delta": 46.9,
  "partner_value_delta": 20.3,
  "my_playoff_odds_delta": 3.4,
  "positional_impact": "Strengthens RB by 6.7 started pts/wk.",
  "risks": []
}
```

An offer only one side gains from is a fleece and will be declined by any manager who does the
same arithmetic, so those are filtered out rather than ranked low. The filter is the feature.

### Who should I pick up?

`find_waiver_targets` ranks free agents by marginal value **to this roster**, not by
projection, and not by how many people are adding them, and always names the drop, because a
pickup recommendation without one is not advice:

```
PLAYER           POS  TM  PROJ  VAL    DROP
D. Ivanov        RB   CIN 12.4  +4.0   T. Whitlock
C. Fairbanks     WR   KC  13.1  +2.7   T. Whitlock
ARI D/ST         D/ST ARI 8.8   +2.1   T. Whitlock
I. Battaglia     WR   CHI 10.6  +0.2   T. Whitlock
```

Caleb Fairbanks is the higher projection. Devin Ivanov is the better pickup, because this roster
is thin at running back and deep at receiver.

### Will I make the playoffs?

`simulate_season` runs a vectorized Monte Carlo over the remaining schedule:

```
2,000 sims, weeks 8-14:
TEAM             REC     MEANW  PLAYOFF%  TITLE%
T. Echo          6-1     10.3   95.8%     27.8%
T. Bravo         6-1     10.0   93.2%     22.8%
T. Foxtrot       5-2     9.2    84.9%     24.8%
T. Hotel         5-2     8.3    56.2%     8.6%
T. Charlie       4-3     8.0    52.5%     12.7%
T. India         2-5     5.7    8.5%      2.1%
T. Juliet        2-5     5.6    5.2%      0.8%
T. Alpha         3-4     5.7    3.6%      0.5%
T. Delta         1-6     4.0    0.1%      0.0%
T. Golf          1-6     3.3    0.1%      0.0%
```

Team Alpha is 3-4 and behind two 2-5 teams in playoff odds. That is correct, and the reason to
simulate rather than extrapolate a record: Alpha's roster projects worse than either of theirs,
and seven weeks is enough for that to matter more than one game of standings.

## How it works

Four layers, one direction of dependency:

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
  providers/  ALL I/O: espn-api, Sleeper over httpx2, disk cache
```

`providers/` builds domain objects and hands them upward. `domain/` never reaches down. That is
what makes the interesting code, the optimizer and the simulator, testable in milliseconds with
no network and no MCP, and it is enforced rather than asserted: `tests/test_layering.py` walks
the AST of every module under `domain/` and fails if one of them imports `mcp`, `httpx2`,
`httpx`, `espn_api`, `requests` or `ffmcp.providers`.

Upstream payloads never reach the model. Sleeper's player document is 14.6 MB. It is fetched at
most daily and reduced to an id→`{name, pos, team}` index *on write*. ESPN league payloads are
cached with TTLs that follow how fast the underlying truth moves (league settings 24 h, rosters
10 min, live scores 60 s) and projected into narrow domain models before anything is returned.
Startup does no network at all, so `tools/list` answers immediately.

Request lifecycle: a single tool invocation flows through resolving settings and the league
handle from context, fetching upstream data (cached, TTL by volatility), calling pure domain
code, and rendering the result. The adapter decodes args and orchestrates. `domain/` never
knows about MCP, ESPN, or credentials.

Error handling: errors are short (one or two lines, never stack traces). A `redact()` helper
strips secrets from every message and log line. When upstream fails and cache holds an expired
entry, serve the stale data and say so: "cached 14m ago; ESPN unreachable" is far more useful
than an exception. See [`docs/architecture.md`](docs/architecture.md) for the full architecture.

## Token efficiency

Response cost here is designed, measured, and regression-tested like any other engineering
property. `src/ffmcp/budgets.py` is the single source of truth for the per-tool budgets.
`tests/test_token_budgets.py` asserts every tool at every detail level against them and fails
CI on a regression. `uv run poe bench` regenerates the table below from real fixture output, and
CI fails if the committed table has gone stale.

Three rules do most of the work. **Return conclusions, not data**: `optimize_lineup` returns
the swaps, not the roster. **Default to the cheapest useful detail**: every list-shaped tool
takes `detail: "compact" | "standard" | "full"` and defaults to compact. **Pick one channel per
payload shape**: the SDK derives `structuredContent` from the return annotation *and* emits a
text block, and both reach the model, so lists return `str` with `structured_output=False` while
small decision-shaped results (`LineupAdvice`, `TradeVerdict`) return Pydantic models where the
schema earns its keep. That last one is a measured trade-off, not a preference: for a 15-row
table, carrying both channels is close to double cost for no gain.

<!-- BENCH:START -->
| tool | detail | tokens | upstream bytes | tokens-if-naive |
|---|---|---:|---:|---:|
| `get_my_team` | compact | 176 | 127,618 | ~31,904 |
| `get_my_team` | standard | 237 | 127,618 | ~31,904 |
| `get_my_team` | full | 262 | 127,618 | ~31,904 |
| `optimize_lineup` | - | 366 | 127,618 | ~31,904 |
| `analyze_matchup` | compact | 47 | 127,618 | ~31,904 |
| `analyze_matchup` | standard | 64 | 127,618 | ~31,904 |
| `find_trades` | - | 173 | 127,618 | ~31,904 |
| `evaluate_trade` | - | 48 | 127,618 | ~31,904 |
| `find_waiver_targets` | compact | 62 | 127,618 | ~31,904 |
| `find_waiver_targets` | standard | 76 | 127,618 | ~31,904 |
| `find_waiver_targets` | full | 84 | 127,618 | ~31,904 |
| `simulate_season` | - | 136 | 127,618 | ~31,904 |
| `league_standings` | compact | 170 | 127,618 | ~31,904 |
| `player_report` | - | 44 | 127,618 | ~31,904 |
| `compare_players` | - | 36 | 127,618 | ~31,904 |

_Every row's naive comparison is against the full fixture league payload (a 10-team league snapshot, standing in for a real upstream fetch) serialized as-is, rather than the reduced, narrow response above it. `tokens` is the deterministic local estimator. `upstream bytes`/`tokens-if-naive` are the same for every row because this benchmark runs against one fixture league. A live league's numbers vary by roster size and week, but the shape of the reduction does not._
<!-- BENCH:END -->

The naive column is the honest comparison: it is what a passthrough server would spend to let
the model answer the same question itself. The reduction runs from 140× to nearly 900×, and the
answer is also *correct*, which the passthrough version would not reliably be.

`tokens` comes from a deterministic local estimator so CI needs no API key and no network. The
estimator's error against Anthropic's real token-counting endpoint has **not** been measured,
since this environment had neither the optional `anthropic` package nor a key. This functionality will be added in the future if requested.

## The optimizer

See: [`src/ffmcp/domain/optimizer.py`](src/ffmcp/domain/optimizer.py).

Assign rostered players to starting slots to maximize total projected points, where each slot
admits a set of positions (`RB/WR/TE` takes a back, receiver or tight end), each player fills at
most one slot, and each slot holds at most one player.

Greedy is wrong, and provably so. The obvious algorithm, walking the slots in order and dropping
the best eligible player into each, strands position-locked slots. ESPN declares its `RB/WR` flex
*before* the locked `WR` slot, so a greedy pass hands the flex the receiver the locked slot
needed and backfills `WR` with whatever is left. The concrete counterexample is the first test
in the file, written before the module was:

| | slot `RB/WR` | slot `WR` | total |
|---|---|---|---|
| greedy | Top Receiver (17.0) | Scrub Receiver (3.0) | **20.0** |
| optimal | Solid Back (12.0) | Top Receiver (17.0) | **29.0** |

Nine points, on a two-slot roster, from an algorithm that looks obviously fine. On a real
lineup the gap is smaller and much harder to notice, which is worse.

So it is solved exactly, as a **maximum-weight bipartite matching**: the Hungarian method in
its successive-shortest-augmenting-path form with dual potentials, O(n²m) for n slots and m
candidates. At roster scale (n ≈ 10, m ≈ 16) that is microseconds, which is what makes it
affordable to call it thousands of times inside the trade search. Hand-written, about sixty
lines of matching, no scipy. `greedy()` is kept in the module purely as a test baseline, marked as such, so the gap
stays measurable: a property test asserts optimal ≥ greedy across 1,000 randomized rosters.

Every slot is also offered a zero-weight "leave it empty" option, which is what lets a roster
too thin or too injured to fill the lineup produce a valid partial lineup instead of an error.
Ties break by player id, so output is stable across runs.

## Simulation

[`src/ffmcp/domain/simulate.py`](src/ffmcp/domain/simulate.py) runs the rest of the season as a
vectorized numpy Monte Carlo: each team's weekly score is drawn from a distribution whose mean
is its optimal-lineup projection and whose variance comes from position-level dispersion. That
variance model is documented in the module as an *assumption*, because it is one.

**Why Δ win-probability instead of Δ points.** "This lineup change is worth 10.8 points" is
not the question. The question is whether it wins the week and the season, and the answer
depends on the opponent, the schedule and the standings. A 10-point gain is decisive in a close
matchup and irrelevant in a blowout, and only a simulation knows which one you are in. So
`optimize_lineup` reports both, and `find_trades` *ranks* on odds.

Common random numbers: comparing two scenarios by running two independent simulations would
bury a one-point lineup gain under sampling noise. The difference you are trying to measure is
far smaller than the standard error of either run. So `win_prob_delta` draws a single
`(n_sims, n_weeks, n_teams)` noise array and evaluates **both** scenarios against it. The
scenarios then differ only by the change under test, and the noise cancels in the difference.
This is the non-obvious technique in the file and it has a test that *is* the argument for it:
the delta for a strictly better lineup must be positive in ≥ 95% of 100 seeds, while the same
comparison with independent draws is measurably noisier.

Everything is seeded through an injected `numpy.random.Generator`. No module-level global
random state, so runs are exactly reproducible. 10,000 sims for a 12-team league finish in
under two seconds, and the work runs in `asyncio.to_thread` with progress reported via
`ctx.report_progress()`.

## Modern MCP

Built against MCP spec `2026-07-28` and Python SDK 2.x (`MCPServer`), **not** the v1
`FastMCP` API that most training data describes. What that buys, and where to look:

| Feature | Where | Why |
|---|---|---|
| **Cache hints** | `server.py:CACHE_HINTS` | `tools/list`, `prompts/list`, `resources/list` cached 1 h public; `resources/read` 5 min private |
| **Tool annotations** | every `mcp/tools_*.py` | `read_only_hint=True` on all ten tools, asserted by `tests/mcp/test_annotations.py` |
| **Resource templates** | `mcp/resources.py` | `ffmcp://team/{team_id}/roster{?week,detail}` — RFC 6570 |
| **Elicitation, with a fallback** | `mcp/_shared.py:resolve_my_team_id` | asked once when `FFMCP_TEAM_ID` is unset; a client that cannot elicit gets a two-line error naming the variable, never a guess |
| **Progress** | `simulate_season`, `find_trades` | `ctx.report_progress()` forwarded from pure domain code via an injected callback, so `domain/` never learns what MCP is |
| **Stateless HTTP** | `server.py:main` | `stateless_http=True`; no session affinity behind a load balancer |
| **Deterministic list order** | `server.py:build_server` | tools registered in documented order for prompt-cache stability, asserted by `tests/mcp/test_tool_order.py` |

And, as deliberately, what is **not** used:

| Not used | Why |
|---|---|
| **Sampling** (`ctx.session.create_message`) | This server makes no LLM calls. Analysis is deterministic Python; the host model narrates. |
| **Roots** (`ListRoots`) | Nothing here is filesystem-scoped. Paths come from config. |
| **Logging** (`ctx.log` / `ctx.info`) | Deprecated in spec `2026-07-28`. Logs go to stderr via the stdlib — and on stdio, stdout is the protocol channel. |
| **Write operations of any kind** | The server holds ESPN session cookies. It recommends; the human acts. The blast radius of a confused model is zero. |

Knowing what a spec revision deprecated is a stronger signal than using every feature it kept.
That table was built by introspecting the installed SDK (`mcp==2.2.0`) directly, not recalled
from memory, since most MCP examples online still target the superseded v1 `FastMCP` API.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `FFMCP_MODE` | `live` | `live` or `demo` (committed fixtures, no network, no credentials) |
| `FFMCP_LEAGUE_ID` | — | required in live mode |
| `FFMCP_SEASON` | current | derived from Sleeper's `/v1/state/nfl` when unset |
| `FFMCP_TEAM_ID` | — | which team is "mine"; elicited once if unset |
| `FFMCP_ESPN_S2` | — | secret; private leagues only |
| `FFMCP_SWID` | — | secret; private leagues only |
| `FFMCP_CACHE_DIR` | `~/.cache/ffmcp` | |
| `FFMCP_SIMS` | `10000` | Monte Carlo iterations |

Secrets are typed `SecretStr`, and `errors.py:redact()` strips anything cookie-shaped from every
error message and log line at the boundary (unit-tested against a fake `espn_s2` and a `SWID`
GUID). See the [Quickstart](#quickstart) above for where to find ESPN cookies.

## Testing

```bash
uv run poe check    # ruff lint + format, mypy --strict, pytest, benchmark freshness
```

146 tests, no network anywhere: fixtures for ESPN, `httpx2.MockTransport` for Sleeper. Beyond
the usual, the suite carries a few tests that exist to turn claims into facts: the layering
purity check, the greedy-vs-optimal counterexample, the common-random-numbers comparison, the
token budgets, and a hygiene test asserting no fixture contains a real league id or anything
cookie-shaped.

The demo league itself is synthetic (invented players, invented projections), generated
deterministically by `scripts/make_demo_fixture.py` and committed, so that shipping a runnable
demo does not mean publishing anyone's real roster. `scripts/record_fixtures.py` is the other
path, for anonymizing a real league you have consent to use.

One gap, stated rather than hidden: the surface has been exercised end-to-end over real stdio
and through an in-process `mcp.Client`, but not yet by hand through a GUI client.

## Docker

```bash
docker build -t ffmcp .
docker run -p 8000:8000 -e FFMCP_MODE=demo ffmcp        # no credentials
docker run -p 8000:8000 --env-file .env ffmcp           # real league, needs FFMCP_AUTH_TOKEN too
```

Slim, non-root, streamable HTTP only: stdio does not make sense across a container boundary.

A live-mode server refuses to start over HTTP without `FFMCP_AUTH_TOKEN` set (see
`.env.example`) — with no auth layer, an internet-reachable endpoint would let anyone who found
the URL read your real league through the tools. Every request must then carry it back as
`Authorization: Bearer <token>`. Demo mode skips this: it only ever serves synthetic fixtures,
so an open demo endpoint is harmless and is the easiest way to let someone try the server
without configuring anything.

## Deploying publicly for free

[Render](https://render.com)'s free web service tier needs no credit card, builds straight from
this repo's `Dockerfile`, and gives you a public HTTPS URL. The tradeoff: a free instance sleeps
after 15 minutes idle and takes 30-50s to wake on the next request, which is fine for a tool an
agent calls occasionally.

1. Push this repo to GitHub (or use your fork).
2. On Render: **New → Web Service**, connect the repo, environment **Docker**. Render detects
   the `Dockerfile` automatically; leave the build/start commands blank.
3. Set environment variables under the service's **Environment** tab:
   - `FFMCP_MODE=live`, `FFMCP_LEAGUE_ID`, and (private leagues only) `FFMCP_ESPN_S2` /
     `FFMCP_SWID` — same as `.env.example`.
   - `FFMCP_AUTH_TOKEN` — generate one locally with
     `python -c "import secrets; print(secrets.token_urlsafe(32))"` and paste it in. The server
     will not start without this in live mode.
   - Leave `PORT` alone; Render injects it and the container's entrypoint binds to
     `$PORT` automatically (falling back to 8000 only when it's unset, e.g. a plain local
     `docker run`).
4. Deploy. Point any Streamable-HTTP MCP client at `https://<your-service>.onrender.com/mcp`
   with header `Authorization: Bearer <your token>`.

Rotate the token (just change the env var and redeploy) if it ever leaks — there is no session
or expiry on it otherwise.

## License

MIT

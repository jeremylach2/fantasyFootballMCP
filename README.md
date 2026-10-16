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
  "playoff_odds_delta": 0.7,
  "win_probability": 34.7,
  "strategy": null,
  "swaps": [
    { "bench": "Tariq Whitlock", "starter": "Marcus Moreau",     "slot": "WR",   "gain": 10.7, "reason": "+10.7 projected points" },
    { "bench": "Devin Obi",      "starter": "Elijah Villanueva", "slot": "FLEX", "gain":  3.5, "reason": "+3.5 projected points" },
    { "bench": "Soren Quintero", "starter": "Dominic Okafor",    "slot": "RB",   "gain": -3.4, "reason": "Soren Quintero is OUT" }
  ],
  "caveats": []
}
```

Three swaps and this week's win probability, about 150 tokens. The alternative, handing the model sixteen players, their
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

Fifteen tools, four resources, three prompts. All output below is verbatim from the demo league.

| Tool | Does |
|---|---|
| `get_my_team` | My roster with projections, injuries, byes and lineup locks; a past week's recap or a future week's byes. |
| `optimize_lineup` | The lineup most likely to win this week's matchup, the swaps to get there, and what to watch. |
| `analyze_matchup` | A week's matchup: win probability (live, once games start), edges, and the playoff stakes. |
| `find_trades` | Trades both lineups gain from and the market calls fair, priced over the rest of the season. |
| `evaluate_trade` | One proposed trade from both sides, and whether the other manager would say yes. |
| `find_waiver_targets` | Free-agent moves ranked by rest-of-season value, drop included; IR stashes; handcuffs. |
| `roster_planner` | Where the bye weeks bite, and which bench players I could drop at least cost. |
| `buy_low_sell_high` | Players scoring ahead of or behind their workload. |
| `simulate_season` | Simulate the rest of the season for playoff odds and seeding. |
| `league_standings` | Standings with the same simulated playoff odds, and strength of schedule. |
| `power_rankings` | Teams by all-play record, with luck and lineup accuracy. |
| `league_activity` | The league's latest adds, drops and trades. |
| `player_report` | One player in depth: range, second opinion, injury return, value, workload, market. |
| `compare_players` | Which of several players my lineup starts, and in which slot. |
| `projection_accuracy` | How accurate ESPN has been in this league, against a second source. |

Plus four resources (`ffmcp://league/settings`, `ffmcp://league/teams`, `ffmcp://glossary`,
`ffmcp://team/{team_id}/roster`) that carry stable reference data with long cache TTLs instead of
being re-fetched as tool calls, and three prompts (`weekly_checkin`, `trade_workshop`,
`playoff_push`) that chain several tools into one routine.

### Who should I start?

`get_my_team` shows the lineup as ESPN has it set. Note the running back ruled out and the
receiver on bye, both still in the lineup: it is Sunday morning and nobody has touched it.

```
Week 8 — Team Alpha (3-4-0)
Injuries: S. Quintero (O), L. Novak (I)
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

`optimize_lineup` is the response at the top of this README: +10.8 projected points, a 34.7%
chance of beating this week's opponent, and +0.7 points of playoff odds. On a real Sunday it also
does three things the demo cannot show. A player whose game has kicked off stays where he is, in
the lineup or on the bench, because ESPN has locked him. A questionable player ESPN has projected
for zero while Sleeper expects him to play gets a caveat instead of a silent benching
(`Saquon Barkley (Questionable, hamstring): ESPN projects 0, Sleeper 10.7; ...`). And when the
flex holds an early game while a later one sits at a position slot, it says to swap their slots,
which keeps a late swap open. `analyze_matchup` prices the week:

```
Week 8 (proj): Team Alpha 106.1 vs Team India 118.6 — 35% win prob
Biggest edges:
  RB: Team India +25.9
  K: Team Alpha +8.8
  WR: Team Alpha +4.8
Swing player: D. Rios (WR GB, ±12.6 pts)
Stakes: win -> 14% playoff odds, loss -> 2% (12-pt swing)
```

`Stakes` is one simulation of the season split by how this week's game went, so it answers "how
much does this game matter" rather than "how likely am I to win it". It is the same simulation
`simulate_season` and `league_standings` report, so the two halves average back to the 6.4%
those tools show. In the live 16-team league in week 6, the same line read "win -> 78%, loss ->
49%": a 29-point game. Once kickoff passes, a `Live win prob` line adds the score so far to each
starter's projection for the share of his game still to play.

**Win probability, not points.** `optimize_lineup` maximises the chance of beating *this*
opponent, which is not always the same lineup as the most projected points. With your score and
theirs both roughly Normal, P(win) = Φ((μ − μₒ) / √(σ² + σₒ²)): an underdog raises it by
raising σ, a favourite by lowering it. So the optimizer re-runs the exact matching with each
player scored `mean + tilt × sd` for a handful of tilts, keeps whichever lineup truly wins most
often, and explains itself in `strategy` whenever that costs projected points. An honest note on
how often that happens: in a real 16-team league in week 3 it changed **none** of 16 lineups.
Spread is nearly proportional to projection, so tilting rescales two same-position players
together and cannot reorder them; it bites only when two close players' spreads genuinely
differ, which mostly means one projection the sources dispute. The feature is cheap, correct,
and occasionally decisive, and it is described here as exactly that.

### Should I make this trade?

`find_trades` enumerates 1-for-1, 2-for-1 and 1-for-2 packages across every other roster,
**keeps only the ones both lineups gain from and the trade market calls fair**, and ranks them by
what they do to *my* playoff odds:

```
Team Bravo [lucky] (id 2): give Q. Hargrove, M. Moreau / get C. Castellan — me +18.8pts, them +15.4pts, +2.3% odds — mkt 682 vs 824 — Started pts/wk ROS: RB +8.5, WR -3.2, QB -2.7.
Team Bravo [lucky] (id 2): give L. Petrov, E. Villanueva / get L. Castellan — me +14.5pts, them +23.2pts, +1.6% odds — mkt 1,034 vs 1,278 — Started pts/wk ROS: RB +10.4, QB -5.5, WR -2.9.
Team Bravo [lucky] (id 2): give Q. Hargrove, M. Moreau / get T. Osei — me +13.0pts, them +20.4pts, +1.4% odds — mkt 682 vs 837 — Started pts/wk ROS: WR +4.5, QB -2.7.
```

The thesis of the trade module is Quentin Hargrove. He is a 19.4-point quarterback, on a roster
that already starts a 22.6-point quarterback, in a league with one QB slot. He is worth **zero**
to Team Alpha, because he never plays, and several points a week to Team Bravo, who start a
12.9-point quarterback. *What a player is worth is a property of the roster, not of the player*,
and that asymmetry is the only reason trades happen. `marginal_value` measures it as the change in
a roster's **optimal lineup** from adding or removing him, which is exactly why the optimizer had
to be both correct and fast.

But managers do not do that arithmetic. They anchor on names and on what players trade for,
and the earlier version of this table proposed exactly the trades they decline: in a real league
its top idea was a backup quarterback (market value 267) for Mike Evans (3,190), marked
"accept". So each side is also priced by the trade market
([FantasyCalc](https://fantasycalc.com) redraft values for the league's size, quarterback count
and reception scoring), shown as `mkt` (what I give vs what I get), and an offer is dropped when
the partner gives up more than 1.25 times what they receive. That threshold is a judgement, not a
fit: there is no record of declined offers to fit it on. Here the market turned "Hargrove for
Caleb Castellan" into "Hargrove and a receiver for Castellan", which is the offer a person might
actually accept. `evaluate_trade` says the same about the bare one-for-one:

```
{
  "verdict": "lean_accept",
  "my_value_delta": 23.4,
  "partner_value_delta": 13.3,
  "my_playoff_odds_delta": 3.0,
  "positional_impact": "Started pts/wk ROS: RB +6.0, QB -2.7.",
  "partner_likely_accepts": false,
  "market": "Market: you give 376, get 824 (FantasyCalc); they give up more than 1.25x what they get, so expect a decline.",
  "risks": [],
  "deadline": null
}
```

`verdict` is whether *I* should want it; `partner_likely_accepts` is whether they will.

Points are priced **week by week over the rest of the season**, not as this week's value times
the weeks left: each player is absent in his own bye week, later weeks use his season-average
projection rather than this week's matchup, and the playoff weeks count in proportion to each
side's simulated playoff odds. That reaches the search's cheap first stage too, which screens at
season rates: screened by this week's projection, a star on his bye week was worth nothing to
anyone and could never be a target. Kickers and defenses are left out unless asked for by name,
a trade that would lower my playoff odds is never suggested, offers that are the same core swap
with a different sweetener are collapsed into one, the league's trade deadline is counted down,
and the partner's tag (`[lucky]`, `[unlucky]`, `[inattentive]`, from `power_rankings`) says what
their record is hiding. An offer only one side gains from is a fleece and is filtered out rather
than ranked low. The filters are the feature.

### Who should I pick up?

`find_waiver_targets` ranks complete moves (the add *and* the drop) by what they do to **this
roster's rest of season**, not by projection, and not by how many people are adding them:

```
PLAYER           POS  TM  PROJ  ROS   VAL   DROP
D. Ivanov        RB   CIN 12.4  +21   +4.0  T. Moreau
C. Fairbanks     WR   KC  13.1  +18   +2.7  T. Moreau
ARI D/ST         D/ST ARI 8.8   +13   +2.1  T. Moreau
C. Brennan       K    JAX 6.2   +6    +0.0  T. Moreau
```

Caleb Fairbanks is the higher projection. Devin Ivanov is the better pickup, because this roster
is thin at running back and deep at receiver. `ROS` is the whole move over every remaining week,
this one included; `VAL` is this week's part of it. Ranking by this week was the old behaviour,
and in a 16-team league it was useless: almost no free agent beats a starter this week, so it
recommended two kickers and a defense, the first worth **-24** for the season, while the moves
that matter (bye cover, a back returning from injury) never appeared.

The drop is whichever player the season misses least, tried one at a time through the whole
remaining calendar. Every simpler rule tried was wrong in a way a manager would notice. By this
week's projection it cut the roster's hottest receiver because he was on bye; by lowest season
rate it cut the backup defense for every pickup, because defenses score least, even while the
roster carried a spare kicker. Now a kicker pickup cuts the spare kicker.

Injured-reserve free agents are listed separately as stashes, valued from when they are projected
back (Sleeper's week-by-week outlook, which ESPN does not have: its season rate for anyone on IR
is zero, which priced one at -91.7 for the season). For `position="D/ST"` a `Matchups` line shows
each defense's next opponents' implied points, as context. And when the roster starts a lead
running back, a `Handcuffs:` line names the teammate who inherits his work if he goes down and
where that player is: free agent, your bench, or another roster.

### Where do the byes bite?

`roster_planner` answers the two questions waivers raise: which weeks byes take regular starters
out of the lineup, and who on the bench could go.

```
Bye weeks ahead (typical week 110.1):
  Wk 8: T. Whitlock out -> 109.5 (-0.6)
  Wk 9: L. Santoro out -> 104.0 (-6.1)
  Wk 10: I. Adeyemi out -> 108.9 (-1.2)
  Wk 11: M. Moreau out -> 109.8 (-0.3)
  Wk 12: L. Petrov out -> 106.9 (-3.2)
  Wk 13: S. Quintero, T. Prescott out -> 97.9 (-12.2)
Drop candidates (rest-of-season cost, cheapest first):
PLAYER           POS  TM  ROS    WHY
T. Moreau        RB   KC  0.0    never starts
D. Holloway      WR   BAL 0.0    never starts
D. Obi           TE   DET -6.9   only starts to cover L. Santoro's wk-9 bye
Q. Hargrove      QB   IND -19.4  only starts to cover L. Petrov's wk-12 bye
```

In the live league this is the tool that found week 11 (quarterback and both starting receivers
on bye, 25 points below a normal week) and said the spare kicker "only starts to cover C.
Boswell's wk-9 bye".

### Who is due to regress?

Points are volume times efficiency times touchdown luck, and only volume holds up week to week.
`buy_low_sell_high` prices every player-game by its opportunity alone (targets, carries, air
yards, pass attempts) as **expected points**, and treats the gap to actual points as luck:

```
Sell high (yours, scoring above their workload):
PLAYER           POS  TM  OWNER          G  SNAP XPPG  PPG   LUCK
T. Whitlock      WR   SEA Team Alpha     7  88%  7.9   18.3  +10.4
Hold, do not sell low (yours, due to improve):
PLAYER           POS  TM  OWNER          G  SNAP XPPG  PPG   LUCK
D. Rios          WR   GB  Team Alpha     6  62%  16.4  11.9  -4.4
L. Santoro       TE   NO  Team Alpha     7  88%  14.1  9.7   -4.4
S. Quintero      RB   DAL Team Alpha     7  88%  11.7  7.4   -4.3
M. Moreau        WR   NYG Team Alpha     7  88%  10.3  6.9   -3.4
Buy low (other rosters):
PLAYER           POS  TM  OWNER          G  SNAP XPPG  PPG   LUCK
T. Osei          WR   KC  Team Bravo     6  88%  19.3  10.4  -8.9
Q. Vasquez       WR   JAX Team Charlie   7  88%  16.2  9.4   -6.8
J. Caldwell      WR   GB  Team India     6  35%  15.1  10.2  -4.9
C. Sarsgaard     WR   NO  Team Golf      7  88%  10.7  6.2   -4.6
E. Brennan       WR   TEN Team Bravo     6  88%  13.1  9.0   -4.1
Buy low (free agents):
PLAYER           POS  TM  OWNER          G  SNAP XPPG  PPG   LUCK
A. Ashworth      TE   NO  FA             7  62%  9.2   6.1   -3.1
Injured, so left out of the buy-low lists: D. Ellington (Injury Reserve)
```

That is a claim about the future, so it was tested on one: fit on 2024, scored on 2025.
Expected points per game through week 4 predicted the next six weeks with a mean absolute error
of **3.44** points, against **3.69** for actual points per game. Players 3+ points per game below
their expected points went on to score **+3.5** more per game; players 3+ above gave back
**2.9**. The ±3 threshold is where that regression was measured, and labels wait for three games
(before that, the tool shows the biggest gaps as a watch list, and says so). A buy-low is only
listed if he would start for somebody, above replacement at his position or an upgrade on your own
lineup, rated at the better of his projection and his volume: in a one-quarterback league, four
free-agent quarterbacks running below their volume are four players nobody will start.

### One player, in depth

```
Damon Rios — WR GB, week 8
Projected: 15.9 pts (range 0.0-32.1, p10-p90)
Second opinion: Sleeper 8.7; sources disagree by 45%, so the range is wider
Status: healthy
Value over replacement: +3.7 pts/wk, +26.4 pts rest of season (7.1 wks, byes and playoff odds counted)
Usage (6 g): 62% snaps (last 62%), 9.6 tgt/g (27% share), expected 16.4 vs actual 11.9 pts/g -> buy low
Game: GB implied 22.8 pts (+6.0 vs NYG, total 51.5)
Market: 96% owned, trade value 934 (FantasyCalc)
```

`Game` is the betting market's implied team total (total ÷ 2 − spread ÷ 2), from The Odds API
when a key is configured. It is context, never an input: ESPN's projection already prices the
matchup, and adding Vegas on top would count the same information twice.

Two cases get words instead of numbers. When ESPN zeroes a questionable player that Sleeper
expects to play, the report says `ESPN projects 0 (expects him out); Sleeper 10.7. Check final
inactives 90 min before kickoff.`, and his range is scaled from Sleeper's number: a spread
proportional to zero had printed "0.0-2.6" for a player who might start, beside a
"sources disagree by 1065%". And a player on injured reserve gets an `Outlook` line from his
week-by-week projections (`back week 9 (Sleeper), 6.0 pts/g after`, or `no return projected`).

### Will I make the playoffs?

`simulate_season` runs a vectorized Monte Carlo over the remaining schedule:

```
10,000 sims, weeks 8-14:
TEAM               REC     MEANW  PLAYOFF%  TITLE%
Team Echo          6-1     10.5   95.3%     27.2%
Team Bravo         6-1     9.9    86.8%     17.5%
Team Foxtrot       5-2     9.1    80.4%     22.9%
Team Hotel         5-2     8.3    53.8%     8.2%
Team Charlie       4-3     7.8    48.1%     13.4%
Team India         2-5     6.2    21.2%     7.1%
Team Juliet        2-5     5.5    7.3%      2.0%
Team Alpha         3-4     5.8    6.4%      1.5%
Team Delta         1-6     3.6    0.4%      0.1%
Team Golf          1-6     3.4    0.3%      0.1%
```

Team Alpha is 3-4 and behind a 2-5 team in playoff odds. That is correct, and the reason to
simulate rather than extrapolate a record: Alpha's roster projects worse, and seven weeks is
enough for that to matter more than one game of standings.

`league_standings` shows these same numbers, run once and cached per league state, and
`analyze_matchup`'s stakes come from the same model. An earlier version had three sources
(ESPN's own percentage in the standings, this simulation, and a third in the stakes line), and
gave one live team 67.6%, 93.8% and 88-97%. A manager who sees three different numbers for his
own team stops trusting all of them.

### Who is actually good?

A record is a noisy measure of a team in head-to-head fantasy, because the schedule decides who
you play on the week you score 140. `power_rankings` scores every team against *every* other
team every week (all-play), calls the gap to actual wins luck, and grades each manager's
lineup decisions separately from the dice:

```
Power rankings after 7 weeks (by all-play record):
RK  ID  TEAM             W-L   ALLPLAY LUCK  PPG    LINEUP% BENCH
1   5   Team Echo        6-1   53-10   +0.1  126.8  92%     21.1
2   2   Team Bravo       6-1   42-21   +1.3  117.7  100%    9.8
3   8   Team Hotel       5-2   40-23   +0.6  114.6  100%    14.7
4   6   Team Foxtrot     5-2   36-27   +1.0  115.3  100%    14.6
5   9   Team India       2-5   35-28   -1.9  111.3  100%    19.6
6   3   Team Charlie     4-3   29-34   +0.8  109.8  100%    10.0
7   1   Team Alpha       3-4   28-35   -0.1  96.7   100%    19.7
8   10  Team Juliet      2-5   23-40   -0.6  102.2  100%    15.1
9   4   Team Delta       1-6   16-47   -0.8  93.6   100%    16.9
10  7   Team Golf        1-6   13-50   -0.4  93.9   100%    14.3
Team Echo: inattentive
Team Bravo: lucky
Team Foxtrot: lucky
Team India: unlucky
```

`LINEUP%` is projected points started over the best lineup *by the projections of the day*. The
obvious alternative, actual points over the best lineup in hindsight (the `BENCH` column, as
points per game), was measured on a real league and is mostly noise: a manager with 99.8%
lineup accuracy scored 83% of hindsight-optimal, because nobody knows which receiver will boom.
Lineup accuracy behaves like a trait instead: the same manager lagged the league in both 2025
and 2026. Team India, 2-5 with the fifth-best all-play record, is the team to trade with.

### How far should I trust the projections?

```
Projection error in this league, weeks 1-7 (mean absolute error, pts):
POS  N     ESPN  SLEEPER AVG   BIAS
QB   135   5.62  5.68    5.62  +0.03
RB   329   4.76  4.85    4.80  +0.12
WR   335   5.35  5.33    5.33  -0.70
TE   132   3.86  3.84    3.84  -0.04
K    67    3.25  3.30    3.26  -0.14
D/ST 68    6.05  6.15    6.09  +0.14
ALL  1066  4.93  4.97    4.94  -0.18
ESPN is closer by 0.03 pts/player-week. Differences under ~0.3 are noise at this sample size; where the sources disagree, trust both less.
Biggest disagreements on your roster this week: D. Rios 15.9 vs 8.7
```

## Measured, not assumed

This project started from a hunch, that ESPN's projections were the problem, and a list of
features that followed from it. Before building any of them, every one was tested against a real
16-team league's full 2025 season (about 3,000 player-weeks of ESPN projection beside actual
score) and, for expected points, against nflverse play-by-play for 2024 and 2025. Half of them
survived. `scripts/calibrate.py` reproduces every number here and every fitted constant in
`domain/`.

| Idea | Result | So |
|---|---|---|
| ESPN is the weak link; blend in a second source | Sleeper MAE 5.68 vs ESPN 5.68; their average 5.66 | Not blended. A second opinion, shown. |
| Where sources **disagree**, projections are worse | Most-disputed 5% missed by 2.5× the least-disputed half | Kept: disagreement widens a player's range |
| Correct each player by his own recent misses | Worse next-half error at every shrinkage strength | Not done. Last month's miss is noise. |
| Per-player spread from his own history | No better than his position's, at any shrinkage | Not done |
| The old variance model (hand-picked, ×1.6 "correlation") | Lineup sd 33 vs a measured 22.4 | Replaced by measured spreads, ×0.92 |
| Opportunity predicts better than points | MAE 3.44 vs 3.69, out of sample | Kept: `buy_low_sell_high` |
| Hindsight lineup efficiency grades managers | Mostly luck (99.8% accurate, 83% efficient) | Replaced by lineup accuracy |
| Chase variance as an underdog | Correct, but changed 0 of 16 real lineups in week 3 | Kept, and documented as rarely decisive |
| Simulate each week on its own projection (byes, season rates) | Playoff-odds Brier 0.2034 vs 0.2065 for the old hold-this-week-constant model, 116 team-predictions from weeks 4-10 of 2024-25 | Kept; the gain is real but small in aggregate |
| A per-season team-strength draw | Teams' rest-of-season averages miss their projections by sd 10.7/wk beyond weekly noise; with it, 0 of 116 90%+ calls missed, against 4 without | Kept: `TEAM_STRENGTH_SD = 10.7` |

The playoff-odds backtest deserves its caveats said plainly. Mid-season playoff prediction in
these leagues is hard: every model scores around 0.20 against 0.247 for a flat 55%, and with 29
teams per starting week the differences between models at any one week are noise (the old model
did slightly better from weeks 4 and 6, the new one from 8 and 10). The reason to change it was
never the aggregate score. It was what the old model did to individual teams in a live league,
where three starters on bye *this* week were treated as on bye *every* week: one 3-2 team went
from ESPN's 54% to 7.9%. In a test league, emulating that model takes a 50% team with three
starters on bye this week to 0.1%; the shipped model has it at 40.5%, one game's worth lower. ESPN's own historical playoff
percentages are not available, so the backtest compares models with each other, not with ESPN.

The variance change matters most, even though it is the least visible. The old model
overstated every lineup's weekly spread by nearly half (33 points against a real 22), which pulled
every win probability toward 50% and understated how much a better lineup, trade or pickup was
worth. Its one-standard-deviation band caught 86% of real team-weeks; a calibrated band should
catch 68%, and the measured model's catches 69%.

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
most four times a day and reduced to an id→`{name, pos, team, injury}` index *on write*. ESPN
data is cached with TTLs that follow how fast the underlying truth moves: the season's pairings
are fetched once a day (calling ESPN's scoreboard per remaining week had downloaded the entire
schedule nine times per tool call), free agents for ten minutes, and the league itself every ten
minutes, or every minute while a game is in progress. If a refresh fails, the last good copy
keeps serving. Everything is projected into narrow domain models before anything is returned.
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
| `get_my_team` | compact | 176 | 134,689 | ~33,672 |
| `get_my_team` | standard | 237 | 134,689 | ~33,672 |
| `get_my_team` | full | 263 | 134,689 | ~33,672 |
| `optimize_lineup` | - | 375 | 134,689 | ~33,672 |
| `analyze_matchup` | compact | 62 | 134,689 | ~33,672 |
| `analyze_matchup` | standard | 78 | 134,689 | ~33,672 |
| `find_trades` | - | 134 | 134,689 | ~33,672 |
| `evaluate_trade` | - | 94 | 134,689 | ~33,672 |
| `find_waiver_targets` | compact | 66 | 134,689 | ~33,672 |
| `find_waiver_targets` | standard | 81 | 134,689 | ~33,672 |
| `find_waiver_targets` | full | 91 | 134,689 | ~33,672 |
| `roster_planner` | - | 156 | 134,689 | ~33,672 |
| `buy_low_sell_high` | - | 302 | 134,689 | ~33,672 |
| `simulate_season` | - | 142 | 134,689 | ~33,672 |
| `league_standings` | compact | 175 | 134,689 | ~33,672 |
| `power_rankings` | - | 212 | 134,689 | ~33,672 |
| `league_activity` | - | 48 | 134,689 | ~33,672 |
| `player_report` | - | 109 | 134,689 | ~33,672 |
| `compare_players` | - | 52 | 134,689 | ~33,672 |
| `projection_accuracy` | - | 143 | 134,689 | ~33,672 |

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
vectorized numpy Monte Carlo. Each team's score in each remaining week is drawn from a
distribution whose mean is its optimal-lineup projection **for that week**: this week's
projections and injuries for this week; every player healthy at his season rate, minus whoever is
on bye, for every week after; and the playoff weeks the same way, summed for two-week rounds.
That is the same definition of "a week" (`domain/schedule.py`) that trade and waiver values use,
so the tools cannot disagree about who is on bye. The variance comes from
[`domain/variance.py`](src/ffmcp/domain/variance.py): per-position spreads measured on a real
league's season, widened for players whose projection sources disagree (see
[Measured, not assumed](#measured-not-assumed)). On top of the weekly noise, each simulated
season draws one team-strength offset per team, held all season: projections are not the truth,
and a team's true level is uncertain in a way that does not average out week to week. Leagues
that play a median game every week, seed division winners first, or play two-week playoff rounds
are simulated that way.

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
| **Tool annotations** | every `mcp/tools_*.py` | `read_only_hint=True` on all fifteen tools, asserted by `tests/mcp/test_annotations.py` |
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
| `FFMCP_ODDS_API_KEY` | — | optional; [The Odds API](https://the-odds-api.com) free key for betting lines (`ODDS_API_KEY` also accepted) |
| `FFMCP_CACHE_DIR` | `~/.cache/ffmcp` | |
| `FFMCP_SIMS` | `10000` | Monte Carlo iterations |
| `FFMCP_FIXTURE` | `demo` | which committed fixture set demo mode serves: `demo` or `snapshot` |

Secrets are typed `SecretStr`, and `errors.py:redact()` strips anything cookie-shaped, and any
`apiKey=` query parameter, from every error message and log line at the boundary (unit-tested
against a fake `espn_s2`, a `SWID` GUID and a key-bearing URL).

Every signal beyond ESPN is optional and degrades to a one-line caveat rather than an error:
no odds key means no `Game` line; Sleeper or nflverse being down means ranges from position
spreads alone, or no usage section; FantasyCalc being down means trades are not checked for
market plausibility, and the response says so. A tool that refused to set a lineup because GitHub was slow
would be worse than one that sets it with slightly less information and says so. See the [Quickstart](#quickstart) above for where to find ESPN cookies.

## Testing

```bash
uv run poe check    # ruff lint + format, mypy --strict, pytest, benchmark freshness
```

325 tests, no network anywhere: fixtures for ESPN, `httpx2.MockTransport` for Sleeper, nflverse,
The Odds API and FantasyCalc. Beyond the usual, the suite carries a few tests that exist to turn claims into
facts: the layering purity check, the greedy-vs-optimal counterexample, the common-random-numbers
comparison, the token budgets, a star on his bye week never being the roster cut, one good
matchup never being extrapolated across a season, and a hygiene test asserting no fixture
contains a real league id or anything cookie-shaped.

The demo league itself is synthetic (invented players, invented projections), generated
deterministically by `scripts/make_demo_fixture.py` and committed, so that shipping a runnable
demo does not mean publishing anyone's real roster. Its history, usage, second-opinion
projections and betting lines come from `scripts/make_demo_insights.py`, derived from that league
so the two agree: each team's starters add up to the scores its standings were built from.

The suite once passed in full while every one of the bugs a live-league audit found was live,
because the synthetic league differed from real ESPN data in exactly the ways that hid them: ESPN
projects a bye as `0.0`, not nothing; injured free agents carry a season rate of zero; released
players sit on no NFL team; a 16-team league's waiver wire is thin. So there is a second fixture
set, `snapshot`: a real 16-team league at week 6, anonymized by `scripts/record_fixtures.py`
(team names replaced, league id replaced, no owner data), with everything the tools read, from
rosters and the full schedule to market values and injured players' outlooks.
`tests/mcp/test_snapshot_regressions.py` reproduces each audit finding on it, and the demo fixture
now projects byes at `0.0` too, so the README's own transcripts run the live code path.

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


# Dashboard — Next Session

Continuation notes for whoever picks this up. Phase 0/1 history + the
substrate they built; Phase 2 retrospective (shipped); Phase 3 candidates
(what's next); architectural rules + anti-patterns that stay in force.

---

## Where we are now (2026-05-26)

The dashboard started as a single 3,471-line
`brain_dashboard_standalone.py` with Python + HTML + CSS + JS all in one
triple-quoted string. Three phases of refactor have shipped:

**Phase 0** (`26b9a88`) — monolith → package + backdate audit. Split into
the `dashboard/` package; `@safe_query` decorator collapsed boilerplate;
identity chips on traces; episodic-refs section; aspect taxonomy view;
S2 Healer cards; loud-by-default sweep; disconnection contract test.

**Phase 1** (`42225ec`) — frontend substrate. `static/css/` (vars/base/
components/types) and `static/lib/` (api/bus/poll/dom/types) — see the
Architecture section. `app.js` became an ES module. Every fetch through
`api.*`, every poll through `poll.register`.

**Dashboard self-monitoring** — `dashboard/log.py` ring buffer behind
`warn()`. `/api/dashboard-errors` exposes it. Logs tab has Errors /
Daemon / Dashboard sub-feeds — the Dashboard sub-feed shows both server
warnings AND browser-side `window.onerror` / `console.error` /
`unhandledrejection` entries. When a panel goes blank, look here first.

**Phase 2** (`P2.1` through `P2.17`) — feature build on top of the
substrate. See the **Phase 2 — what shipped** section below.

---

## Architecture, current

```
dashboard/
  brain_dashboard_standalone.py   # 30-line entry shim
  server.py                       # HTTP routes only — no SQL
  daemon_client.py                # TCP client (ONE sanctioned bridge)
  db.py                           # path resolution + ro_connect helper
  clock.py                        # utc_cutoff + iso_window_around
  log.py                          # warn() + recent() ring buffer
  query.py                        # @safe_query decorator
  contract.py                     # Prometheus envelope helpers
                                  # (helpers ready; NOT YET adopted)
  queries/                        # 12 modules, one per data source
    recalls.py / encoding.py / s2_runs.py / traces.py /
    errors.py / system.py / sessions.py / stats.py /
    explorer.py / graph.py / aspects.py / insights_scanner.py /
    _meta.py
  static/
    index.html
    style.css                     # LEGACY — being migrated to
                                  # components.css tab-by-tab. New code
                                  # reaches for vars/base/components/types
                                  # first. Loaded AFTER components.css so
                                  # legacy rules win on same-specificity
                                  # collisions — bump new rules by stacking
                                  # classes (.nd-section.nd-section--source)
                                  # rather than !important.
    css/
      vars.css                    # design tokens
      base.css                    # reset + tabs + stats-bar + toolbar
      components.css              # .card .badge .chip .notif .btn .code
                                  # + nd-* + insights-* + recall-entry
                                  # + graph-pin-chip + trace-event--target
      types.css                   # .type--<name> single source
    lib/
      api.js                      # fetch wrapper + named endpoints
      bus.js                      # ~40-line pub/sub
      poll.js                     # one 1Hz scheduler (visibility-gated)
      dom.js                      # el() + escapeHtml + localTime +
                                  # identityChip + relativeTime
      types.js                    # TYPE_COLORS map for the 3D graph
      scales.js                   # SCALE_COLORS + scaleColor() helper
      node_detail.js              # node-detail panel (el()-builder rewrite)
    tabs/                         # per-tab modules — each exports
                                  # {init, activate, deactivate}
      live.js                     # decoding + encoding feeds, layout
                                  # picker, insights panel, recall card
                                  # hover/pin
      graph.js                    # 3D ForceGraph + search highlighter
                                  # + pin chip + WebGL lifecycle
      explorer.js                 # node search + type filter
      logs.js                     # errors + daemon + dashboard sub-feeds
                                  # + client-error capture wrap
      health.js                   # system status + aspects + insights
      traces.js                   # trace chain rendering + flash target
    app.js                        # 90-line bootstrap — switchTab,
                                  # page-chrome polls (stats/sessions),
                                  # inline-handler window.* exposures,
                                  # init-once guard for tab modules
tests/
  test_dashboard_disconnection.py # invariants: no servers.*/hooks.*
                                  # imports, all SQLite connects = ro
```

Run: `python3 dashboard/brain_dashboard_standalone.py` →
`http://127.0.0.1:47303`

Lifecycle contract (every `tabs/*.js` module):
- `init()` — called ONCE on app boot. Wires polls + bus subs.
- `activate()` — called when the tab becomes visible. Lazy-loads data.
- `deactivate()` — called when leaving. Most modules no-op; `poll.js`
  auto-gates on `activeWhen` + `document.hidden`.

`app.js` guards `init()` with an `_inittedTabs` Set so a future
double-init can't duplicate subscriptions.

---

## Conventions established

**Architectural rules** (in priority order):

1. **The dashboard never imports from `servers.*` or `hooks.*`.** Locked
   by `tests/test_dashboard_disconnection.py`. One documented exception:
   `queries/aspects.py:_repo_seed_path()` reads the brain's
   `aspects_v1.json` — same source-of-truth, not a parallel funnel.

2. **All SQLite connections open with `mode=ro`.** Locked by the same
   contract test. Use `ro_connect()` or `@safe_query`.

3. **Loud by default.** Every `except` either logs via
   `dashboard.log.warn(component, msg, exc=e)` OR is an intentional
   inner-row silence with a comment. Silent `pass` is a bug.

4. **Time windows route through `clock.iso_window_around()`.**
   `utc_cutoff(...)` for "N hours ago" cutoffs. No hand-rolled string
   slicing on timestamps.

5. **`@safe_query(component, db_path_fn)` for single-DB queries.**
   First arg is `conn`. Errors auto-route to `warn()`. Returns default
   `[]` on failure.

6. **Frontend: ES module.** Inline `onclick="X()"` requires
   `window.X = X` at the bottom of `app.js`. Migrating to
   `addEventListener` is incremental — new code prefers attached
   listeners (see `node_detail.js`, recall card click handlers).

7. **Polling routes through `poll.register({key,interval,activeWhen,fetcher})`.**
   Never `setInterval` directly. `key` deduplicates — re-registering
   replaces the entry, no duplicate firings.

8. **Fetch routes through `api.*` named endpoints.** Never bare `fetch()`.
   The wrapper dedups concurrent identical URLs.

9. **DOM construction uses `el(tag, attrs, ...kids)` from `lib/dom.js`.**
   Strings auto-escape; raw HTML via `html('...')`. Migrating
   incrementally; `loadEncodingActivity` still has innerHTML soup and is
   the next target.

10. **Cross-module signals via `bus.publish(topic, payload)`.** Topic
    naming `<source>:<event>`. Current catalog:
    - `tab:active` — fires on switchTab with `{name}`.
    - `recall:event` — published by live.js poller; both live.js
      renderer and graph.js highlight subscribe.
    - `graph:pinned` — published by graph.js when a recall card pins
      the highlight (`{eventId}`) or unpins (`{eventId:null}`).
      live.js applies `.recall-entry--pinned` accordingly.
    - `live:layout` — fires on divider drag / layout change. graph.js
      resizes ForceGraph3D in response.
    - `insights:tick` — fires every 60s with the insights payload.

11. **`data-scale` is the contract for top-level cards only.**
    `filterByScale` selects `[data-scale]` — inner sub-rows
    (CREATED / REVISED / CONNECTED inside encoding cards) must NOT
    carry `data-scale`, or they'll get filtered when their parent
    card is visible. Class names like `.enc-entry` are overloaded
    across both layers; only the attribute is reliable.

**Naming conventions**:

- CSS classes: `.card .card--s1 .card--success`; `.badge .badge--green
  .badge--ghost-amber`; `.chip .chip--id .chip--session .chip--identity`.
  Generic primitive + accent modifier.
- Type colors: `.type--<name>`. Single source: `static/css/types.css`
  mirrored by `static/lib/types.js`.
- JS module-local globals OK at module top; cross-module state goes
  through `bus.publish`.

---

## Cross-module flash-survives-poll pattern

When module A wants to apply a visible effect to DOM owned by module B,
AND module B re-renders periodically (poll), the effect dies on each
re-render. The pattern shipped in P2.17 for trace-flash:

1. Module A (`node_detail.js`) stores pending state at module scope:
   `_pendingFlashTraceId` + `_pendingFlashUntil` (timeout).
2. Module A exports a `reapplyFlashIfPending()` function.
3. Module B (`traces.js`) calls `reapplyFlashIfPending()` at the end of
   its render loop.
4. The pending state auto-clears after the timeout window so subsequent
   B renders don't re-flash forever.

Use this when bus events feel like overkill (single producer/consumer
pair, no fan-out) and direct import is acceptable.

---

## Phase 2 — what shipped

Commits `P2.1` through `P2.17`. The original plan had 3 items; the
session expanded scope as features uncovered substrate gaps.

**P2.1 Per-tab module split.** `app.js` 1500 → 90 lines. Six tab
modules each exporting `{init, activate, deactivate}`. The split was
the prerequisite for everything else — separating concerns lets each
tab own its own polls + bus subs without leaking globals.

**P2.2 Layout pivot.** Live tab became a graph + stream split with
draggable divider, 4-orientation picker, drag-divider persistence to
localStorage. Standalone Graph tab dropped — only one ForceGraph3D
instance now, mounted in Live's left pane. Explorer + Health pushed
into a "⋯" overflow menu. Persist via `dashboard.liveSplitPct` and
`dashboard.liveLayoutMode` localStorage keys.

**P2.3 Cleanup pass.** Inline `style=` migration for several panels
(insights, health, node-detail). `node_detail.js` rewritten to `el()`
builder with section helpers. `logs.js` extracted shared
`_renderLogEntry`. `health.js` migrated inline styles to component
classes. `queries/legacy.py` deleted (dead). Endpoint cleanup:
`/api/hook-log` and `/api/assembler-comparison` removed.

**P2.4 Graph search highlighter.** Replaced the community-legend pane
with a search input that dim-by-defaults non-matches. `_searchQuery`
+ `_nodeMatches`; pan-to-first-match on Enter. Search also works
when WebGL fails (`_searchableNodes()` falls back to `graph3dData`).

**P2.5 Recall activations on graph.** Three-tier persistent
highlight: used (white, 1.0) / activation (green, 0.7) /
returned (blue, 0.4). Persistent — no decay — until next recall or
Refresh. Latest / Per-session modes. Pinned mode (added in P2.16)
locks the highlight to a specific event.

**P2.6 Insights MVP.** `queries/insights_scanner.py` with 3 rule
families: S2 silence, empty selections, error spike. `/api/insights/live`
endpoint. Panel renders into Live tab top; cards have severity / icon /
title / detail / evidence; dismiss × per page-session.

**P2.13 Persistent dim-by-default highlight + WebGL error pane.**
Replaced 5s pulse decay with the persistent model. When WebGL fails
the graph pane shows a chrome://gpu diagnostic instead of silent blank.

**P2.14/P2.15 ForceGraph3D pin to 1.80.0.** Library pinned to a
specific version after a fabrication caught by Tom (see P2.15 commit
body for the honest correction).

**P2.16 WebGL hygiene + pin-to-graph UX + GC audit.** WebGL context
leak fixes (3 vectors closed); whole-card click pin with hover
preview; dropped redundant highlight-mode dropdown; scale-filter
selector bug fix; encoding feed session filter; logs unified badge;
layout picker repositioned out of absolute over content;
`s2RenderedChains` prune + idempotency guards + central init guard.

**P2.17 Source-ref navigation.** Source-refs in node-detail get
prominent `--source` styling (accent-blue heading, tinted card, chip
button). Clicking "Open in Traces tab" navigates with session filter,
scrolls target into view, flashes gold. The flash survives the 5s
traces poll re-render via `reapplyTraceFlashIfPending`. Traces query
switched ASC→DESC + limit 200→500 (was silently dropping target
traces in busy sessions).

---

## Phase 3 candidates

These are real but uncommitted-to. Pick when you start the next
session.

### loadEncodingActivity → el() migration (cleanup debt)

`tabs/live.js:loadEncodingActivity` (~250 lines) still builds HTML via
string concatenation with inline `style=` and inline `onclick=`. Last
big inline-style holdout. Migration mirrors what `node_detail.js`
already does — `el()` builder, section helpers, addEventListener for
clicks. Drop window.* exposure for `toggleEncPrompt` /
`toggleConsolPrompt` etc. in favor of attached listeners.

### Prometheus envelope adoption

Server-side `contract.envelope_ok/error` exist but no route emits them.
Client `lib/api.js:unwrap()` handles both shapes. Migration plan: each
new endpoint uses the envelope; each existing endpoint converts when
its caller's render code is being rewritten anyway. Don't half-migrate
one route without updating its caller in the same commit.

### Dashboard self-health surface

No one's asked for it but it'd close the loop. A small panel surfacing:
- Active polls + last-fire times (debug data already in
  `poll._debug_entries()`).
- WebGL context count (estimate via probe).
- Bus topic subscriber counts (`bus._debug_topics()`).
- Memory growth indicators (DOM node count, `_eventsById` size,
  `s2RenderedChains` size).

Probably lives under Health tab as a "Dashboard diagnostics"
sub-section. Not urgent — `warn()` ring already covers most failure
visibility.

### Type-color server endpoint

`static/css/types.css` and `static/lib/types.js` mirror each other
manually. Eventual fix: `/api/type-colors` returns the canonical map
(probably derived from brain aspect taxonomy). Low priority — only
two files, both dashboard-owned.

### Static asset hardening

`server.py` serves `/static/*` from a path-checked `STATIC_DIR`.
Defends against `..` traversal but doesn't set `Cache-Control`
headers. Fine for local-dev (forces reload during refactoring); add
cache headers if/when packaging.

### Insights LLM-summarized version

Rule-based MVP shipped in P2.6. Eventual: LLM that reads the same
trace data and produces narrative anomalies. Each insight still has
`{severity, icon, title, detail, evidence:[trace_ids|node_ids]}` — the
contract doesn't change, only the producer.

---

## Anti-patterns — don't do these

These came up during P0/P1/P2 and would have rotted the codebase:

- **Don't write a new `_serve_X` method that takes raw SQL.** Add a
  function in `queries/<area>.py` with `@safe_query`; the route handler
  just calls + renders JSON. `server.py` has ZERO SQL.

- **Don't add a new `setInterval`.** Use `poll.register`.

- **Don't add a new `let global_state_X` at module top in `app.js`.**
  Module-local in a tab is fine; cross-tab state via `bus.publish`.

- **Don't open `sqlite3.connect(...)` directly.** Use `ro_connect()`.
  Contract test catches this.

- **Don't import from `servers.*`.** TCP via `daemon_send(cmd, args)`.

- **Don't write `style="background:...;color:..."` inline.** Reach for
  `.card .card--s1` etc. Add to `components.css` if missing.

- **Don't write `} catch(e) {}` or `except Exception: pass`.**
  `warn(component, 'what failed', exc=e)` or
  `console.error('[dashboard] X failed:', e)`. Silent failure is the
  bug class CLAUDE.md spends 200 words on.

- **Don't apply `addEventListener` to children inside an
  `innerHTML`-rewriting container without thinking about lifecycle.**
  Each re-render orphans the listeners and re-attaches new ones.
  Either: (a) use event delegation on the stable parent, (b) bind
  listeners on `document.createElement` results coupled to a
  bounded-eviction strategy (see recall card listeners + MAX_ENTRIES),
  or (c) use inline `onclick` (still acceptable for simple cases).

- **Don't `sel.value = X` to apply a filter when the `<option value=X>`
  may not exist yet.** It silently becomes `''`. Pass the override
  through the function signature (see `loadTraces(opts.session)`),
  then sync the dropdown after the data arrives.

- **Don't bump CSS class specificity with `!important`.** Stack
  classes on the same element (`.nd-section.nd-section--source`).
  Required because `style.css` ships after `components.css` and wins
  on same-specificity collisions.

---

## Quick reference

- **Run dashboard**: `python3 dashboard/brain_dashboard_standalone.py`
- **Run with eval brain**: `BRAIN_DB_DIR=~/AgentsContext/brain-eval
  DASHBOARD_PORT=47304 python3 dashboard/brain_dashboard_standalone.py`
- **Run contracts**: `./dev pytest tests/test_dashboard_disconnection.py
  tests/test_time_window_contract.py`
- **Inspect ring buffer**: `curl http://127.0.0.1:47303/api/dashboard-errors`
- **Clear ring buffer**: `curl 'http://127.0.0.1:47303/api/dashboard-errors?clear=1'`
- **Inspect poll registry from devtools**:
  `await import('/static/lib/poll.js').then(m => m.poll._debug_entries())`
- **Inspect bus subscriptions**:
  `await import('/static/lib/bus.js').then(m => m.bus._debug_topics())`
- **Hard-refresh assets**: Cmd+Shift+R — `<link>`'d CSS and ES module
  imports both cache aggressively in Chrome.

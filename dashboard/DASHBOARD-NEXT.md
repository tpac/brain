# Dashboard — Next Session

Continuation notes for whoever picks this up. Supersedes the old
`Dashboard-nextwork.md` (kept around for historical context but stale —
the bug list it tracked is closed).

---

## Where we are now (2026-05-25)

The dashboard was a single 3,471-line `brain_dashboard_standalone.py` with
Python + HTML + CSS + JS all in one triple-quoted string, ~30 raw SQL
queries inline, 162 inline `style=` attrs, 15 global JS variables, 9 racing
`setInterval`s. Two cleanup phases landed:

**Phase 0 (commit `26b9a88`)** — monolith → package + backdate audit.
- Split into `dashboard/` package: `server.py`, `queries/*.py` (12
  modules), `db.py`, `daemon_client.py`, `clock.py`, `log.py`, plus
  thin `brain_dashboard_standalone.py` entry shim.
- `@safe_query` decorator collapsed ~50 copies of
  `with ro_connect/try/except/return []` boilerplate.
- `clock.iso_window_around()` replaced two hand-rolled string-clamped
  timestamp window sites; fixes the midnight/hour rollover bug.
- 7 audit items: identity chips on traces, correction enrichment on node
  detail, episodic-refs section, aspect taxonomy view, S2 Healer cards,
  loud-by-default sweep, disconnection contract test.

**Phase 1 (commit `42225ec`)** — frontend substrate.
- `static/css/`: `vars.css` (design tokens) → `base.css` (chrome) →
  `components.css` (card/badge/chip/notif/btn primitives) → `types.css`
  (type-color single source).
- `static/lib/`: `api.js` (fetch wrapper + named endpoints + dedup),
  `bus.js` (pub/sub for cross-tab signals), `poll.js` (one 1Hz
  scheduler replaces 9 setIntervals), `dom.js` (escape-by-default
  `el()` builder + `escapeHtml` + `localTime` + `identityChip`),
  `types.js` (TYPE_COLORS map for the 3D graph).
- `app.js` became an ES module. Every fetch goes through `api.*`, every
  poll through `poll.register({key,interval,activeWhen,fetcher})`.
- Real bugs fixed: `activeFeed='surface'` (caused all polls to misfire
  on first load) → `'decoding'`; `% since` undefined-variable in two
  branches of errors.py; `_corrections is list not dict` for single-id
  get_node; 7 dead symbols deleted.

**Dashboard self-monitoring** (most recent commit, see git log for SHA).
- `dashboard/log.py` now has a 200-entry ring buffer behind `warn()`.
- `/api/dashboard-errors` exposes it. Logs tab has a third "Dashboard"
  sub-feed showing both server (`warn()`) and browser (`window.onerror`
  + `console.error` + `unhandledrejection`) entries.
- The 4 remaining silent `catch(e) {}` blocks in app.js now log.
- Use this aggressively next session — when a panel shows blank, look
  here first.

---

## Architecture, current

```
dashboard/
  brain_dashboard_standalone.py   # 30-line entry shim (preserves the
                                  # launch path: .claude/launch.json,
                                  # test_time_window_contract.py, eval/longmem
                                  # all hardcode this path)
  server.py                       # HTTP routes only — no SQL
  daemon_client.py                # TCP client (ONE sanctioned bridge)
  db.py                           # path resolution + ro_connect helper
                                  # (mode=ro pinned; contract test catches drift)
  clock.py                        # utc_cutoff + iso_window_around
  log.py                          # warn() + recent() ring buffer
  query.py                        # @safe_query decorator
  contract.py                     # Prometheus envelope helpers
                                  # (helpers ready; NOT YET adopted by routes)
  queries/                        # 12 modules, one per data source
    recalls.py / encoding.py / s2_runs.py / traces.py /
    errors.py / system.py / sessions.py / stats.py /
    explorer.py / graph.py / aspects.py / legacy.py / _meta.py
  static/
    index.html
    style.css                     # LEGACY — being migrated to components.css
                                  # tab-by-tab. New code should reach for
                                  # vars/base/components/types first.
    css/
      vars.css                    # design tokens (color, spacing, font, radius)
      base.css                    # reset + tabs + stats-bar + toolbar
      components.css              # .card .badge .chip .notif .btn .code
      types.css                   # .type--<name> single source for type → color
    lib/
      api.js                      # fetch wrapper + named endpoints
      bus.js                      # ~40-line pub/sub
      poll.js                     # one 1Hz scheduler
      dom.js                      # el() / escapeHtml / localTime / identityChip
      types.js                    # TYPE_COLORS (mirrors types.css)
    app.js                        # ES module — top has imports, bottom has
                                  # window.* exposures for inline handlers
tests/
  test_dashboard_disconnection.py # 2 invariants: no servers.* imports,
                                  # no non-ro SQLite connects (docstrings
                                  # may legitimately mention sqlite3.connect)
```

Run: `python3 dashboard/brain_dashboard_standalone.py` → `http://127.0.0.1:47303`

Contract tests (`./dev pytest tests/test_dashboard_disconnection.py`) lock
the architecture boundary. Don't relax them — when something needs daemon
behavior, route through `daemon_client.daemon_send` over TCP.

---

## Conventions established

**Architectural rules** (in priority order):

1. **The dashboard never imports from `servers.*` or `hooks.*`.** Locked
   by `tests/test_dashboard_disconnection.py:test_no_imports_from_servers_or_hooks`.
   If you need brain behavior, ask the daemon over TCP via
   `daemon_client.daemon_send(cmd, args)`. The ONE documented exception
   is `queries/aspects.py:_repo_seed_path()` which reads the brain's
   `aspects_v1.json` config — same file the brain reads, so it's
   consuming the same source-of-truth, not creating a parallel funnel.

2. **All SQLite connections open with `mode=ro`.** Locked by
   `test_all_sqlite_connects_are_read_only`. Use
   `dashboard.db.ro_connect(path)` or the `@safe_query(component, path_fn)`
   decorator — both pin `mode=ro` for you.

3. **Loud by default.** Every `except Exception` either logs via
   `dashboard.log.warn(component, message, exc=e)` OR is an intentional
   inner-row silence with a comment explaining why. Silent `pass` is a
   bug. The ring buffer makes loud cheap — entries don't disappear, they
   sit in the Dashboard sub-feed under the Logs tab.

4. **Time windows route through `clock.iso_window_around()`.** No
   hand-rolled string slicing on timestamps — broken on hour / midnight
   rollover. `utc_cutoff(...)` for general "N hours ago" cutoffs.

5. **`@safe_query(component, db_path_fn)` for single-DB queries.**
   First arg of the wrapped function is `conn`. Errors auto-route to
   `warn()`. Returns `default` (usually `[]`) on any failure. Don't
   re-implement the with-ro_connect/try/except shape — the decorator
   already does it.

6. **Frontend: ES module. `<script type="module" src="app.js">` in
   `index.html`.** All identifiers are module-scoped; inline
   `onclick="X()"` requires `window.X = X` mounted at the bottom of
   `app.js`. Future: migrate inline handlers to `addEventListener` so
   the window-exposure block shrinks.

7. **Polling routes through `poll.register({key,interval,activeWhen,fetcher})`.**
   Never call `setInterval` directly. Per-tab `activeWhen` predicates
   stop polling when the tab is hidden / inactive.

8. **Fetch routes through `api.*` named endpoints.** Never call `fetch()`
   directly. `api.recalls({...})` not `fetch('/api/recalls?...')`. The
   wrapper dedups concurrent identical URLs.

9. **DOM construction uses `el(tag, attrs, ...kids)` from `lib/dom.js`.**
   Strings auto-escape; insert raw HTML via `html('...')`. This is
   incrementally migrating — `app.js` still has `innerHTML += '...'`
   patterns. New code should use `el()`.

**Naming conventions**:

- CSS classes: `.card .card--s1 .card--success`; `.badge .badge--green
  .badge--ghost-amber`; `.chip .chip--id .chip--session .chip--identity`.
  Generic primitive + accent modifier.
- Type colors: `.type--<name>` (e.g. `.type--lesson`). The `.type-badge`
  base class plus the modifier. Single source: `static/css/types.css`
  mirrored by `static/lib/types.js`.
- JS module-local globals OK at module top; cross-module signals go
  through `bus.publish(topic, payload)`. Topic naming `<source>:<event>`
  (e.g. `recall:event`, `graph:activation`).
- Endpoint shapes: still inconsistent (some bare arrays, some
  `{events:[...]}`, some `{runs:[...]}`). Prometheus envelope helpers
  are in `contract.py` but no route migrated yet — see remaining work
  below.

---

## Other-session boundary

A second session is doing Phase B (episodic-references write path) +
schema v29 (hex `trace_events.id`). They're in `servers/*`, `eval/*`,
and a handful of test files. Their commits land between mine:

```
cd0c26a Phase B surface: encoder input markers + MCP source_refs schema
f088343 Schema v29 + DAL/dispatch trace_id discipline
42225ec Dashboard P1: frontend substrate
26b9a88 Dashboard refactor: monolith → package, audit + reground, P0
```

No file overlap — dashboard is purely additive in its own subtree. Their
work makes `node_source_refs` actually populate; once it does, the
"Encoded from N traces" section in the node-detail panel will start
showing real data (the endpoint already works against the empty table).

When you start next session: `git status` first, see what they pushed,
align before touching anything.

---

## Remaining cleanup debt (do as you touch each tab)

These are tracked here, NOT as a separate "cleanup phase" — Tom's rule is
**every feature includes downstream cleanup**. As you touch a tab to add
features, drag its share of this list with you.

### Frontend split (the big one)

`app.js` is still ~1500 lines, one file. The substrate (api, bus, poll,
dom) is in; the per-tab split is not. Per-tab modules planned:

```
static/tabs/
  live.js       # decoding + encoding feeds + S2 entries on Live
  graph.js      # 3D ForceGraph + legend + activations
  explorer.js   # node search + detail panel
  logs.js       # errors + daemon + dashboard sub-feeds
  health.js     # system status + aspects + insights
  traces.js     # trace chain rendering
```

Each module exports `{init, activate, deactivate}`. `app.js` becomes a
~30-line bootstrap that imports and routes.

**Trigger**: The layout pivot (Live = graph + stream side-by-side, push
Explorer + Health into a "⋯ More" overflow) — there's no clean way to
merge two tabs into one without first separating them out, so the pivot
forces the split.

### Inline styles → component classes

162 inline `style="..."` attributes in `app.js` (counted at audit time).
The primitives in `components.css` (`.card`, `.badge`, `.chip`, etc.)
cover the common shapes. Migration policy: when you rewrite a panel's
render code (which the per-tab split forces), convert its `style=`
attributes to classes. Don't do a separate "migration phase" — too easy
to break rendering.

### Prometheus envelope adoption

Server-side: `contract.envelope_ok(data, warnings=...)` and
`envelope_error(msg, error_type=...)` exist; no route emits them yet.
Client-side: `lib/api.js:unwrap(body)` handles BOTH legacy shapes and
the envelope.

**Migration plan**: each new endpoint uses the envelope. Each existing
endpoint converts when its caller's render code is being rewritten
anyway (i.e. during the per-tab split). Don't half-migrate one route at
a time without updating its caller in the same commit.

### Type-color server endpoint

`static/css/types.css` and `static/lib/types.js` mirror each other
manually. Both encode the same name → color map. Eventual fix: add
`/api/type-colors` that returns the canonical map (probably derived from
the brain's aspect taxonomy), have CSS variables and JS read from a
single place at runtime. Not urgent — the duplication is only between
two files, both of which I own.

### "Encoded from N traces" — waiting on Phase B writes

`/api/node/{id}/source-refs` works against an empty `node_source_refs`
table today. Once Phase B's write path ships (the other session is
on this), the section will populate automatically. No work needed on
the dashboard side until/unless the shape changes.

### Static asset hardening

`server.py` serves `/static/*` from a path-checked `STATIC_DIR`. It
defends against `..` traversal but doesn't set `Cache-Control` headers.
For local-dev this is fine (forces reload during refactoring); for any
hypothetical packaged distribution, add cache headers.

### Other-session schema changes the dashboard should watch

The other session is changing schemas. The dashboard's read queries are
loud-by-default — if they break, `warn()` lands in the Dashboard
sub-feed. But proactive monitoring beats reactive: when reviewing their
commits, scan for `ALTER TABLE` / `DROP COLUMN` / new tables that the
dashboard might want to surface. Recent additions worth surfacing:

- `trace_embeddings` (v27) — no dashboard view yet. Could be a "% of
  traces embedded" indicator on the Health tab.
- `node_source_refs` (v27, populating from Phase B) — endpoint exists,
  UI section exists, just waiting for writes.
- v29 hex `trace_events.id` — handled (cursor switched to ISO ts).

---

## Phase 2 — features Tom asked for

Order matters. Each pulls a tab through the per-module split + migrates
its inline styles, per the cleanup rule above.

### 1. Layout pivot — Live = graph + stream + "⋯ More" overflow

The big visual move. Live becomes a split view: 3D graph on the left
(~60% wide), activity stream on the right (~40%), so recall activations
on the graph are visible while the stream still updates. Explorer and
Health move into a "⋯" overflow menu in the tab bar.

Bus topics that come online during this:
- `tab:active` — fires on switchTab, carries the tab name. Tabs use
  this to lazy-load.
- `live:layout` — when the user drags the split divider, the graph
  needs to re-fit.

The 3D graph (`ForceGraph3D` from CDN) needs `renderer.setSize(w, h)`
+ `camera.updateProjectionMatrix()` on resize — already done in the
existing switchTab logic; just needs wiring to the new split.

### 2. Recall activations on the graph

When a new recall trace lands, light up the surfaced nodes on the 3D
graph for 5s. Pulse intensity = `used_ids` (selected by judge) brighter
than `returned_ids` (candidates).

Substrate:
- `poll.js` already polls `/api/recalls` and publishes new events.
- New bus topic: `graph:activation` with `{used_ids:[...], returned_ids:[...]}`.
- Graph tab subscribes; uses the existing `ForceGraph3D.nodeColor()` or a
  custom three.js material to apply the pulse.

Risk: ForceGraph3D from CDN doesn't expose a per-node pulse cleanly. If
it doesn't, fall back to color-flash + size-flash on the existing node
material. Document the limitation if the visual is degraded.

### 3. Insights agent MVP — rule-based

Read recent encoding/recall/error activity. Detect anomalies:
- An S2 unit hasn't fired in N hours
- A node was recalled K times but never selected → probably noise
- A community has > N members but no encoding edge in M days
- The judge success rate dropped > 30% in the last hour
- Error count spiked > 3× over the prior baseline

Implementation: `queries/insights_scanner.py` — pure derived analysis
from existing query outputs, no new tables. Renders into a "Insights"
section at the top of the Live tab.

LLM-summarized version comes later — keep rule-based for the MVP. Each
insight has `{severity, icon, title, detail, evidence:[trace_ids|node_ids]}`.

---

## Anti-patterns — don't do these

These came up during P0/P1 and would have rotted the codebase:

- **Don't write a new `_serve_X` method that takes raw SQL.** Add a
  function in `queries/<area>.py` decorated with `@safe_query`; the
  route handler just calls it and renders JSON. `server.py` should
  contain ZERO SQL.

- **Don't add a new `setInterval`.** Use `poll.register({key, ...})`.
  Inactive tabs should poll zero unless they need background badges.

- **Don't add a new `let global_state_X` at module top in app.js.**
  Module-local is fine; cross-tab state goes through `bus.publish`.

- **Don't open `sqlite3.connect(...)` directly.** Use `ro_connect()`.
  The contract test catches this but the better defense is reflex.

- **Don't import from `servers.*`.** If you need brain state, send a
  TCP command via `daemon_send(cmd, args)`. If the daemon doesn't expose
  the command yet, that's a brain-side change, not a dashboard hack.

- **Don't write `style="background:...;color:..."` inline.** Reach for
  `.card .card--s1` etc. If the primitive doesn't exist for what you
  need, add it to `components.css` — but try the existing ones first.

- **Don't write `} catch(e) {}` or `except Exception: pass`.** Use
  `warn(component, 'what failed', exc=e)` (Python) or
  `console.error('[dashboard] X failed:', e)` (JS). Silent failure is
  the bug class CLAUDE.md spends 200 words on.

---

## Quick reference

- **Run dashboard**: `python3 dashboard/brain_dashboard_standalone.py`
- **Run with eval brain**: `BRAIN_DB_DIR=~/AgentsContext/brain-eval
  DASHBOARD_PORT=47304 python3 dashboard/brain_dashboard_standalone.py`
- **Run contracts**: `./dev pytest tests/test_dashboard_disconnection.py
  tests/test_time_window_contract.py`
- **Inspect ring buffer**: `curl http://127.0.0.1:47303/api/dashboard-errors`
- **Clear ring buffer**: `curl 'http://127.0.0.1:47303/api/dashboard-errors?clear=1'`
- **Inspect poll registry from devtools console**:
  `await import('/static/lib/poll.js').then(m => m.poll._debug_entries())`
- **Inspect bus subscriptions**:
  `await import('/static/lib/bus.js').then(m => m.bus._debug_topics())`

// ===========================================================================
// app.js — bootstrap + switchTab orchestrator.
// ---------------------------------------------------------------------------
// Per-tab logic lives in /static/tabs/{name}.js — each module exports
// {init, activate, deactivate}.
//
//   init()        called once on app boot (below). Registers polls + bus
//                 subs; does NOT fetch data.
//   activate()    called when this tab becomes visible. Lazy-loads data.
//   deactivate()  called when leaving this tab. Most modules no-op
//                 (poll.js auto-gates on activeWhen + document.hidden).
//
// `tab:active` bus topic fires on every successful switchTab, carrying
// {name} so any cross-cutting subscriber (e.g. a layout helper that needs
// to resize a panel) can react.
//
// app.js also owns "global" page chrome that no single tab can claim:
// the top stats bar, the daemon banner, the session-filter dropdown.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import bus from '/static/lib/bus.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

import * as live     from '/static/tabs/live.js';
import * as graph    from '/static/tabs/graph.js';
import * as explorer from '/static/tabs/explorer.js';
import * as logs     from '/static/tabs/logs.js';
import * as health   from '/static/tabs/health.js';
import * as traces   from '/static/tabs/traces.js';

// ── Tab registry ──────────────────────────────────────────────────────
// Order matches the visible tabs in index.html. The active tab on first
// render is 'live' (driven by the .active class in markup).
const TAB_ORDER = ['live', 'graph', 'explorer', 'logs', 'health', 'traces'];
const TABS = { live, graph, explorer, logs, health, traces };
let activeTab = 'live';

function switchTab(name) {
  if (!TABS[name]) return;
  const prev = activeTab;
  try { TABS[prev]?.deactivate?.(); } catch(e) { console.error('[app] deactivate failed:', e); }
  activeTab = name;

  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', TAB_ORDER[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');

  try { TABS[name]?.activate?.(); } catch(e) { console.error('[app] activate failed:', e); }
  bus.publish('tab:active', { name });
}

// ── Page-chrome state (stats bar, session dropdown) ───────────────────
// These don't belong to any one tab — they appear on multiple tabs and are
// owned at the app level. The stats poll feeds Live's stats-bar AND the
// Explorer type-filter dropdown.

let daemonAlive = false;

async function loadStats() {
  try {
    const d = await api.stats();
    daemonAlive = d.daemon === 'alive';
    const statusClass = daemonAlive ? 'alive' : 'unavailable';
    const statusText = daemonAlive ? 'Daemon: alive' : 'Daemon: offline';
    document.getElementById('stats-bar').innerHTML =
      `<div class="stat"><span class="stat-value">${d.nodes}</span><span class="stat-label">Nodes</span></div>
       <div class="stat"><span class="stat-value">${d.edges}</span><span class="stat-label">Edges</span></div>
       <div class="stat"><span class="stat-value">${d.locked}</span><span class="stat-label">Locked</span></div>
       <div class="stat"><span class="stat-value">${d.recent_24h}</span><span class="stat-label">24h</span></div>
       <div class="stat"><span class="stat-value">${d.orphans}</span><span class="stat-label">Orphans</span></div>
       <div class="daemon-status ${statusClass}">${statusText}</div>
       <div class="daemon-status alive" style="font-size:10px;padding:3px 8px">${d.encoding ? 'Encode: ' + d.encoding.position + '/5' + (d.encoding.position === 0 ? ' ⚡' : '') : ''}</div>`;

    const banner = document.getElementById('daemon-banner');
    if (!daemonAlive) {
      banner.innerHTML = '<div class="no-daemon-banner">Daemon is not running — showing read-only data from database. Live events unavailable.</div>';
    } else {
      banner.innerHTML = '';
    }

    const sel = document.getElementById('type-filter');
    const current = sel.value;
    sel.innerHTML = '<option value="">All types</option>';
    Object.entries(d.types).forEach(([t, c]) => {
      sel.innerHTML += `<option value="${t}" ${t===current?'selected':''}>${t} (${c})</option>`;
    });
  } catch(e) { console.error('[dashboard] loadStats failed:', e); }
}

async function loadSessions() {
  try {
    const sessions = await api.sessions();
    const sel = document.getElementById('session-filter');
    const current = sel.value;
    sel.innerHTML = '<option value="">All sessions</option>';
    for (const s of sessions) {
      const label = s.short + ' (' + s.events + ' events)';
      sel.innerHTML += '<option value="' + s.id + '">' + label + '</option>';
    }
    if (current) sel.value = current;
  } catch(e) { console.error('loadSessions error:', e); }
}

// ── Inline-handler exposure ───────────────────────────────────────────
// Inline `onclick="X()"` in index.html (and in dynamically-rendered
// innerHTML strings) look up `X` on the global `window`. ES modules don't
// pollute globals, so each handler the HTML names is mounted explicitly.
//
// As the migration to addEventListener progresses, entries disappear.

window.switchTab             = switchTab;
window.switchFeed            = live.switchFeed;
window.onSessionFilterChange = live.onSessionFilterChange;
window.filterByScale         = live.filterByScale;
window.toggleHookBody        = live.toggleHookBody;
window.toggleSurfacePrompt   = live.toggleSurfacePrompt;
window.toggleEncPrompt       = live.toggleEncPrompt;
window.toggleConsolPrompt    = live.toggleConsolPrompt;
window.switchLogFeed         = logs.switchLogFeed;
window.loadLogs              = logs.loadLogs;
window.loadGraph3D           = graph.loadGraph3D;
window.toggleLegend          = graph.toggleLegend;
window.focusCommunity        = graph.focusCommunity;
window.searchNodes           = explorer.searchNodes;
window.onTraceScaleChange    = traces.onTraceScaleChange;
window.loadTraces            = traces.loadTraces;
window._loadMoreTraces       = traces._loadMoreTraces;
window.loadNodeDetail        = loadNodeDetail;

// ── Boot ──────────────────────────────────────────────────────────────

// Page-chrome polls — 30s stats, 60s sessions. These run regardless of
// active tab (header is always visible).
poll.register({ key: 'stats',    interval: 30000, fetcher: loadStats });
poll.register({ key: 'sessions', interval: 60000, fetcher: loadSessions });

// Wire every tab module's polls + bus subs once.
for (const name of TAB_ORDER) {
  try { TABS[name]?.init?.(); } catch(e) { console.error('[app] init', name, 'failed:', e); }
}

// Activate the initial tab (Live by default — matches the active class
// in index.html). Subsequent switches go through switchTab.
try { TABS[activeTab]?.activate?.(); } catch(e) { console.error('[app] initial activate failed:', e); }
bus.publish('tab:active', { name: activeTab });

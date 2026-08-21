// ===========================================================================
// tabs/live.js — the session, as it happens.
// ---------------------------------------------------------------------------
// One chronological feed of S1 activity (lib/activity.js): surface runs and
// encode runs, interleaved on one time axis but rendered as themselves.
//
// They were briefly merged into a single "moment" card per turn. That was
// wrong: recognition runs every turn, encoding every Nth, so the merged card's
// two-pane body left one pane empty on most turns — a layout claiming an
// alignment the data doesn't have. Two card kinds, one axis.
//
// Previously this tab held two feeds behind a toggle (Decoding / Encoding) plus
// interleaved S2 runs behind a scale filter: three time-scales in one pane,
// needing two dropdowns to be legible. S2 moved to its own tab, and stream
// focus moved to the rail, so both dropdowns are gone.
//
// This module owns:
//   • the split layout (graph ⇄ stream) + graph visibility
//   • the insights panel
//   • the stream rail — who is live, and which stream the feed is focused on
//   • the activity feed: fetch, assemble, incremental render
//   • the graph's live coupling (hover previews, click pins)
//
// It owns no card markup: cards render via lib/activity.js, residue via
// lib/journal.js, session identity via lib/sessions.js.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import bus from '/static/lib/bus.js';
import { el, escapeHtml } from '/static/lib/dom.js';
import * as graph from './graph.js';
import { assembleActivity, renderActivity, fingerprint } from '/static/lib/activity.js';
import * as sessions from '/static/lib/sessions.js';

// Recall cursor is an ISO timestamp, not a rowid: trace_events.id is an 8-char
// hex string under schema v29, so integer ordering no longer applies.
// created_at is monotonic per writer and stable across that change.
let lastRecallTs = '';
const MAX_ITEMS = 110;

// Raw halves, keyed so a re-fetch replaces rather than duplicates. Recall
// events arrive incrementally (2s cursor poll); encode runs arrive as a full
// window each time (they mutate — nodes/edges/journal land after the run's
// first trace, so a run must be replaceable, not append-only).
const _recallsById = new Map();
const _encodesByChain = new Map();

// Rendered cards: key → { el, fingerprint }. The feed patches per card instead
// of rebuilding, so an expanded card stays expanded and the scroll position
// holds under a 2s poll.
const _rendered = new Map();

let _focusSession = '';
// Whether the collapsed resting-stream group is expanded. Page-session state:
// a rail that reopened itself on every poll would fight the operator.
let _restingOpen = false;

// ── Stream rail ────────────────────────────────────────────────────────────
// The session dropdown is gone. Streams are shown, not listed: one pill per
// stream in its own hue. Clicking a pill focuses the feed on it; clicking again
// clears. With several streams interleaving, the hue is what makes a row
// readable at a glance — the pill is where you learn which hue is whom.
//
// Dormant streams collapse behind a count. A machine accumulates them (12 in a
// week here), and at full width they wrapped the rail onto three rows and
// pushed the feed down — so the ones actually thinking get the space, and the
// rest are one click away.

function _renderStreamRail() {
  const rail = document.getElementById('stream-rail');
  if (!rail) return;

  // Streams worth a pill: live ones, plus any that own a card on screen
  // (a stream that just ended is still the thing you're reading).
  const live = sessions.liveSessions();
  const seen = new Set(live.map(s => s.id));
  const inFeed = [];
  for (const key of _rendered.keys()) {
    const sid = _rendered.get(key).el?.dataset.session || '';
    if (sid && !seen.has(sid)) { seen.add(sid); inFeed.push(sessions.sessionInfo(sid)); }
  }
  const rows = [...live, ...inFeed];

  // Awake = actively thinking. Everything else (dormant on the roster, or ended
  // but still owning cards) collapses behind a count.
  const isAwake = (s) => s.live && s.state === 'active';
  const awake = rows.filter(isAwake);
  const resting = rows.filter(s => !isAwake(s));

  const pill = (s) => {
    const color = sessions.sessionColor(s.id);
    const active = _focusSession === s.id;
    const awakeNow = isAwake(s);
    return el('button', {
      class: ['stream-pill', active && 'stream-pill--active', !awakeNow && 'stream-pill--dormant']
        .filter(Boolean),
      style: { '--stream-color': color },
      title: sessions.sessionTooltip(s.id),
      onclick: () => setFocusSession(active ? '' : s.id),
    },
      el('span', {
        class: ['stream-pill-dot', awakeNow && 'stream-pill-dot--live'].filter(Boolean),
        style: { background: color },
      }),
      el('span', { class: 'stream-pill-name' }, sessions.sessionLabel(s.id)),
      s.turns ? el('span', { class: 'stream-pill-count' }, String(s.turns)) : null,
    );
  };

  const pills = [
    el('button', {
      class: ['stream-pill', 'stream-pill--all', !_focusSession && 'stream-pill--active']
        .filter(Boolean),
      title: 'Every stream, newest first',
      onclick: () => setFocusSession(''),
    }, 'all streams', el('span', { class: 'stream-pill-count' }, String(rows.length))),
    ...awake.map(pill),
  ];

  // Keep a focused resting stream visible even while collapsed — hiding the
  // pill you just clicked would leave the filter on with nothing indicating it.
  const focusedResting = resting.find(s => s.id === _focusSession);
  if (focusedResting) pills.push(pill(focusedResting));

  const hidden = resting.filter(s => s.id !== _focusSession);
  if (hidden.length) {
    pills.push(el('button', {
      class: ['stream-pill', 'stream-pill--more', _restingOpen && 'stream-pill--active']
        .filter(Boolean),
      title: hidden.map(s => sessions.sessionLabel(s.id)).join('\n'),
      onclick: () => { _restingOpen = !_restingOpen; _renderStreamRail(); },
    }, (_restingOpen ? '▾ ' : '▸ ') + hidden.length + ' resting'));
    if (_restingOpen) pills.push(...hidden.map(pill));
  }
  rail.replaceChildren(...pills);
}

export function setFocusSession(sessionId) {
  if (_focusSession === sessionId) return;
  _focusSession = sessionId || '';
  // Focus is a filter over what we already hold — no refetch, so switching
  // streams is instant. Re-assert visibility on every rendered card.
  _applyFocus();
  _renderStreamRail();
}

function _applyFocus() {
  for (const { el: node } of _rendered.values()) {
    const sid = node.dataset.session || '';
    node.style.display = (!_focusSession || sid === _focusSession) ? '' : 'none';
  }
  _renderFeedEmptyState();
}

function _renderFeedEmptyState() {
  const feed = document.getElementById('feed-activity');
  if (!feed) return;
  const visible = [..._rendered.values()].filter(r => r.el.style.display !== 'none').length;
  let note = feed.querySelector('.feed-empty');
  if (visible) { if (note) note.remove(); return; }
  if (!note) {
    note = el('div', { class: 'feed-empty' });
    feed.appendChild(note);
  }
  note.textContent = _focusSession
    ? 'Nothing from ' + sessions.sessionLabel(_focusSession) + ' in this window.'
    : 'Waiting for brain activity…';
}

// ── Fetch ──────────────────────────────────────────────────────────────────

// First fetch pulls a backfill, not a tick. The encode window is 48h/200, so
// a 20-event first page would leave the feed almost all encode runs — the two
// halves have to cover comparable ground or one side dominates the feed.
let _recallBackfilled = false;

async function pollRecalls() {
  try {
    const d = await api.recalls(_recallBackfilled
      ? { limit: 20, since_ts: lastRecallTs || undefined }
      : { limit: 300 });
    _recallBackfilled = true;
    for (const evt of (d.events || [])) {
      if (evt.id) _recallsById.set(evt.id, evt);
    }
    if (d.latest_ts) lastRecallTs = d.latest_ts;

    // Judge-output backfill: a hook recall lands before the surfacer has
    // answered, so its selection is empty on arrival. Re-fetch the slice
    // covering still-pending events and let the merge overwrite them.
    //
    // Bounded to the last few minutes: an OLD recall with no judge output
    // never gains one (the surfacer already finished, or that run had no
    // selection), so an unbounded scan would re-fetch a growing slice on
    // every 2s tick forever — the backfill made that a real cost, not a
    // theoretical one.
    const staleBefore = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const pending = [..._recallsById.values()].filter(
      e => e.source === 'hook' && !e.judge_output && (e.timestamp || '') > staleBefore);
    if (pending.length) {
      const minTs = pending.map(e => e.timestamp || '').filter(Boolean).sort()[0];
      if (minTs) {
        const back = new Date(new Date(minTs).getTime() - 1000).toISOString();
        const jd = await api.recalls({ since_ts: back, limit: pending.length + 5 });
        for (const evt of (jd.events || [])) {
          if (evt.id && evt.judge_output) _recallsById.set(evt.id, evt);
        }
      }
    }
    _syncFeed();
  } catch (e) { console.error('[live] recall poll failed:', e); }
}

async function pollEncodes() {
  try {
    // 48h/200 covers normal browsing without pagination. The per-run payload
    // is ~1KB (prompt is lazy, no inline blobs), so the window is cheap.
    const d = await api.encodingRuns({ limit: 200, hours: 48 });
    for (const run of (d.runs || [])) {
      if (run.chain_id) _encodesByChain.set(run.chain_id, run);
    }
    _syncFeed();
  } catch (e) { console.error('[live] encode poll failed:', e); }
}

// ── Render ─────────────────────────────────────────────────────────────────

// Only surface cards wire these (an encode run has no candidate set to light),
// so `item.data` is always the recall event here.
const _graphHooks = {
  onHover: (item) => graph.previewRecallOnGraph(item.data),
  onLeave: () => graph.clearRecallPreview(),
  onPin:   (item) => graph.pinRecallToGraph(item.data),
};

function _syncFeed() {
  const feed = document.getElementById('feed-activity');
  if (!feed) return;
  const items = assembleActivity([..._recallsById.values()], [..._encodesByChain.values()]);

  const live = new Set();
  for (const item of items.slice(0, MAX_ITEMS)) {
    live.add(item.key);
    const fp = fingerprint(item);
    const prior = _rendered.get(item.key);
    if (prior && prior.fingerprint === fp) continue;
    const wasOpen = prior?.el.classList.contains('open');
    const node = renderActivity(item, { showSession: true, hooks: _graphHooks });
    if (wasOpen) {
      // Body is built lazily on expand — a card that was already open has to
      // be populated here, or replacing it would leave the operator staring
      // at an empty body after a data change.
      node._buildActivityBody?.();
      node.classList.add('open');
    }
    if (_focusSession && (item.session_id || '') !== _focusSession) node.style.display = 'none';
    if (prior) {
      prior.el.replaceWith(node);
    } else {
      // Insert in timestamp order — items arrive out of order (an encode run
      // for an older turn lands after newer recalls).
      const at = [...feed.querySelectorAll('.act')]
        .find(n => (n.dataset.ts || '') < (item.ts || ''));
      if (at) feed.insertBefore(node, at); else feed.appendChild(node);
    }
    _rendered.set(item.key, { el: node, fingerprint: fp });
  }

  // Evict cards that fell out of the window — DOM and map together, so the
  // page's memory stays flat over a long session.
  for (const [key, rec] of _rendered) {
    if (!live.has(key)) { rec.el.remove(); _rendered.delete(key); }
  }
  // Bound the raw caches to what the feed can show, or they grow all day.
  _trim(_recallsById, MAX_ITEMS * 3, e => e.timestamp || '');
  _trim(_encodesByChain, MAX_ITEMS, r => r.start_ts || '');

  _renderFeedEmptyState();
  _renderStreamRail();
}

// Drop the OLDEST entries past `max`, by timestamp.
//
// Not by insertion order: both feeds arrive newest-first (ORDER BY created_at
// DESC), and Map.set keeps a re-set key at its ORIGINAL position — so deleting
// from the front of a Map deletes the newest rows, which is the exact opposite
// of eviction. The 300-row recall backfill sits just under the cap, so the
// inverted version started dropping the freshest recalls after ~20 new ones.
// Sorting only runs when actually over the cap.
function _trim(map, max, tsOf) {
  if (map.size <= max) return;
  const byAge = [...map.entries()].sort(
    (a, b) => (tsOf(a[1]) || '').localeCompare(tsOf(b[1]) || ''));
  for (let i = 0; i < byAge.length - max; i++) map.delete(byAge[i][0]);
}

// ── Insights panel ─────────────────────────────────────────────────────────
// Pure subscriber on `insights:tick`. Dismissal is page-session-scoped only —
// no localStorage, so a reload restores the signal and a real problem can't be
// permanently silenced.
const _dismissedInsights = new Set();
let _lastInsights = [];

function _renderInsightsPanel({ insights }) {
  _lastInsights = insights || [];
  _redrawInsightsPanel();
}

function _redrawInsightsPanel() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  const visible = _lastInsights.filter(i => !_dismissedInsights.has(i.title || ''));
  if (!visible.length) { panel.innerHTML = ''; return; }
  panel.innerHTML = visible.map(i => {
    const sev = (i.severity || 'low').toLowerCase();
    return '<div class="insights-card insights-card--' + escapeHtml(sev) + '">' +
      '<div class="insights-icon">' + (i.icon || '') + '</div>' +
      '<div class="insights-body">' +
        '<div class="insights-title">' + escapeHtml(i.title || '') + '</div>' +
        '<div class="insights-detail">' + escapeHtml(i.detail || '') + '</div>' +
      '</div>' +
      '<button class="insights-dismiss" title="Dismiss until reload" ' +
              'data-title="' + escapeHtml(i.title || '') + '">&times;</button>' +
    '</div>';
  }).join('');
}

// Delegated once on the panel — per-render listeners would leak.
function _wireInsightsDismiss() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  panel.addEventListener('click', (e) => {
    const btn = e.target.closest('.insights-dismiss');
    if (!btn) return;
    _dismissedInsights.add(btn.dataset.title || '');
    _redrawInsightsPanel();
  });
}

// ── Live split layout ──────────────────────────────────────────────────────
// Four orientations: graph-left / graph-right / graph-top / graph-bottom.
// _applyLayout is the single state-driven writer: it sets the grid template on
// the active axis, the child order (graph-first / graph-last), and the axis
// class that drives cursor + drag direction. graphPct is always "the graph
// pane's share of the split, 0-100", whatever the orientation — so the drag
// handler inverts when the graph sits on the far side.

const LAYOUT_MODE_KEY = 'dashboard.liveLayoutMode';
const LAYOUT_PCT_KEY  = 'dashboard.liveSplitPct';
const LAYOUT_DEFAULT_MODE = 'graph-left';
const LAYOUT_DEFAULT_PCT  = 60;

const LAYOUTS = {
  'graph-left':   { axis: 'columns', graphFirst: true  },
  'graph-right':  { axis: 'columns', graphFirst: false },
  'graph-top':    { axis: 'rows',    graphFirst: true  },
  'graph-bottom': { axis: 'rows',    graphFirst: false },
};

let _layoutMode = LAYOUT_DEFAULT_MODE;
let _graphPct = LAYOUT_DEFAULT_PCT;

// Graph visibility — orthogonal to the four orientations. Default is
// width-driven: below GRAPH_AUTOLOAD_MIN_WIDTH the dashboard is almost
// certainly in a narrow/embedded pane (e.g. inside Claude), where a continuous
// render loop competes with the host, so we don't mount it at all. An explicit
// operator toggle wins over the width default from then on.
const GRAPH_VISIBLE_KEY = 'dashboard.graphVisible';
const GRAPH_AUTOLOAD_MIN_WIDTH = 1000;
let _graphVisible = true;

function _applyLayout(mode, graphPct) {
  const split = document.getElementById('live-split');
  if (!split) return;
  const cfg = LAYOUTS[mode] || LAYOUTS[LAYOUT_DEFAULT_MODE];
  const pct = Math.max(0, Math.min(100, graphPct));
  _layoutMode = mode;
  _graphPct = pct;

  const sizing = cfg.graphFirst
    ? `${pct}fr 6px ${100 - pct}fr`
    : `${100 - pct}fr 6px ${pct}fr`;
  if (cfg.axis === 'columns') {
    split.style.gridTemplateColumns = sizing;
    split.style.gridTemplateRows = '1fr';
  } else {
    split.style.gridTemplateRows = sizing;
    split.style.gridTemplateColumns = '1fr';
  }

  split.classList.remove('layout--horizontal', 'layout--vertical', 'graph-first', 'graph-last');
  split.classList.add(cfg.axis === 'columns' ? 'layout--horizontal' : 'layout--vertical');
  split.classList.add(cfg.graphFirst ? 'graph-first' : 'graph-last');

  document.querySelectorAll('.live-layout-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.layout === mode);
  });
  // The graph resizes itself via a ResizeObserver on its container, which
  // catches this grid-template change along with window resizes and drags.
}

function _restoreLayout() {
  let mode = LAYOUT_DEFAULT_MODE;
  let pct  = LAYOUT_DEFAULT_PCT;
  try {
    const savedMode = localStorage.getItem(LAYOUT_MODE_KEY);
    if (savedMode && LAYOUTS[savedMode]) mode = savedMode;
    const savedPct = parseFloat(localStorage.getItem(LAYOUT_PCT_KEY));
    if (!Number.isNaN(savedPct) && savedPct >= 0 && savedPct <= 100) pct = savedPct;
  } catch (e) { /* localStorage may be blocked; defaults are fine */ }
  _applyLayout(mode, pct);
}

function _persistLayout() {
  try {
    localStorage.setItem(LAYOUT_MODE_KEY, _layoutMode);
    localStorage.setItem(LAYOUT_PCT_KEY, String(_graphPct));
  } catch (e) { /* blocked storage → silently skip */ }
}

function _restoreGraphVisibility() {
  let visible;
  try {
    const saved = localStorage.getItem(GRAPH_VISIBLE_KEY);
    if (saved === '1') visible = true;
    else if (saved === '0') visible = false;
  } catch (e) { /* blocked → fall through to width default */ }
  if (visible === undefined) visible = window.innerWidth >= GRAPH_AUTOLOAD_MIN_WIDTH;
  _graphVisible = visible;
  _applyGraphVisibility();
}

// Reflect _graphVisible in the DOM. Does NOT mount/destroy — callers do that,
// so this stays a pure view-sync safe to call before the graph module is ready.
function _applyGraphVisibility() {
  const split = document.getElementById('live-split');
  if (split) split.classList.toggle('graph-hidden', !_graphVisible);
  const btn = document.getElementById('graph-toggle-btn');
  if (btn) {
    btn.classList.toggle('active', _graphVisible);
    btn.textContent = _graphVisible ? 'Hide graph' : 'Show graph';
    btn.title = _graphVisible
      ? 'Hide the graph — frees its render loop'
      : 'Show the graph';
  }
}

/** Toolbar toggle. Full teardown, not pause: a hidden graph costs zero. */
export function toggleGraph() {
  _graphVisible = !_graphVisible;
  try { localStorage.setItem(GRAPH_VISIBLE_KEY, _graphVisible ? '1' : '0'); } catch (e) { /* blocked */ }
  _applyGraphVisibility();
  if (_graphVisible) graph.activate();
  else graph.destroy();
}

export function setLiveLayout(mode) {
  if (!LAYOUTS[mode]) return;
  _applyLayout(mode, _graphPct);
  _persistLayout();
}

// Idempotency guard — _setupDivider attaches document-level listeners. If
// init() ever fires twice (hot reload, future double-wire bug) we'd get N
// parallel handlers on every mouse move.
let _dividerWired = false;
function _setupDivider() {
  if (_dividerWired) return;
  const divider = document.getElementById('live-divider');
  const split = document.getElementById('live-split');
  if (!divider || !split) return;
  _dividerWired = true;
  let dragging = false;

  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    divider.classList.add('dragging');
    split.classList.add('dragging');
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const cfg = LAYOUTS[_layoutMode];
    const rect = split.getBoundingClientRect();
    const raw = cfg.axis === 'columns'
      ? ((e.clientX - rect.left) / rect.width) * 100
      : ((e.clientY - rect.top)  / rect.height) * 100;
    // Graph on the FAR side → dragging toward it shrinks the first pane.
    _applyLayout(_layoutMode, cfg.graphFirst ? raw : 100 - raw);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    split.classList.remove('dragging');
    _persistLayout();
  });
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

export function init() {
  _restoreLayout();
  _restoreGraphVisibility();
  _setupDivider();
  _renderFeedEmptyState();

  bus.subscribe('insights:tick', _renderInsightsPanel);
  bus.subscribe('sessions:tick', () => { _renderStreamRail(); });
  _wireInsightsDismiss();

  // Pinned-card highlight. graph.js publishes `graph:pinned` when a pin is
  // set or cleared; mirroring it here means graph.js never reaches across the
  // module boundary to mutate this feed's DOM.
  bus.subscribe('graph:pinned', ({ eventId }) => {
    document.querySelectorAll('.act--pinned')
      .forEach(n => n.classList.remove('act--pinned'));
    if (!eventId) return;
    const target = document.querySelector('.act[data-recall-id="' + eventId + '"]');
    if (target) target.classList.add('act--pinned');
  });

  const liveVisible = () => document.getElementById('tab-live').classList.contains('active');

  poll.register({ key: 'recalls',  interval: 2000,  activeWhen: liveVisible, fetcher: pollRecalls });
  poll.register({ key: 'encodes',  interval: 6000,  activeWhen: liveVisible, fetcher: pollEncodes });
  poll.register({
    key: 'insights-live',
    interval: 60000,
    activeWhen: liveVisible,
    fetcher: async () => {
      try {
        const env = await api.insightsLive();
        if (env && env.status === 'success') {
          bus.publish('insights:tick', { insights: env.data || [] });
        } else if (env && env.status === 'error') {
          console.error('[live] insights endpoint error:', env.error);
        }
      } catch (e) { console.error('[live] insights fetch failed:', e); }
    },
  });
}

export function activate() {
  // Live owns the graph — drive its activate() so the scene mounts (first
  // time) or resizes (after). Skipped entirely when hidden so no render loop
  // spins; toggleGraph() mounts on demand.
  if (_graphVisible) {
    try { graph.activate(); } catch (e) { console.error('[live] graph activate failed:', e); }
  }
  pollEncodes();
}

export function deactivate() {
  // Graph stays mounted in the hidden tab; its canvas keeps state warm so
  // re-activating is instant.
}

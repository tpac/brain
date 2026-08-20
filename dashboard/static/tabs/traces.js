// ===========================================================================
// tabs/traces.js — trace chain rendering for arbitrary time-window + scale
// + session filter combos.
// ---------------------------------------------------------------------------
// Renders chains incrementally in batches of _TRACE_BATCH; a "Load more"
// button fetches the next page client-side (the server already returned
// the full result).
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime, identityChipHTML } from '/static/lib/dom.js';
import { SCALE_COLORS } from '/static/lib/scales.js';
import { reapplyTraceFlashIfPending, loadNodeDetail } from '/static/lib/node_detail.js';
import { renderTraceDetail, collapsedBadges } from '/static/lib/trace_detail.js';
import { renderFriendlyChain } from '/static/lib/trace_friendly.js';

let _traceChainEntries = [];
let _traceRendered = 0;
const _TRACE_BATCH = 30;

// View mode: 'friendly' (jargon-free activity digest, one card per chain) or
// 'technical' (the full O/K/Δ event drill-down). Persisted so each operator
// keeps their choice; defaults to technical (hardening is the current use).
let _traceMode = (() => {
  try { return localStorage.getItem('traceMode') || 'technical'; } catch (_) { return 'technical'; }
})();

// Friendly relabelling of the jargon controls. Same filter values (s0/s1/s2),
// human words on top — a newcomer picks "Remembering", not "S1".
const _FRIENDLY_SCALE_LABELS = {
  '': 'All activity', s0: 'Conversations', s1: 'Remembering & learning',
  s2: 'Organizing', s3: 'Reflecting', s4: 'Growing',
};
const _TECH_SCALE_LABELS = {
  '': 'All scales', s0: 'S0 (Exchange)', s1: 'S1 (Turn)',
  s2: 'S2 (Graph)', s3: 'S3 (Sleep)', s4: 'S4 (Growth)',
};

// Expand-on-click state. Tracked by trace id (not DOM) so expansion survives
// the 5s poll that rewrites #traces-content from scratch — the same reason
// the source-ref flash is re-applied after each render. _traceEventById maps
// id → row so the delegated click handler can build detail without re-fetching.
const _traceExpanded = new Set();
let _traceEventById = {};
// Signature of the last rendered result. The 5s poll calls loadTraces
// unconditionally; without this guard it rebuilt #traces-content every tick —
// blanking innerHTML resets the page scroll (jumps to top) and wipes the scroll
// position inside any expanded <pre> the operator is reading. We skip the
// rebuild when nothing relevant changed. Expansions go through the click
// handler (direct DOM), not loadTraces, so skipping never strands them.
let _lastTraceFp = null;

function _detailHTML(ev) {
  return '<div class="trace-detail" data-detail-for="' + escapeHtml(String(ev.id)) + '" '
    + 'style="background:#070710;border-top:1px solid #15151f;padding:6px 14px 10px 22px">'
    + renderTraceDetail(ev) + '</div>';
}

export function onTraceScaleChange() {
  const scale = document.getElementById('trace-scale-filter').value;
  const hoursEl = document.getElementById('trace-hours-filter');
  if (scale && scale >= 's2' && parseInt(hoursEl.value) < 168) {
    hoursEl.value = '168';
  }
  loadTraces();
}

export async function loadTraces(opts = {}) {
  try {
    const scaleFilter = document.getElementById('trace-scale-filter').value;
    const hours = document.getElementById('trace-hours-filter').value;
    // Callers (e.g. node_detail's "Open in Traces tab") can force a
    // session that isn't in the dropdown yet — setting the dropdown's
    // .value to a missing option silently resets to '', so reading from
    // it would lose the override. Explicit opts.session wins.
    const sessionFilter = (opts.session !== undefined)
      ? opts.session
      : document.getElementById('trace-session-filter').value;
    const traces = await api.traces({
      hours,
      scale: scaleFilter || undefined,
      session: sessionFilter || undefined,
    });
    // Skip the rebuild when nothing changed (traces use absolute timestamps, so
    // identical data → identical render). Newest id+count+filters+mode capture
    // every structural change; a new trace flips the newest id. Bypass via
    // opts.force for explicit refreshes that must re-render regardless.
    const fp = [_traceMode, scaleFilter, hours, sessionFilter,
      traces.length, traces[0] && traces[0].id, traces[traces.length - 1] && traces[traces.length - 1].id].join('|');
    if (!opts.force && fp === _lastTraceFp) return;
    _lastTraceFp = fp;
    const el = document.getElementById('traces-content');
    const techLabel = hours <= 1 ? '1h' : hours <= 6 ? '6h' : hours <= 24 ? '24h' : '7d';
    const friendlyLabel = hours <= 1 ? 'the last hour' : hours <= 6 ? 'the last 6 hours'
      : hours <= 24 ? 'the last day' : 'the last week';
    const chainCount = new Set(traces.map(t => t.chain_id)).size;
    document.getElementById('trace-count').textContent = _traceMode === 'friendly'
      ? chainCount + ' things the brain did in ' + friendlyLabel
      : traces.length + ' events (' + techLabel + ')';

    const sessSelect = document.getElementById('trace-session-filter');
    // Sync the dropdown to the resolved sessionFilter — without this,
    // an override passed via opts.session wouldn't visually appear in
    // the dropdown, so the user couldn't tell which session they're
    // looking at. `prevVal` used to track the dropdown's prior value
    // for preservation; sessionFilter supersedes it once a caller has
    // explicitly chosen.
    try {
      const sessions = await api.sessions();
      // Note: not `opts` — that's the function parameter (opts.force/.session).
      // Sessions read by their handle (worktree / branch tail), never raw
      // hex — resolved once server-side in queries/sessions.py so the Traces
      // dropdown, the stream rail, and activity cards all say the same name.
      const optsHtml = '<option value="">All sessions</option>' + sessions.map(s =>
        '<option value="' + s.id + '"' + (s.id === sessionFilter ? ' selected' : '') + '>'
        + escapeHtml(s.handle || s.short) + ' · ' + s.events + ' events</option>'
      ).join('');
      sessSelect.innerHTML = optsHtml;
    } catch(e) { /* keep existing options */ }

    if (!traces.length) {
      el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">No trace events yet. Traces will appear after your next prompt.</div>';
      _traceChainEntries = [];
      return;
    }

    const chains = {};
    traces.forEach(t => {
      if (!chains[t.chain_id]) chains[t.chain_id] = [];
      chains[t.chain_id].push(t);
    });
    const chainEntries = Object.entries(chains);
    chainEntries.forEach(([_, events]) => events.sort(
      (a, b) => (a.created_at || '').localeCompare(b.created_at || '')));
    chainEntries.sort((a, b) => {
      const aMax = a[1][a[1].length - 1]?.created_at || '';
      const bMax = b[1][b[1].length - 1]?.created_at || '';
      return bMax.localeCompare(aMax);
    });
    _traceChainEntries = chainEntries;
    _traceRendered = 0;
    _traceEventById = {};
    el.innerHTML = '';
    _renderTracesBatch(el);
    // Re-apply the source-ref flash if one is still pending. The 5s
    // poll rewrites #traces-content from scratch, which would otherwise
    // destroy the .trace-event--target class added by node_detail.js.
    // reapplyTraceFlashIfPending is a no-op when nothing's pending.
    reapplyTraceFlashIfPending();
  } catch(e) { console.error('loadTraces', e); }
}

function _traceChainLabel(chainId) {
  // Map chain IDs to readable labels. Chain IDs remain string-prefixed
  // (s0-/s1r-/s1e-/s2-) even after trace_events.id became hex — those are
  // different identifiers.
  if (chainId.startsWith('s0-')) { const p = chainId.split('-'); return 'S0 Exchange #' + (p[2] || '?'); }
  if (chainId.startsWith('s1r-')) { const p = chainId.split('-'); return 'S1 Recall (Surface) #' + (p[2] || '?'); }
  if (chainId.startsWith('s1e-')) { const p = chainId.split('-'); return 'S1 Encode #' + (p[2] || '?'); }
  if (chainId.startsWith('s2-')) {
    const op = chainId.split('-').slice(2).join('-');
    const labels = {community_detection:'S2 Community Detection', consolidation:'S2 Consolidation', edge_family_integration:'S2 Edge Families', healer:'S2 Healer', relation_reclassify:'S2 Edge Reclassify'};
    return labels[op] || 'S2 ' + op.replace(/_/g, ' ');
  }
  if (chainId.startsWith('s3-')) return 'S3 ' + chainId.split('-').slice(2).join(' ');
  return chainId;
}

function _renderTracesBatch(el) {
  const typeLabels = {O:'Observed', K:'Selected', delta:'Changed', outcome:'Outcome'};
  const typeColors = {O:'#45B7D1', K:'#ffaa33', delta:'#33ff88', outcome:'#aa66ff'};
  const end = Math.min(_traceRendered + _TRACE_BATCH, _traceChainEntries.length);

  let html = '';
  for (let i = _traceRendered; i < end; i++) {
    const [chainId, events] = _traceChainEntries[i];

    // Friendly mode: one plain-language card per chain, no event drill-down.
    if (_traceMode === 'friendly') {
      html += renderFriendlyChain(chainId, events);
      continue;
    }

    const firstTime = events[0].created_at;
    const chainScale = events[0].scale;
    const color = SCALE_COLORS[chainScale] || '#666';
    const label = _traceChainLabel(chainId);
    const sessionId = events[0].session_id || '';
    const sessionTag = sessionId ? '<span style="color:#444;font-size:9px;margin-left:6px">' + sessionId.substring(0,8) + '</span>' : '';
    // Identity per chain: first event with non-empty identity wins. Chains
    // often span multiple writes from the same speaker pair so one rendering
    // per chain is enough; a per-event chip would just repeat ad nauseam.
    let chainHi = '', chainAi = '';
    for (const ev of events) {
      if (ev.human_identity || ev.agent_identity) {
        chainHi = ev.human_identity; chainAi = ev.agent_identity; break;
      }
    }
    const identityTag = (chainHi || chainAi)
      ? '<span style="margin-left:6px">' + identityChipHTML(chainHi, chainAi) + '</span>'
      : '';

    // data-chain-id on the wrapper + data-trace-id on each event row let
    // node_detail.js scroll a specific trace into view after navigating
    // from a source-ref card. Without these attributes there'd be no
    // selector to target the right event.
    html += '<div class="trace-chain" data-chain-id="' + escapeHtml(chainId) + '" style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid ' + color + '">';
    html += '<div style="padding:8px 12px;display:flex;justify-content:space-between;align-items:center">';
    html += '<div><span style="color:' + color + ';font-size:12px;font-weight:bold">' + label + '</span>' + sessionTag + identityTag + '</div>';
    html += '<span style="color:#555;font-size:10px">' + localTime(firstTime) + '</span>';
    html += '</div>';

    events.forEach(ev => {
      if (ev.id) _traceEventById[ev.id] = ev;
      const tColor = typeColors[ev.event_type] || '#666';
      const tLabel = typeLabels[ev.event_type] || ev.event_type;
      const traceIdAttr = ev.id ? ' data-trace-id="' + escapeHtml(String(ev.id)) + '"' : '';
      const expanded = ev.id && _traceExpanded.has(ev.id);
      const caret = '<span class="trace-caret" style="flex-shrink:0;width:10px;color:#556;font-size:9px;margin-top:3px">' + (expanded ? '▾' : '▸') + '</span>';
      // Whole row is the expand affordance — cursor:pointer signals it; the
      // delegated #traces-content handler toggles on click.
      html += '<div class="trace-event"' + traceIdAttr + ' style="padding:4px 12px 4px 16px;border-top:1px solid #111;display:flex;gap:6px;align-items:flex-start;cursor:pointer">';
      html += caret;
      html += '<span style="flex-shrink:0;font-size:10px;font-weight:bold;color:' + tColor + ';min-width:55px">' + tLabel + '</span>';
      html += '<div style="flex:1;min-width:0">';
      if (ev.ref_type) html += '<span style="color:#666;font-size:10px;background:#1a1a2a;padding:1px 4px;border-radius:2px;margin-right:4px">' + ev.ref_type + '</span>';
      html += collapsedBadges(ev);
      html += '<div style="color:#ccc;font-size:12px;margin-top:2px;white-space:pre-wrap;word-break:break-word">' + escapeHtml((ev.summary || '').substring(0, 300)) + '</div>';
      html += '</div>';
      html += '<span style="color:#444;font-size:9px;flex-shrink:0;white-space:nowrap">' + localTime(ev.created_at, 'time') + '</span>';
      html += '</div>';
      if (expanded) html += _detailHTML(ev);
    });

    html += '</div>';
  }
  el.insertAdjacentHTML('beforeend', html);
  _traceRendered = end;

  if (_traceRendered < _traceChainEntries.length) {
    el.insertAdjacentHTML('beforeend', '<div id="trace-load-more" style="text-align:center;padding:12px"><button onclick="_loadMoreTraces()" style="background:#1a1a2a;color:#7eb8ff;border:1px solid #3a3a5a;padding:4px 16px;border-radius:4px;cursor:pointer">Load more (' + (_traceChainEntries.length - _traceRendered) + ' remaining)</button></div>');
  }
}

export function _loadMoreTraces() {
  const btn = document.getElementById('trace-load-more');
  if (btn) btn.remove();
  _renderTracesBatch(document.getElementById('traces-content'));
}

// Delegated click handler on the stable #traces-content container (its
// innerHTML is rewritten every poll, but the element itself persists, so one
// listener outlives every re-render). Two targets:
//   .trace-nodeid → open that node in the detail panel (no inline onclick, so
//                   arbitrary node-id chars never get interpolated into JS).
//   .trace-event  → toggle the expanded technical detail for that event.
// The detail panel is a SIBLING of .trace-event (inserted afterend), not a
// child, so clicks inside it (e.g. selecting text in a code block) don't
// match `.trace-event` and won't collapse the row.
function _onTracesClick(e) {
  const nodeEl = e.target.closest('.trace-nodeid');
  if (nodeEl) {
    const id = nodeEl.dataset.nodeid;
    if (id) loadNodeDetail(id);
    return;
  }
  const row = e.target.closest('.trace-event');
  if (!row) return;
  const id = row.getAttribute('data-trace-id');
  if (!id) return;
  const caret = row.querySelector('.trace-caret');
  const existing = document.querySelector('.trace-detail[data-detail-for="' + id + '"]');
  if (_traceExpanded.has(id)) {
    _traceExpanded.delete(id);
    if (existing) existing.remove();
    if (caret) caret.textContent = '▸';
  } else {
    const ev = _traceEventById[id];
    if (!ev) return;   // not in the rendered batch's map — don't flip the caret with no detail to show
    _traceExpanded.add(id);
    if (!existing) row.insertAdjacentHTML('afterend', _detailHTML(ev));
    if (caret) caret.textContent = '▾';
  }
}

// ── View mode (Friendly ⇄ Technical) ──────────────────────────────────

// Relabel the jargon controls + sync the toggle buttons to the active mode.
function _applyModeLabels() {
  const labels = _traceMode === 'friendly' ? _FRIENDLY_SCALE_LABELS : _TECH_SCALE_LABELS;
  const sel = document.getElementById('trace-scale-filter');
  if (sel) [...sel.options].forEach(o => { if (o.value in labels) o.textContent = labels[o.value]; });
  ['friendly', 'technical'].forEach(m => {
    const b = document.getElementById('trace-mode-' + m);
    if (b) b.classList.toggle('active', _traceMode === m);
  });
}

export function setTraceMode(mode) {
  if (mode !== 'friendly' && mode !== 'technical' || mode === _traceMode) return;
  _traceMode = mode;
  try { localStorage.setItem('traceMode', mode); } catch (_) {}
  // Expanded-detail state is technical-only; clear on switch so it's not
  // stranded behind the friendly cards.
  _traceExpanded.clear();
  _applyModeLabels();
  loadTraces();
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // One delegated listener on the persistent container — survives every
  // poll-driven innerHTML rewrite (see _onTracesClick).
  const container = document.getElementById('traces-content');
  if (container && !container._traceClickBound) {
    container.addEventListener('click', _onTracesClick);
    container._traceClickBound = true;
  }
  _applyModeLabels();
  // 5s refresh while the Traces tab is active. Matches the pattern other
  // tabs use; the older _startTraceAutoRefresh/_stopTraceAutoRefresh helpers
  // were converted to a poll.register call during P1.
  poll.register({
    key: 'traces',
    interval: 5000,
    activeWhen: () => {
      const tab = document.getElementById('tab-traces');
      return tab && tab.classList.contains('active');
    },
    fetcher: loadTraces,
  });
}

export function activate() {
  _applyModeLabels();
  loadTraces();
}

export function deactivate() {}

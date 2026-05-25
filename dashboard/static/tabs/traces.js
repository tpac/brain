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
import { escapeHtml, localTime } from '/static/lib/dom.js';

let _traceChainEntries = [];
let _traceRendered = 0;
const _TRACE_BATCH = 30;

function renderIdentityChip(human, agent) {
  if (!human && !agent) return '';
  const h = human ? escapeHtml(human) : '?';
  const a = agent ? escapeHtml(agent) : '?';
  return '<span class="identity-chip" title="speaker → responder">' +
    h + '<span style="color:#555;margin:0 3px">→</span>' + a + '</span>';
}

export function onTraceScaleChange() {
  const scale = document.getElementById('trace-scale-filter').value;
  const hoursEl = document.getElementById('trace-hours-filter');
  if (scale && scale >= 's2' && parseInt(hoursEl.value) < 168) {
    hoursEl.value = '168';
  }
  loadTraces();
}

export async function loadTraces() {
  try {
    const scaleFilter = document.getElementById('trace-scale-filter').value;
    const hours = document.getElementById('trace-hours-filter').value;
    const sessionFilter = document.getElementById('trace-session-filter').value;
    const traces = await api.traces({
      hours,
      scale: scaleFilter || undefined,
      session: sessionFilter || undefined,
    });
    const el = document.getElementById('traces-content');
    const label = hours <= 1 ? '1h' : hours <= 6 ? '6h' : hours <= 24 ? '24h' : '7d';
    document.getElementById('trace-count').textContent = traces.length + ' events (' + label + ')';

    const sessSelect = document.getElementById('trace-session-filter');
    const prevVal = sessSelect.value;
    try {
      const sessions = await api.sessions();
      const opts = '<option value="">All sessions</option>' + sessions.map(s =>
        '<option value="' + s.id + '"' + (s.id === prevVal ? ' selected' : '') + '>' + s.short + ' (' + s.events + ' events)</option>'
      ).join('');
      sessSelect.innerHTML = opts;
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
    el.innerHTML = '';
    _renderTracesBatch(el);
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
  const scaleColors = {s0:'#888', s1:'#7eb8ff', s2:'#ffaa33', s3:'#33ff88', s4:'#ff66aa'};
  const typeLabels = {O:'Observed', K:'Selected', delta:'Changed', outcome:'Outcome'};
  const typeColors = {O:'#45B7D1', K:'#ffaa33', delta:'#33ff88', outcome:'#aa66ff'};
  const end = Math.min(_traceRendered + _TRACE_BATCH, _traceChainEntries.length);

  let html = '';
  for (let i = _traceRendered; i < end; i++) {
    const [chainId, events] = _traceChainEntries[i];
    const firstTime = events[0].created_at;
    const chainScale = events[0].scale;
    const color = scaleColors[chainScale] || '#666';
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
      ? '<span style="margin-left:6px">' + renderIdentityChip(chainHi, chainAi) + '</span>'
      : '';

    html += '<div style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid ' + color + '">';
    html += '<div style="padding:8px 12px;display:flex;justify-content:space-between;align-items:center">';
    html += '<div><span style="color:' + color + ';font-size:12px;font-weight:bold">' + label + '</span>' + sessionTag + identityTag + '</div>';
    html += '<span style="color:#555;font-size:10px">' + localTime(firstTime) + '</span>';
    html += '</div>';

    events.forEach(ev => {
      const tColor = typeColors[ev.event_type] || '#666';
      const tLabel = typeLabels[ev.event_type] || ev.event_type;
      html += '<div style="padding:4px 12px 4px 20px;border-top:1px solid #111;display:flex;gap:8px;align-items:flex-start">';
      html += '<span style="flex-shrink:0;font-size:10px;font-weight:bold;color:' + tColor + ';min-width:55px">' + tLabel + '</span>';
      html += '<div style="flex:1;min-width:0">';
      if (ev.ref_type) html += '<span style="color:#666;font-size:10px;background:#1a1a2a;padding:1px 4px;border-radius:2px;margin-right:4px">' + ev.ref_type + '</span>';
      html += '<div style="color:#ccc;font-size:12px;margin-top:2px;white-space:pre-wrap;word-break:break-word">' + escapeHtml((ev.summary || '').substring(0, 300)) + '</div>';
      html += '</div>';
      html += '<span style="color:#444;font-size:9px;flex-shrink:0;white-space:nowrap">' + localTime(ev.created_at, 'time') + '</span>';
      html += '</div>';
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

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
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
  loadTraces();
}

export function deactivate() {}

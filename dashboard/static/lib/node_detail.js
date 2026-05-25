// ===========================================================================
// lib/node_detail.js — the right-side node detail panel.
// ---------------------------------------------------------------------------
// Called from many tabs (Graph onNodeClick, Explorer card click, Live entries
// clicking a candidate, recursive open-correction clicks). Lives in lib/ —
// not a tab module — because it's a shared overlay, not tab-scoped.
//
// Sections (top → bottom):
//   1. Close button
//   2. Title (type badge + lock/critical glyphs + node title)
//   3. Meta strip (id, conf, access, source)
//   4. Timestamps (created / last accessed / revised, relative + absolute)
//   5. Content (n.content)
//   6. Fields (every non-empty top-level + metadata KV pair — deny-list,
//      not allow-list, so new keys surface automatically)
//   7. Encoded-from traces (source_refs, clickable → Traces tab)
//   8. Corrections (aspect-edge walk)
//   9. Connections (top 20 by edge weight)
//
// Renders via el() — auto-escaping, addEventListener wiring, no inline
// `onclick="..."` strings in rendered HTML.
// ===========================================================================

import { api } from './api.js';
import { el, html, mount, localTime, relativeTime } from './dom.js';
import { SCALE_COLORS } from './scales.js';

// ── Field rendering policy ────────────────────────────────────────────
// Fields rendered elsewhere (title, content, timestamps, the meta strip)
// don't appear in the Fields section. Everything else does — including
// metadata KV keys the encoder may add over time. New keys surface
// automatically, no curation needed.

const NODE_FIELDS_HIDDEN_IN_LIST = new Set([
  'id', 'type', 'title', 'content', 'locked', 'critical', 'metadata',
  'created_at', 'last_accessed', 'revised_at',
  'access_count', 'confidence', 'encoding_source',
]);

// Pretty labels for known keys. Anything not listed renders with the raw
// snake_case key — never silently drops, because Tom asked for ALL fields.
const FIELD_LABELS = {
  emotion:            'emotion',
  personal:           'personal',
  personal_context:   'personal context',
  evolution_status:   'evolution status',
  situation:          'situation',
  reasoning:          'reasoning',
  question:           'question',
  user_raw_quote:     'user raw quote',
  anchor_raw_quote:   'anchor raw quote',
  correction_of:      'correction_of',
  correction_pattern: 'correction pattern',
  source_context:     'source context',
};

function _labelFor(key) { return FIELD_LABELS[key] || key.replace(/_/g, ' '); }

// ── Section helpers (each returns an element or array of elements) ────

function _closeButton() {
  return el('div', {
    class: 'nd-close',
    onclick: () => { document.getElementById('node-detail').style.display = 'none'; },
  }, html('&times;'));
}

function _title(n) {
  return el('div', { class: 'nd-title' },
    el('span', { class: 'type-badge type-' + (n.type || '') }, n.type || ''),
    ' ',
    n.locked   ? html('&#x1f512; ') : null,
    n.critical ? '⚠️ '              : null,
    n.title || '',
  );
}

function _metaStrip(n) {
  return el('div', { class: 'nd-meta' },
    el('span', null, 'id: ' + (n.id || '').substring(0, 8)),
    el('span', null, 'accessed: ' + n.access_count + 'x'),
    el('span', null, 'conf: ' + (n.confidence || 0).toFixed(2)),
    el('span', null, 'source: ' + (n.encoding_source || '?')),
  );
}

// Three timestamp pills — created / last_accessed / revised — each with a
// big relative time and a small absolute below it. `title=` on the pill
// shows the full ISO so power users can copy-paste.
function _timestampsSection(n) {
  const pills = [];
  const addPill = (label, iso) => {
    if (!iso) return;
    pills.push(el('div', { class: 'nd-time-pill', title: iso },
      el('span', { class: 'nd-time-label' }, label),
      el('span', { class: 'nd-time-rel' }, relativeTime(iso) || '—'),
      el('span', { class: 'nd-time-abs' }, localTime(iso)),
    ));
  };
  addPill('CREATED',       n.created_at);
  addPill('LAST ACCESSED', n.last_accessed);
  addPill('REVISED',       n.revised_at);
  if (!pills.length) return null;
  return el('div', { class: 'nd-times' }, ...pills);
}

// Render a single field value — string, number, bool, null. Special case
// for correction_of: render as a recursive-open link.
function _fieldValue(key, value) {
  if (key === 'correction_of' && value) {
    return el('a', {
      class: 'nd-correction-of',
      onclick: () => loadNodeDetail(String(value)),
    }, String(value));
  }
  if (typeof value === 'string') {
    if (key === 'user_raw_quote' || key === 'anchor_raw_quote') {
      return el('em', null, '"' + value + '"');
    }
    return value;
  }
  // numbers, booleans → monospaced rendering so a sea of fields scans cleanly.
  return el('span', { class: 'nd-field-mono' }, String(value));
}

function _fieldsSection(n, meta) {
  const fields = [];
  const pushField = (key, value) => {
    if (value === null || value === undefined || value === '' || value === false) return;
    fields.push(el('div', { class: 'nd-field' },
      el('span', { class: 'nd-fk' }, _labelFor(key) + ':'),
      ' ',
      _fieldValue(key, value),
    ));
  };

  // 1. Top-level node fields (deny-list).
  for (const [k, v] of Object.entries(n)) {
    if (NODE_FIELDS_HIDDEN_IN_LIST.has(k)) continue;
    pushField(k, v);
  }
  // 2. Metadata KV pairs — same policy, never silently drop a key.
  for (const [k, v] of Object.entries(meta || {})) {
    // `situation` is sometimes promoted to top-level by the query — skip
    // the duplicate if we already rendered it from `n.situation`.
    if (k === 'situation' && n.situation) continue;
    pushField(k, v);
  }

  if (!fields.length) return [];
  return [el('div', { class: 'nd-section' }, 'Fields'), ...fields];
}

// Navigate to Traces tab + filter to the source session of this ref. The
// user lands on a list of chains from that session — finding THE specific
// chain among them is still scroll-work, but at least we narrow the haul.
function _openTraceInTracesTab(ref) {
  const sessionId = ref && ref.session_id;
  if (!sessionId || typeof window.switchTab !== 'function') return;
  window.switchTab('traces');
  // The Traces tab's session-filter dropdown options are populated by its
  // own loadTraces() call. We may need to wait for that to populate before
  // setting the value. Try immediate; if the option isn't there yet, the
  // assignment is silently dropped — set a short retry.
  const setFilter = () => {
    const sel = document.getElementById('trace-session-filter');
    if (!sel) return false;
    // Set even if the option isn't present yet; loadTraces preserves
    // the prior value when re-populating.
    sel.value = sessionId;
    if (typeof window.loadTraces === 'function') window.loadTraces();
    return true;
  };
  if (!setFilter()) setTimeout(setFilter, 400);
}

// Episodic refs — which traces this node was encoded from. v27 substrate
// (node_source_refs). Each card is clickable → Traces tab + session filter.
function _sourceRefsSection(refs) {
  if (!refs.length) return [];
  const out = [el('div', { class: 'nd-section' },
    'Encoded from ' + refs.length + ' trace(s)')];
  for (const ref of refs) {
    if (ref.missing) {
      out.push(el('div', { class: 'nd-field nd-conn-missing' },
        'trace ' + (ref.trace_id || '') + ' (not found — log-rotated or archived)'));
      continue;
    }
    const sc = SCALE_COLORS[ref.scale] || '#666';
    const sess = ref.session_id ? ref.session_id.substring(0, 8) : '';
    const tagText = ref.scale + ' · ' + (ref.event_type || '')
                  + (ref.ref_type ? ' · ' + ref.ref_type : '')
                  + (sess ? ' · ' + sess : '');
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable',
      style: { borderLeftColor: sc },
      onclick: () => _openTraceInTracesTab(ref),
    },
      el('div', { class: 'nd-conn-tag', style: { color: sc } }, tagText),
      el('div', { class: 'nd-conn-summary' }, (ref.summary || '').substring(0, 180)),
      el('div', { class: 'nd-conn-trace', title: ref.trace_id || '' },
        'trace ' + (ref.trace_id || '').substring(0, 8)
        + ' · pos ' + (ref.position || 1)
        + ' · ' + (ref.trace_created_at ? localTime(ref.trace_created_at) : '?')
        + (ref.trace_created_at ? ' (' + relativeTime(ref.trace_created_at) + ')' : '')),
      ref.chain_id
        ? el('div', { class: 'nd-conn-trace', title: ref.chain_id }, 'chain: ' + ref.chain_id)
        : null,
      el('div', { class: 'nd-conn-action' }, '→ Open in Traces tab'),
    ));
  }
  return out;
}

// Corrections — aspect-edge walk via daemon. Direction matters: 'corrects'
// means the neighbor IS this node's corrector; 'corrected_by' is the inverse.
function _correctionsSection(corrections) {
  if (!corrections.length) return [];
  const out = [el('div', { class: 'nd-section' },
    'Corrections (' + corrections.length + ')')];
  for (const c of corrections) {
    const dirColor = c.direction === 'corrects' ? '#ff8866' : '#88ccff';
    const dirLabel = c.direction === 'corrects' ? '⤴ corrects this' : '↪ corrected by this';
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable',
      style: { borderLeftColor: dirColor },
      onclick: () => loadNodeDetail(c.id || ''),
    },
      el('div', { class: 'nd-conn-tag', style: { color: dirColor } },
        dirLabel + ' · ' + (c.relation || '')),
      el('div', { class: 'nd-conn-title' },
        el('span', { class: 'type-badge type-' + (c.type || '') }, c.type || ''),
        ' ',
        (c.title || '').substring(0, 80)),
      c.edge_description
        ? el('div', { class: 'nd-conn-edge-why' },
            'why: ' + c.edge_description.substring(0, 180))
        : null,
      c.content
        ? el('div', { class: 'nd-conn-content' }, c.content.substring(0, 200))
        : null,
      c.user_raw_quote
        ? el('div', { class: 'nd-conn-quote' },
            '"' + c.user_raw_quote.substring(0, 150) + '"')
        : null,
    ));
  }
  return out;
}

function _connectionsSection(conns) {
  const out = [el('div', { class: 'nd-section' }, 'Connections (' + conns.length + ')')];
  for (const c of conns) {
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable',
      onclick: () => loadNodeDetail(c.id),
    },
      el('div', { class: 'nd-conn-title' },
        el('span', { class: 'type-badge type-' + (c.type || '') }, c.type || ''),
        ' ',
        (c.title || '').substring(0, 60)),
      el('div', { class: 'nd-conn-meta' },
        (c.relation || '') + ' · weight ' + (c.weight || 0).toFixed(2)),
    ));
  }
  if (!conns.length) {
    out.push(el('div', { class: 'nd-conn-empty' }, 'No connections'));
  }
  return out;
}

// ── Public entry point ─────────────────────────────────────────────────

export async function loadNodeDetail(nodeId) {
  const panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  mount(panel, el('div', { class: 'nd-loading' }, 'Loading...'));
  try {
    const [d, crData, srrData] = await Promise.all([
      api.node(nodeId),
      api.nodeCorrections(nodeId).catch(() => ({ corrections: [] })),
      api.nodeSourceRefs(nodeId).catch(() => ({ refs: [] })),
    ]);
    const corrections = (crData && crData.corrections) || [];
    const sourceRefs  = (srrData && srrData.refs) || [];
    const n           = d.node;
    const conns       = d.connections || [];
    const meta        = n.metadata || {};

    mount(panel,
      _closeButton(),
      _title(n),
      _metaStrip(n),
      _timestampsSection(n),
      el('div', { class: 'nd-section' }, 'Content'),
      el('div', { class: 'nd-content' }, n.content || '(empty)'),
      _fieldsSection(n, meta),
      _sourceRefsSection(sourceRefs),
      _correctionsSection(corrections),
      _connectionsSection(conns),
    );
  } catch(e) {
    mount(panel, el('div', { class: 'feed-empty feed-empty--error' },
      'Failed to load: ' + (e && e.message ? e.message : String(e))));
  }
}

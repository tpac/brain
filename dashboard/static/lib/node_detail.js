// ===========================================================================
// lib/node_detail.js — the right-side node detail panel.
// ---------------------------------------------------------------------------
// Called from many tabs (Graph onNodeClick, Explorer card click, Live entries
// clicking a candidate, recursive open-correction clicks). Lives in lib/ —
// not a tab module — because it's a shared overlay, not tab-scoped.
//
// The overlay element (#node-detail) lives in index.html so any caller can
// open it. We don't construct/destroy it per-call.
//
// Renders via `el()` from lib/dom.js — every string child is auto-escaped,
// every onclick is wired with addEventListener (no inline onclick="..." in
// rendered HTML). The function is split into one section helper per area
// so the read order matches the visual top-down: title → meta → content →
// fields → source-refs → corrections → connections.
// ===========================================================================

import { api } from './api.js';
import { el, html, mount, localTime } from './dom.js';
import { SCALE_COLORS } from './scales.js';

// ── Section helpers (each returns an array of elements, or [] when empty) ──

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
    el('span', null, localTime(n.created_at)),
  );
}

function _fieldsSection(n, meta) {
  const fields = [];
  const pushField = (label, value, extra) => {
    if (!value) return;
    fields.push(el('div', { class: 'nd-field' },
      el('span', { class: 'nd-fk' }, label + ':'),
      ' ',
      extra || value,
    ));
  };

  pushField('situation',           n.situation);
  pushField('reasoning',           meta.reasoning);
  pushField('user_raw_quote',      meta.user_raw_quote,
            el('em', null, '"' + meta.user_raw_quote + '"'));
  pushField('correction_of',       meta.correction_of,
            el('a', {
              class: 'nd-correction-of',
              onclick: () => loadNodeDetail(meta.correction_of),
            }, meta.correction_of));
  pushField('correction_pattern',  meta.correction_pattern);
  pushField('source_context',      meta.source_context);
  pushField('personal',            n.personal);
  pushField('evolution_status',    n.evolution_status);
  if (n.revised_at) pushField('revised', localTime(n.revised_at));

  if (!fields.length) return [];
  return [el('div', { class: 'nd-section' }, 'Fields'), ...fields];
}

// Episodic refs — which traces this node was encoded from. v27 substrate
// (node_source_refs). Empty = pre-v27 node OR no refs attached at encode.
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
      class: 'nd-conn',
      style: { borderLeftColor: sc },
    },
      el('div', { class: 'nd-conn-tag', style: { color: sc } }, tagText),
      el('div', { class: 'nd-conn-summary' }, (ref.summary || '').substring(0, 180)),
      el('div', { class: 'nd-conn-trace' },
        'trace ' + (ref.trace_id || '').substring(0, 8)
        + ' · pos ' + (ref.position || 1)
        + ' · ' + localTime(ref.trace_created_at)),
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
      class: 'nd-conn',
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
      class: 'nd-conn',
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
    // Fan out three calls in parallel:
    //   node          — base node + connections (direct SQL, works without daemon)
    //   corrections   — aspect-edge walk (via daemon; falls back to [] if down)
    //   sourceRefs    — episodic refs from node_source_refs (v27)
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

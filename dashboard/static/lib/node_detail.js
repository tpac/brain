// ===========================================================================
// lib/node_detail.js — the right-side node detail panel.
// ---------------------------------------------------------------------------
// Called from many tabs (Graph onNodeClick, Explorer card click, Live entries
// clicking a candidate, recursive open-correction clicks). Lives in lib/ —
// not a tab module — because it's a shared overlay, not tab-scoped.
//
// The overlay element (#node-detail) lives in index.html so any caller can
// open it. We don't construct/destroy it per-call.
// ===========================================================================

import { api } from './api.js';
import { escapeHtml, localTime } from './dom.js';
import { SCALE_COLORS } from './scales.js';

export async function loadNodeDetail(nodeId) {
  const panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  panel.innerHTML = '<div style="color:#666;padding:20px">Loading...</div>';
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
    const sourceRefs = (srrData && srrData.refs) || [];
    const n = d.node;
    const conns = d.connections || [];
    const meta = n.metadata || {};
    let h = '';
    h += '<div class="nd-close" onclick="document.getElementById(&quot;node-detail&quot;).style.display=&quot;none&quot;">&times;</div>';
    h += '<div class="nd-title"><span class="type-badge type-' + (n.type||'') + '">' + (n.type||'') + '</span> ';
    if (n.locked) h += '&#x1f512; ';
    if (n.critical) h += '⚠️ ';
    h += escapeHtml(n.title || '') + '</div>';
    h += '<div class="nd-meta">';
    h += '<span>id: ' + (n.id||'').substring(0,8) + '</span>';
    h += '<span>accessed: ' + n.access_count + 'x</span>';
    h += '<span>conf: ' + (n.confidence||0).toFixed(2) + '</span>';
    h += '<span>source: ' + (n.encoding_source||'?') + '</span>';
    h += '<span>' + localTime(n.created_at) + '</span>';
    h += '</div>';
    h += '<div class="nd-section">Content</div>';
    h += '<div class="nd-content">' + escapeHtml(n.content || '(empty)') + '</div>';
    const fields = [];
    if (n.situation) fields.push('<div class="nd-field"><span class="nd-fk">situation:</span> ' + escapeHtml(n.situation) + '</div>');
    if (meta.reasoning) fields.push('<div class="nd-field"><span class="nd-fk">reasoning:</span> ' + escapeHtml(meta.reasoning) + '</div>');
    if (meta.user_raw_quote) fields.push('<div class="nd-field"><span class="nd-fk">user_raw_quote:</span> <em>"' + escapeHtml(meta.user_raw_quote) + '"</em></div>');
    if (meta.correction_of) fields.push('<div class="nd-field"><span class="nd-fk">correction_of:</span> <a style="color:#7eb8ff;cursor:pointer" onclick="loadNodeDetail(&quot;' + meta.correction_of + '&quot;)">' + meta.correction_of + '</a></div>');
    if (meta.correction_pattern) fields.push('<div class="nd-field"><span class="nd-fk">correction_pattern:</span> ' + escapeHtml(meta.correction_pattern) + '</div>');
    if (meta.source_context) fields.push('<div class="nd-field"><span class="nd-fk">source_context:</span> ' + escapeHtml(meta.source_context) + '</div>');
    if (n.personal) fields.push('<div class="nd-field"><span class="nd-fk">personal:</span> ' + escapeHtml(n.personal) + '</div>');
    if (n.evolution_status) fields.push('<div class="nd-field"><span class="nd-fk">evolution_status:</span> ' + escapeHtml(n.evolution_status) + '</div>');
    if (n.revised_at) fields.push('<div class="nd-field"><span class="nd-fk">revised:</span> ' + localTime(n.revised_at) + '</div>');
    if (fields.length) h += '<div class="nd-section">Fields</div>' + fields.join('');

    // Episodic refs — which traces this node was encoded from. v27 substrate
    // (node_source_refs). If empty, either pre-v27 node OR no refs were
    // attached at encode time.
    if (sourceRefs.length) {
      h += '<div class="nd-section">Encoded from ' + sourceRefs.length + ' trace(s)</div>';
      for (const ref of sourceRefs) {
        if (ref.missing) {
          h += '<div class="nd-field" style="opacity:0.5">trace ' + (ref.trace_id||'') + ' (not found — log-rotated or archived)</div>';
          continue;
        }
        const sc = SCALE_COLORS[ref.scale] || '#666';
        const sess = ref.session_id ? ref.session_id.substring(0,8) : '';
        h += '<div class="nd-conn" style="border-left-color:' + sc + '">';
        h += '<div style="font-size:9px;color:' + sc + ';text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">' + ref.scale + ' · ' + (ref.event_type || '') + (ref.ref_type ? ' · ' + ref.ref_type : '') + (sess ? ' · ' + sess : '') + '</div>';
        h += '<div style="color:#ccc;font-size:11px">' + escapeHtml((ref.summary || '').substring(0,180)) + '</div>';
        h += '<div style="color:#555;font-size:9px;margin-top:2px">trace ' + (ref.trace_id||'').substring(0,8) + ' · pos ' + (ref.position || 1) + ' · ' + localTime(ref.trace_created_at) + '</div>';
        h += '</div>';
      }
    }

    // Corrections — aspect-edge walk via daemon. Direction matters: 'corrects'
    // means the neighbor IS this node's corrector; 'corrected_by' is the inverse.
    if (corrections.length) {
      h += '<div class="nd-section">Corrections (' + corrections.length + ')</div>';
      for (const c of corrections) {
        const dirColor = c.direction === 'corrects' ? '#ff8866' : '#88ccff';
        const dirLabel = c.direction === 'corrects' ? '⤴ corrects this' : '↪ corrected by this';
        h += '<div class="nd-conn" onclick="loadNodeDetail(&quot;' + (c.id||'') + '&quot;)" style="border-left-color:' + dirColor + '">';
        h += '<div style="font-size:9px;color:' + dirColor + ';text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">' + dirLabel + ' · ' + escapeHtml(c.relation || '') + '</div>';
        h += '<div class="nd-conn-title"><span class="type-badge type-' + (c.type||'') + '">' + (c.type||'') + '</span> ' + escapeHtml((c.title||'').substring(0,80)) + '</div>';
        if (c.edge_description) h += '<div style="color:#888;font-size:10px;margin-top:3px;font-style:italic">why: ' + escapeHtml(c.edge_description.substring(0,180)) + '</div>';
        if (c.content) h += '<div style="color:#aaa;font-size:11px;margin-top:3px">' + escapeHtml(c.content.substring(0,200)) + '</div>';
        if (c.user_raw_quote) h += '<div style="color:#7eb8ff;font-size:10px;margin-top:3px">"' + escapeHtml(c.user_raw_quote.substring(0,150)) + '"</div>';
        h += '</div>';
      }
    }

    h += '<div class="nd-section">Connections (' + conns.length + ')</div>';
    for (const c of conns) {
      h += '<div class="nd-conn" onclick="loadNodeDetail(&quot;' + c.id + '&quot;)">';
      h += '<div class="nd-conn-title"><span class="type-badge type-' + (c.type||'') + '">' + (c.type||'') + '</span> ' + escapeHtml((c.title||'').substring(0,60)) + '</div>';
      h += '<div class="nd-conn-meta">' + (c.relation||'') + ' · weight ' + (c.weight||0).toFixed(2) + '</div>';
      h += '</div>';
    }
    if (!conns.length) h += '<div style="color:#555;padding:8px">No connections</div>';
    panel.innerHTML = h;
  } catch(e) {
    panel.innerHTML = '<div style="color:#ff6666;padding:20px">Failed to load: ' + e.message + '</div>';
  }
}

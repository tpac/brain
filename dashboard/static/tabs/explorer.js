// ===========================================================================
// tabs/explorer.js — node search + browse.
// ---------------------------------------------------------------------------
// Click a node-card → opens the right-side detail panel via lib/node_detail.js.
// The type-filter dropdown is populated by app.js's loadStats() (cross-tab
// concern), this module only reads the dropdown's current value on search.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { localTime } from '/static/lib/dom.js';

export async function searchNodes() {
  const search = document.getElementById('search-input').value;
  const type = document.getElementById('type-filter').value;
  try {
    const d = await api.nodes({ limit: 100, search: search || undefined, type: type || undefined });
    const list = document.getElementById('node-list');
    list.innerHTML = d.nodes.map(n => `
      <div class="node-card" onclick="loadNodeDetail('${n.id}')" style="cursor:pointer">
        <div class="node-title">
          <span class="type-badge type-${n.type}">${n.type}</span>
          ${n.locked ? '<span class="locked-icon">&#x1f512;</span>' : ''}
          ${n.title || '(untitled)'}
        </div>
        <div class="node-meta">
          <span>conf: ${(n.confidence||0).toFixed(2)}</span>
          <span>accessed: ${n.access_count}x</span>
          <span>${n.encoding_source || ''}</span>
          <span>${localTime(n.created_at)}</span>
        </div>
      </div>
    `).join('');
  } catch(e) { console.error('[dashboard] searchNodes failed:', e); }
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // No polling — explorer is search-on-demand. The dropdown is wired via
  // index.html's onkeyup/onchange handlers calling window.searchNodes.
}

export function activate() {
  searchNodes();
}

export function deactivate() {}

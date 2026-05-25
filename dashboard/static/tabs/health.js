// ===========================================================================
// tabs/health.js — system status grid + aspect taxonomy + health summary.
// ---------------------------------------------------------------------------
// Three render targets:
//   #status-grid    — Daemon/DB/Judge/Embedder cards (polled, 5s)
//   #aspects-grid   — 14 aspects with type + relation chips
//   #health-content — node/edge totals + insights + type breakdown
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime } from '/static/lib/dom.js';

async function loadSystemStatus() {
  try {
    const d = await api.systemStatus();
    const grid = document.getElementById('status-grid');
    grid.innerHTML = '';

    const components = [
      {key: 'daemon', label: 'Brain Daemon', icon: '🧠'},
      {key: 'brain_db', label: 'Brain DB', icon: '💾'},
      {key: 'logs_db', label: 'Logs DB', icon: '📋'},
      {key: 'judge', label: 'Haiku Judge', icon: '⚖️'},
      {key: 'embedder', label: 'Embedder', icon: '🔮'},
    ];

    for (const comp of components) {
      const s = d.status[comp.key] || {alive: false, error: 'unknown'};
      const alive = s.alive;
      const card = document.createElement('div');
      card.style.cssText = 'background:#111118;border-radius:8px;padding:12px 16px;border:1px solid ' + (alive ? '#1a3a1a' : '#3a1a1a');

      let details = '';
      if (comp.key === 'daemon' && alive) {
        details = 'PID: ' + (s.pid || '?') + ' · Uptime: ' + Math.round((s.uptime || 0) / 60) + 'min';
      } else if (comp.key === 'brain_db' && alive) {
        details = s.nodes + ' nodes · ' + (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'logs_db' && alive) {
        details = (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'dashboard_db' && alive) {
        details = (s.size_mb || '?') + 'MB · Last: ' + localTime(s.last_entry);
      } else if (comp.key === 'embedder' && alive) {
        details = s.model || '?';
      } else if (!alive) {
        details = s.error || 'unreachable';
      }

      const pathLine = s.path ? '<div style="font-size:9px;color:#444;margin-top:4px;word-break:break-all">' + escapeHtml(s.path) + '</div>' : '';
      card.innerHTML =
        '<div style="display:flex;align-items:center;gap:8px">' +
          '<span style="font-size:20px">' + comp.icon + '</span>' +
          '<div>' +
            '<div style="color:#ccc;font-weight:bold;font-size:13px">' + comp.label + '</div>' +
            '<div style="font-size:11px;margin-top:2px;color:' + (alive ? '#4a4' : '#f44') + '">' +
              (alive ? '● Live' : '● Down') +
            '</div>' +
          '</div>' +
          '<div style="margin-left:auto;font-size:10px;color:#666;text-align:right;max-width:200px;overflow:hidden;text-overflow:ellipsis">' + escapeHtml(details) + '</div>' +
        '</div>' + pathLine;
      grid.appendChild(card);
    }
  } catch(e) {
    document.getElementById('status-grid').innerHTML = '<div style="color:#f66;padding:20px">Failed to load status: ' + e + '</div>';
  }
}

// Aspect taxonomy — 14 aspects classifying node_types + edge_relations.
// Source: aspects_v1.json (live at $BRAIN_DB_DIR/aspects_v1.json, seed in
// servers/scales/s2/aspects_v1.json). Counts come from brain.db.
async function loadAspects() {
  try {
    const d = await api.aspects();
    const grid = document.getElementById('aspects-grid');
    if (!grid) return;
    const aspects = d.aspects || [];
    if (!aspects.length) {
      grid.innerHTML = '<div style="color:#666;padding:12px">aspects_v1.json not found or empty</div>';
      return;
    }
    let html = '';
    for (const a of aspects) {
      const lockBadge = a.locked ? ' <span style="color:#ffaa33;font-size:9px">🔒</span>' : '';
      const dim = a.dimension ? '<span style="color:#555;font-size:10px;margin-left:6px">' + escapeHtml(a.dimension) + '</span>' : '';
      const topTypes = a.node_types.slice(0, 12);
      const moreTypes = a.node_types.length > 12 ? a.node_types.length - 12 : 0;
      const topRels = a.edge_relations.slice(0, 12);
      const moreRels = a.edge_relations.length > 12 ? a.edge_relations.length - 12 : 0;
      const chip = (label, count, color) =>
        '<span style="display:inline-block;background:#1a1a2a;border:1px solid #2a2a3a;color:' + color + ';padding:1px 6px;border-radius:3px;font-size:10px;margin:2px 3px 0 0">' +
        escapeHtml(label) + '<span style="color:#555;margin-left:4px">' + count + '</span></span>';
      html += '<div style="background:#111118;border-radius:8px;padding:12px;margin:6px 0;border-left:3px solid #45B7D1">';
      html += '<div style="display:flex;justify-content:space-between;align-items:baseline">';
      html += '<div><span style="color:#7eb8ff;font-weight:bold;font-size:13px">' + escapeHtml(a.name) + '</span>' + lockBadge + dim + '</div>';
      html += '<div style="color:#666;font-size:10px">' + a.totals.nodes + ' nodes · ' + a.totals.edges + ' edges</div>';
      html += '</div>';
      if (a.meaning) html += '<div style="color:#888;font-size:11px;margin-top:6px;line-height:1.4">' + escapeHtml(a.meaning) + '</div>';
      if (topTypes.length) {
        html += '<div style="margin-top:6px"><div style="color:#555;font-size:9px;text-transform:uppercase;letter-spacing:0.5px">Node types</div><div>';
        for (const t of topTypes) html += chip(t.name, t.count, '#ccc');
        if (moreTypes) html += '<span style="color:#555;font-size:10px;margin-left:4px">+' + moreTypes + '</span>';
        html += '</div></div>';
      }
      if (topRels.length) {
        html += '<div style="margin-top:6px"><div style="color:#555;font-size:9px;text-transform:uppercase;letter-spacing:0.5px">Edge relations</div><div>';
        for (const r2 of topRels) html += chip(r2.name, r2.count, '#aa66ff');
        if (moreRels) html += '<span style="color:#555;font-size:10px;margin-left:4px">+' + moreRels + '</span>';
        html += '</div></div>';
      }
      html += '</div>';
    }
    grid.innerHTML = html;
  } catch(e) { console.error('loadAspects error:', e); }
}

async function loadHealth() {
  try {
    const d = await api.stats();
    const ins = await api.insights();
    const hc = document.getElementById('health-content');
    const orphanClass = d.orphans > 20 ? 'bad' : d.orphans > 5 ? 'warn' : 'ok';
    const sevColors = {high: '#ff6666', medium: '#ffaa33', low: '#7eb8ff'};
    const insightsHtml = (ins.insights || []).map(i => `
      <div style="background:#111118;border-radius:8px;padding:14px;margin:8px 0;border-left:4px solid ${sevColors[i.severity] || '#555'}">
        <div style="font-size:15px;font-weight:bold;color:${sevColors[i.severity]}">${i.icon} ${i.title}</div>
        <div style="color:#999;margin-top:6px;font-size:12px;line-height:1.5">${i.detail}</div>
        ${i.nodes ? '<div style="margin-top:8px;font-size:11px;color:#666">' + i.nodes.map(n =>
          '<div style="padding:2px 0">&#8226; ' + (n.title||'').substring(0,80) + ' <span style="color:#555">(' + (n.type||n.count||'') + ')</span></div>'
        ).join('') + '</div>' : ''}
      </div>
    `).join('');
    hc.innerHTML = `
      <div class="health-grid">
        <div class="health-card ok"><div class="hc-value">${d.nodes}</div><div class="hc-label">Total Nodes</div></div>
        <div class="health-card ok"><div class="hc-value">${d.edges}</div><div class="hc-label">Total Edges</div></div>
        <div class="health-card ok"><div class="hc-value">${d.locked}</div><div class="hc-label">Locked</div></div>
        <div class="health-card ${d.recent_24h > 0 ? 'ok' : 'warn'}"><div class="hc-value">${d.recent_24h}</div><div class="hc-label">Last 24h</div></div>
        <div class="health-card ${orphanClass}"><div class="hc-value">${d.orphans}</div><div class="hc-label">Orphans</div></div>
      </div>
      ${insightsHtml ? '<h3 style="color:#ccc;margin:20px 0 8px">Anchor Insights</h3>' + insightsHtml : '<div style="color:#33ff88;padding:20px;text-align:center;font-size:16px">No issues detected</div>'}
      <h3 style="color:#888;margin:20px 0 8px">Node Types</h3>
      <div class="health-grid">
        ${Object.entries(d.types).map(([t,c]) => `
          <div class="health-card ok" style="padding:10px">
            <span class="type-badge type-${t}">${t}</span>
            <span style="float:right;font-size:18px;font-weight:bold;color:#7eb8ff">${c}</span>
          </div>
        `).join('')}
      </div>
    `;
  } catch(e) { console.error(e); }
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  poll.register({
    key: 'system-status',
    interval: 5000,
    activeWhen: () => {
      // Note: legacy "#tab-status" id existed in an earlier version. Check
      // both so this doesn't silently stop refreshing if Health is renamed.
      const t = document.getElementById('tab-health') || document.getElementById('tab-status');
      return t && t.classList.contains('active');
    },
    fetcher: loadSystemStatus,
  });
}

export function activate() {
  loadHealth();
  loadSystemStatus();
  loadAspects();
}

export function deactivate() {}

// ===========================================================================
// tabs/graph.js — 3D ForceGraph + search-driven highlighter +
//                 persistent recall-highlight ('spotlight') mode.
// ---------------------------------------------------------------------------
// Lifecycle contract (shared by every tabs/*.js module):
//
//   init()         called once on app boot. Wires bus subs.
//   activate()     called when Live tab becomes visible. Lazy-loads
//                  graph data + sizes renderer to container.
//   deactivate()   no-op — 3D scene keeps animating in background.
//
// Since the P2.2 layout pivot the graph mounts inside Live's left pane;
// `activate()` is now driven by live.activate().
//
// ── Visual model ─────────────────────────────────────────────────────
//
// The graph has TWO independent dimming axes that combine via AND:
//
//   1. SEARCH    — type a query into the search box. Non-matching nodes
//                  go dark. Empty query = everything passes.
//
//   2. HIGHLIGHT — when a recall lands, the surfaced nodes become the
//                  "highlight set" (persistent — no decay). All non-set
//                  nodes go dark. Empty set = everything passes.
//                  Cleared by clicking Refresh.
//
// A node only renders in its community color if it passes BOTH gates.
// Within the highlight set, tier (used/activation/returned) drives the
// color blend + size bump:
//
//   used        white blend, 1.0× intensity   — judge picked these
//   activation  green blend, 0.7× intensity   — spread-expanded
//   returned    blue blend,  0.4× intensity   — candidate pool
//
// ── Highlight mode ───────────────────────────────────────────────────
//
//   latest         (default) — every new recall REPLACES the set.
//                  Pre-loaded with the most recent recall on switch.
//   session=<id>   union of every recall whose session_id matches.
//                  Pre-loaded with that session's history on switch.
//
// The mode dropdown lives in .graph-controls; session options are
// injected by _populateSessionOptions() on activate.
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { escapeHtml } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

let graph3d = null;
let graph3dData = null;

// ── Search state ──────────────────────────────────────────────────────

let _searchQuery = '';

function _nodeMatches(node) {
  if (!_searchQuery) return true;
  const q = _searchQuery;
  return (node.name || '').toLowerCase().includes(q)
      || (node.type || '').toLowerCase().includes(q)
      || (node.community_title || '').toLowerCase().includes(q);
}

// ── Highlight state ───────────────────────────────────────────────────

const DIM_COLOR = '#1a1a1a';
const HIGHLIGHT_SIZE_MULT = 1.8;

// tier → { color, intensity }. Used for color blend + size bump within
// the highlight set. Adding a tier means a new label here + extending
// HIGHLIGHT_TIER_ORDER.
const HIGHLIGHT_TIERS = {
  used:       { color: '#ffffff', intensity: 1.00 },
  activation: { color: '#33ff88', intensity: 0.70 },
  returned:   { color: '#7eb8ff', intensity: 0.40 },
};
// Weakest → strongest so later writes win in the per-id Map.
const HIGHLIGHT_TIER_ORDER = ['returned', 'activation', 'used'];

// nodeId → tier string. Persistent (no decay). Empty = no highlight
// active = all nodes show their community color.
const _highlightTier = new Map();
// 'latest' (default) — each recall REPLACES the set.
// 'session'          — union of recalls for `_highlightSessionId`.
let _highlightMode = 'latest';
let _highlightSessionId = null;

// ── Color math ────────────────────────────────────────────────────────

function _hexToRgb(hex) {
  const m = hex.replace('#', '');
  return {
    r: parseInt(m.slice(0, 2), 16),
    g: parseInt(m.slice(2, 4), 16),
    b: parseInt(m.slice(4, 6), 16),
  };
}
function _rgbToHex(r, g, b) {
  const h = v => Math.max(0, Math.min(255, v|0)).toString(16).padStart(2, '0');
  return '#' + h(r) + h(g) + h(b);
}
function _blend(c1, c2, t) {
  const a = _hexToRgb(c1), b = _hexToRgb(c2);
  return _rgbToHex(a.r * t + b.r * (1 - t), a.g * t + b.g * (1 - t), a.b * t + b.b * (1 - t));
}

// ── Per-node color + size resolvers ────────────────────────────────

function _colorFor(n) {
  // A node renders in color iff it passes BOTH gates:
  //   - search:    matches the current query (or no query)
  //   - highlight: in the set (or no set is active)
  const setActive = _highlightTier.size > 0;
  const inHighlight = !setActive || _highlightTier.has(n.id);
  if (!_nodeMatches(n) || !inHighlight) return DIM_COLOR;

  // In-highlight nodes get tier-blended color (when the set is active).
  // When no set is active, just show the community color.
  const tierName = setActive ? _highlightTier.get(n.id) : null;
  const baseColor = n.color || '#666';
  if (!tierName) return baseColor;
  const tier = HIGHLIGHT_TIERS[tierName];
  return _blend(tier.color, baseColor, tier.intensity);
}

function _valFor(n) {
  const base = n.hub ? n.val : 2;
  const tierName = _highlightTier.get(n.id);
  if (!tierName) return base;
  const tier = HIGHLIGHT_TIERS[tierName];
  return base * (1 + (HIGHLIGHT_SIZE_MULT - 1) * tier.intensity);
}

// Force ForceGraph3D to re-evaluate per-node color/size. The library
// treats setter calls with the SAME function reference as no-ops, so we
// pass fresh closures each time we need a refresh.
function _refreshGraph() {
  if (!graph3d) return;
  graph3d.nodeColor(n => _colorFor(n));
  graph3d.nodeVal(n => _valFor(n));
}

// ── Search public surface ────────────────────────────────────────────

function _focusFirstMatch() {
  if (!graph3d || !_searchQuery) return;
  const nodes = graph3d.graphData().nodes;
  const matches = nodes.filter(_nodeMatches);
  if (!matches.length) return;
  const target = matches.find(n => n.hub) || matches[0];
  graph3d.cameraPosition(
    { x: target.x + 120, y: target.y + 60, z: target.z + 120 },
    target,
    1200,
  );
}

function _updateMatchCount() {
  const el = document.getElementById('graph-search-count');
  if (!el) return;
  if (!_searchQuery || !graph3d) { el.textContent = ''; return; }
  const nodes = graph3d.graphData().nodes;
  const n = nodes.filter(_nodeMatches).length;
  el.textContent = n + ' match' + (n === 1 ? '' : 'es');
}

export function setSearchQuery(q) {
  _searchQuery = (q || '').toLowerCase().trim();
  _refreshGraph();
  _updateMatchCount();
}

export function onGraphSearch() {
  const input = document.getElementById('graph-search');
  setSearchQuery(input ? input.value : '');
}

export function onGraphSearchKey(event) {
  if (event && event.key === 'Enter') {
    onGraphSearch();
    _focusFirstMatch();
  }
}

// ── Highlight + mode public surface ──────────────────────────────────

// Layer a recall event into the highlight set. Weakest tier first so
// stronger tiers overwrite (Map.set last-write).
function _applyEventToHighlight(event) {
  if (!event) return;
  const idsByTier = {
    returned:   event.returned_ids   || [],
    activation: event.activation_ids || [],
    used:       event.used_ids       || [],
  };
  for (const tier of HIGHLIGHT_TIER_ORDER) {
    for (const id of idsByTier[tier]) _highlightTier.set(id, tier);
  }
}

function _onRecallEvent({ event }) {
  if (!graph3d || !event) return;
  if (_highlightMode === 'session') {
    // Only union recalls for the watched session — ignore others.
    if (event.session_id !== _highlightSessionId) return;
  } else {
    // 'latest': replace the entire set with just this recall.
    _highlightTier.clear();
  }
  _applyEventToHighlight(event);
  _refreshGraph();
}

// Switch highlight mode. Clears the existing set, then pre-loads:
//   latest      → the single most recent recall (so the spotlight is
//                 immediately visible without waiting for the next one)
//   session=X   → every recall for session X, unioned into the set
async function setHighlightMode(mode, sessionId) {
  _highlightMode = (mode === 'session' && sessionId) ? 'session' : 'latest';
  _highlightSessionId = _highlightMode === 'session' ? sessionId : null;
  _highlightTier.clear();
  try {
    if (_highlightMode === 'latest') {
      const d = await api.recalls({ limit: 1 });
      const evt = (d.events || [])[0];
      if (evt) _applyEventToHighlight(evt);
    } else {
      const d = await api.recalls({ limit: 200, session_id: sessionId });
      for (const evt of (d.events || [])) _applyEventToHighlight(evt);
    }
  } catch (e) {
    console.error('[graph] highlight mode preload failed:', e);
  }
  _refreshGraph();
}

export function onGraphHighlightModeChange() {
  const sel = document.getElementById('graph-highlight-mode');
  if (!sel) return;
  const v = sel.value;
  if (v === 'latest') return setHighlightMode('latest');
  if (v.startsWith('session:')) return setHighlightMode('session', v.slice('session:'.length));
}

// Refresh = clear the highlight + reload the graph data from /api/graph3d.
export function onGraphRefresh() {
  _highlightTier.clear();
  loadGraph3D();
}

// Populate the mode dropdown with session options. Called once on
// activate(); not polled — sessions change rarely and the dropdown
// only matters when the user is actively switching modes.
async function _populateSessionOptions() {
  const sel = document.getElementById('graph-highlight-mode');
  if (!sel) return;
  try {
    const sessions = await api.sessions();
    // Preserve "Latest recall" as option 0; replace any prior session-*.
    while (sel.options.length > 1) sel.remove(1);
    for (const s of (sessions || [])) {
      const opt = document.createElement('option');
      opt.value = 'session:' + s.id;
      opt.textContent = 'Session: ' + s.short + ' (' + s.events + ' events)';
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error('[graph] session list load failed:', e);
  }
}

// ── Graph load + lifecycle ────────────────────────────────────────────

function _renderGraphError(message, hint, hintHTML) {
  const container = document.getElementById('graph-3d');
  if (!container) return;
  container.innerHTML =
    '<div class="graph-error">' +
      '<div class="graph-error-title">3D graph unavailable</div>' +
      '<div class="graph-error-msg">' + escapeHtml(message || 'unknown error') + '</div>' +
      (hintHTML ? '<div class="graph-error-hint">' + hintHTML + '</div>'
                : (hint ? '<div class="graph-error-hint">' + escapeHtml(hint) + '</div>' : '')) +
    '</div>';
}

// Pre-flight WebGL check. ForceGraph3D's THREE.WebGLRenderer constructor
// catches its own context-creation failure but logs a noisy stack first;
// detecting up-front gives us a friendlier error AND avoids the noise.
// Returns null on success, or a {reason, hintHTML} on failure.
function _detectWebGLBlocker() {
  // 1. Library loaded?
  if (typeof window.ForceGraph3D !== 'function') {
    return {
      reason: 'ForceGraph3D library failed to load.',
      hintHTML:
        'The CDN script (<code>unpkg.com/3d-force-graph</code>) didn\'t finish loading. ' +
        'Check the Network tab — likely a network/CSP block. Reload the page after the ' +
        'CDN is reachable.',
    };
  }
  // 2. WebGL context creatable?
  let gl = null;
  try {
    const c = document.createElement('canvas');
    gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
  } catch (e) { /* fall through */ }
  if (!gl) {
    return {
      reason: 'Browser refused to create a WebGL context.',
      hintHTML:
        'Open <code>chrome://gpu</code> in a new tab and look for "WebGL" status. ' +
        'If it says "Hardware accelerated" but you still see this, try: ' +
        '<ol style="text-align:left;margin:6px 0 0 18px;padding:0">' +
          '<li>chrome://settings → System → "Use hardware acceleration when available" ON, then restart Chrome.</li>' +
          '<li>chrome://flags → search "WebGL" → ensure none are explicitly disabled.</li>' +
          '<li>Hard refresh this page (Cmd+Shift+R).</li>' +
        '</ol>',
    };
  }
  return null;
}

export async function loadGraph3D() {
  // Pre-flight: bail with a clean error if the browser can't render at all.
  // Avoids the noisy THREE.WebGLRenderer console stack trace and gives the
  // operator actionable Chrome-specific hints instead of a generic catch.
  const block = _detectWebGLBlocker();
  if (block) {
    _renderGraphError(block.reason, null, block.hintHTML);
    return;
  }
  try {
    graph3dData = await api.graph3d();
    if (!graph3dData.nodes || !graph3dData.nodes.length) {
      _renderGraphError('No graph data returned',
        'The /api/graph3d endpoint returned an empty payload. Check daemon health on the Logs tab.');
      return;
    }

    // Filter: only show nodes IN communities + community hub nodes.
    // Orphans hidden — they clutter without adding structure.
    const communityNodeIds = new Set();
    const hubIds = new Set();
    graph3dData.nodes.forEach(n => {
      if (n.hub) hubIds.add(n.id);
      if (n.community) communityNodeIds.add(n.id);
    });
    hubIds.forEach(id => communityNodeIds.add(id));

    const visibleNodes = graph3dData.nodes.filter(
      n => communityNodeIds.has(n.id) || hubIds.has(n.id));
    const visibleIds = new Set(visibleNodes.map(n => n.id));

    const visibleLinks = graph3dData.edges
      .filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
      .filter(e => e.relation !== 'co_accessed' && e.relation !== 'emergent_bridge')
      .map(e => ({source: e.source, target: e.target, relation: e.relation}));

    const container = document.getElementById('graph-3d');
    const w = container.offsetWidth || 800;
    const h = container.offsetHeight || 600;

    if (graph3d) {
      graph3d.graphData({nodes: visibleNodes, links: visibleLinks});
    } else {
      graph3d = ForceGraph3D()(container)
        .width(w).height(h)
        .graphData({nodes: visibleNodes, links: visibleLinks})
        .backgroundColor('#08080f')
        .nodeVal(n => _valFor(n))
        .nodeColor(n => _colorFor(n))
        .nodeOpacity(0.85)
        .nodeLabel(n => {
          if (n.hub) return '<div style="text-align:center;font-size:14px"><b>' + n.name + '</b><br><span style="color:#aaa">' + (n.val/0.8|0) + ' members</span></div>';
          const comm = n.community_title ? '<br><span style="color:#666">' + n.community_title.substring(0, 40) + '</span>' : '';
          return '<div style="text-align:center"><b>' + n.name + '</b><br><span style="color:#888">' + n.type + '</span>' + comm + '</div>';
        })
        .linkColor(l => l.relation === 'community_member' ? '#333' : '#222')
        .linkOpacity(l => l.relation === 'community_member' ? 0.15 : 0.08)
        .linkWidth(l => l.relation === 'community_member' ? 0.3 : 0.15)
        .d3AlphaDecay(0.08)
        .d3VelocityDecay(0.5)
        .warmupTicks(150)
        .cooldownTicks(300)
        .onEngineTick(() => {
          if (!graph3d._forcesConfigured) {
            const charge = graph3d.d3Force('charge');
            if (charge) { charge.strength(-15).distanceMax(200); }
            const link = graph3d.d3Force('link');
            if (link) { link.distance(l => l.relation === 'community_member' ? 3 : 40).strength(l => l.relation === 'community_member' ? 0.9 : 0.05); }
            graph3d._forcesConfigured = true;
          }
        })
        .onNodeClick(node => {
          graph3d.cameraPosition({x: node.x + 150, y: node.y + 80, z: node.z + 150}, node, 1000);
          loadNodeDetail(node.id);
        });
      const controls = graph3d.controls();
      if (controls) controls.zoomSpeed = 5.0;
    }

    // Apply any current state (search + highlight) once nodes are live.
    _refreshGraph();
    _updateMatchCount();
  } catch(e) {
    console.error('Graph3D load failed:', e);
    graph3d = null;   // let next loadGraph3D rebuild from scratch
    const msg = (e && e.message) ? e.message : String(e);
    _renderGraphError(msg,
      /webgl|gl\b/i.test(msg)
        ? 'WebGL context could not be created. Common causes: GPU driver, browser policy, sandboxed iframe. Try reloading or opening in a regular Chrome window.'
        : 'See the Logs tab → Dashboard sub-feed for the full stack.');
  }
}

/** Resize ForceGraph3D to the current container. Run after tab-switch
 * or layout drag — three.js doesn't observe its host. */
export function resize() {
  if (!graph3d) return;
  const c = document.getElementById('graph-3d');
  if (!c) return;
  void c.offsetHeight;   // trigger reflow before reading offset*
  const w = c.offsetWidth || 800;
  const h = c.offsetHeight || 600;
  graph3d.width(w).height(h);
  graph3d.renderer().setSize(w, h);
  graph3d.camera().aspect = w / h;
  graph3d.camera().updateProjectionMatrix();
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  bus.subscribe('live:layout', () => {
    requestAnimationFrame(resize);
  });
  bus.subscribe('recall:event', _onRecallEvent);
}

export function activate() {
  // 300ms delay matches the legacy behavior: tab-content display:block
  // hasn't laid out yet by the time switchTab returns.
  setTimeout(() => {
    if (!graph3dData) loadGraph3D();
    else resize();
    // Populate the mode dropdown + apply default mode (latest) so the
    // spotlight pre-loads on first open.
    _populateSessionOptions();
    if (_highlightTier.size === 0) setHighlightMode('latest');
  }, 300);
}

export function deactivate() {
  // 3D scene keeps animating in the background — cheap; re-activating
  // a paused scene introduces a visible re-warmup.
}

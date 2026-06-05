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
//   latest   (default) — every new recall REPLACES the set. Pre-loaded
//                        with the most recent recall on first open.
//   pinned   — entered by clicking a recall card in the activity stream.
//              Locked until Refresh / the × on the pin chip; incoming
//              recalls are ignored. There is no UI dropdown — the chip
//              in .graph-controls is the only signal + escape hatch.
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
// Two modes, no dropdown:
//   'latest' (default) — each incoming recall REPLACES the set.
//   'pinned'           — set frozen to a specific event clicked from the
//                        activity stream; incoming live events are ignored
//                        until Refresh. Set by pinRecallToGraph().
// Session filtering used to live here but was removed once the activity
// stream gained its own session-filter dropdown — two parallel filters
// for the same dimension were redundant.
let _highlightMode = 'latest';
let _pinnedEventId = null;     // for the chip display while pinned

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
  // Match against the live graph only — camera pan needs the laid-out
  // x/y/z coords from the simulation, which the raw API payload lacks.
  const matches = graph3d.graphData().nodes.filter(_nodeMatches);
  if (!matches.length) return;
  const target = matches.find(n => n.hub) || matches[0];
  graph3d.cameraPosition(
    { x: target.x + 120, y: target.y + 60, z: target.z + 120 },
    target,
    1200,
  );
}

// Single source of truth for "what nodes can we filter against." Pulls
// from the live ForceGraph3D instance when available, otherwise falls
// back to the raw API payload. The fallback matters when WebGL failed
// to mount: graph3dData is populated by api.graph3d() before the canvas
// is built, so search keeps giving the user feedback ("23 matches") even
// when the visualization itself is unavailable. Returns null when neither
// is loaded.
function _searchableNodes() {
  if (graph3d) return graph3d.graphData().nodes;
  if (graph3dData?.nodes) return graph3dData.nodes;
  return null;
}

function _updateMatchCount() {
  const el = document.getElementById('graph-search-count');
  if (!el) return;
  if (!_searchQuery) { el.textContent = ''; return; }
  const nodes = _searchableNodes();
  if (!nodes) { el.textContent = ''; return; }
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
  // Pinned mode means the user clicked a specific recall to lock the
  // highlight on it — don't let auto-incoming events steal that focus.
  if (_highlightMode === 'pinned') return;
  // 'latest': replace the entire set with just this recall.
  _highlightTier.clear();
  _applyEventToHighlight(event);
  _refreshGraph();
}

// ── Hover-preview ────────────────────────────────────────────────────
// previewRecallOnGraph / clearRecallPreview let the activity stream
// "audition" a recall on the graph while the cursor is over its card,
// without committing — the prior highlight state is snapshotted on enter
// and restored on leave. Click upgrades the preview to a pin
// (pinRecallToGraph) which marks the preview as consumed so the
// subsequent mouseleave is a no-op.
let _previewSnapshot = null;   // { tier: Map, mode, pinnedEventId }

function _saveSnapshot() {
  _previewSnapshot = {
    tier: new Map(_highlightTier),
    mode: _highlightMode,
    pinnedEventId: _pinnedEventId,
  };
}

function _restoreSnapshot() {
  if (!_previewSnapshot) return;
  _highlightTier.clear();
  for (const [k, v] of _previewSnapshot.tier) _highlightTier.set(k, v);
  _highlightMode = _previewSnapshot.mode;
  _pinnedEventId = _previewSnapshot.pinnedEventId;
  _previewSnapshot = null;
  _refreshGraph();
}

export function previewRecallOnGraph(event) {
  if (!graph3d || !event) return;
  if (!_previewSnapshot) _saveSnapshot();
  _highlightTier.clear();
  _applyEventToHighlight(event);
  _refreshGraph();
}

export function clearRecallPreview() {
  if (!_previewSnapshot) return;
  _restoreSnapshot();
}

// Pin highlight to a specific past recall event — called from the
// activity stream when the operator clicks a recall card. Replaces the
// highlight set with this event's nodes and locks the mode so subsequent
// live recalls don't override. Clear by clicking Refresh or selecting a
// different mode in the dropdown. Publishes 'graph:pinned' so the
// activity stream can mark its corresponding card with the selected
// background — graph.js doesn't reach into live.js DOM directly, the
// bus topic is the boundary.
export function pinRecallToGraph(event) {
  if (!event) return;
  // Consume any in-flight preview snapshot — the user upgraded a hover
  // into a commit, so the post-hover mouseleave must not restore the
  // pre-hover state. Without this, leaving the card after a click
  // would silently undo the pin.
  _previewSnapshot = null;
  _highlightMode = 'pinned';
  _pinnedEventId = event.id || null;
  _highlightTier.clear();
  _applyEventToHighlight(event);
  _refreshGraph();
  _renderPinIndicator();
  bus.publish('graph:pinned', { eventId: _pinnedEventId });
  // Reflect the lock in the mode dropdown — Latest is no longer correct.
  // We keep the dropdown selectable so the user can leave pinned mode by
  // picking Latest or a session.
  const sel = document.getElementById('graph-highlight-mode');
  if (sel) sel.value = 'latest';   // semantically pinned overrides; chip shows the truth
}

// Small chip rendered into .graph-controls when pinned, with a × that
// returns to Latest. Removed automatically when leaving pinned mode.
function _renderPinIndicator() {
  const controls = document.querySelector('.graph-controls');
  if (!controls) return;
  let chip = document.getElementById('graph-pin-chip');
  if (_highlightMode !== 'pinned') {
    if (chip) chip.remove();
    return;
  }
  if (!chip) {
    chip = document.createElement('span');
    chip.id = 'graph-pin-chip';
    chip.className = 'graph-pin-chip';
    controls.appendChild(chip);
  }
  const idShort = (_pinnedEventId || '').toString().slice(0, 8);
  chip.innerHTML = '<span class="graph-pin-chip-label">Pinned: #' + idShort + '</span>' +
                   '<button class="graph-pin-chip-close" title="Unpin (back to Latest)">&times;</button>';
  const closeBtn = chip.querySelector('.graph-pin-chip-close');
  if (closeBtn) closeBtn.onclick = () => setHighlightMode('latest');
}

// Switch highlight mode. Only used to escape pinned mode (back to
// 'latest') — there's no longer a dropdown to drive this from the UI.
// The × on the pin chip and onGraphRefresh both call it.
async function setHighlightMode(mode) {
  _highlightMode = 'latest';   // 'pinned' is set only by pinRecallToGraph
  _pinnedEventId = null;
  _renderPinIndicator();       // removes the chip if leaving pinned mode
  bus.publish('graph:pinned', { eventId: null });
  _highlightTier.clear();
  try {
    const d = await api.recalls({ limit: 1 });
    const evt = (d.events || [])[0];
    if (evt) _applyEventToHighlight(evt);
  } catch (e) {
    console.error('[graph] latest-recall preload failed:', e);
  }
  _refreshGraph();
}

// Refresh = nuke any zombie state, then reload from scratch. We dispose
// the renderer rather than just calling graphData() because the user
// typically clicks Refresh when something feels stuck — Chrome may have
// dropped our context. Cheap rebuild beats a half-alive canvas. Also
// clears any pinned-event lock so the rebuilt graph follows live events.
export function onGraphRefresh() {
  _highlightTier.clear();
  _highlightMode = 'latest';
  _pinnedEventId = null;
  _renderPinIndicator();
  bus.publish('graph:pinned', { eventId: null });
  graph3dData = null;
  _destroyGraph();
  loadGraph3D();
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
//
// CRITICAL: Chrome caps WebGL contexts at ~16 per tab. The probe context
// MUST be explicitly released via WEBGL_lose_context — letting it fall
// out of scope leaves the slot allocated until GC, which during a busy
// session never catches up. Before this fix, each loadGraph3D() burned
// a slot, and after ~16 tab switches the real graph mount silently failed.
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
  // Always release the probe slot — see comment above.
  if (gl) {
    try { gl.getExtension('WEBGL_lose_context')?.loseContext(); } catch (_) { /* best-effort */ }
  }
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

// Fully tear down the current graph instance + DOM. Used on error paths
// and on user-triggered Refresh so we never accumulate orphan
// WebGLRenderer instances (Chrome's ~16-context-per-tab cap is the root
// cause of "flaky / works sometimes / blank after refresh"). Safe to
// call when graph3d is already null.
function _destroyGraph() {
  if (graph3d) {
    try {
      // Force the GPU slot to free immediately rather than waiting for GC.
      const renderer = graph3d.renderer?.();
      renderer?.forceContextLoss?.();
      renderer?.dispose?.();
    } catch (_) { /* best-effort */ }
    try {
      // ForceGraph3D exposes _destructor for clean unmount.
      graph3d._destructor?.();
    } catch (_) { /* best-effort */ }
    graph3d = null;
  }
  // Either way, wipe the host element — a half-mounted canvas can linger
  // here even when the JS handle is null (e.g. mount threw mid-init).
  const c = document.getElementById('graph-3d');
  if (c) c.innerHTML = '';
}

// Wire webglcontextlost / webglcontextrestored on the renderer's canvas.
// Called once after the initial mount. Context loss = Chrome's GPU
// process crashed or VRAM ran out — the #1 cause of flakiness per the
// 2026 troubleshooting threads. Without this, the canvas goes black and
// the user sees no signal. With it, they get a recovery prompt + auto-
// rebuild on restore.
function _wireContextLossHandlers() {
  const canvas = graph3d?.renderer?.()?.domElement;
  if (!canvas) return;
  canvas.addEventListener('webglcontextlost', (e) => {
    // preventDefault is required for the restore event to fire later.
    e.preventDefault();
    console.warn('[graph] WebGL context lost — GPU process crash or VRAM exhaustion');
    _destroyGraph();
    _renderGraphError(
      'GPU context lost.',
      null,
      'Chrome\'s GPU process likely crashed or VRAM filled up. ' +
      'Hard refresh (Cmd+Shift+R) — if it keeps happening, check ' +
      '<code>chrome://gpu</code> for repeated GPU crashes.'
    );
  });
  canvas.addEventListener('webglcontextrestored', () => {
    console.info('[graph] WebGL context restored — reloading graph');
    loadGraph3D();
  });
}

export async function loadGraph3D() {
  // Pre-flight: bail with a clean error if the browser can't render at all.
  // Avoids the noisy THREE.WebGLRenderer console stack trace and gives the
  // operator actionable Chrome-specific hints instead of a generic catch.
  const block = _detectWebGLBlocker();
  if (block) {
    _renderGraphError(block.reason, null, block.hintHTML);
    // Still fetch the graph payload so search keeps working as a node
    // browser even when the 3D canvas can't mount. Match-count updates
    // via _updateMatchCount → _searchableNodes() falling back to
    // graph3dData. Without this, search would silently no-op every time
    // WebGL was unavailable.
    try {
      if (!graph3dData) graph3dData = await api.graph3d();
      _updateMatchCount();
    } catch (_) { /* search degradation only; not worth surfacing */ }
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
      // Hook into WebGL context loss so a GPU process crash surfaces
      // as a recovery prompt instead of a silent black canvas. Only
      // wire on first mount — subsequent loadGraph3D() reuse the
      // existing renderer + canvas, so the listeners persist.
      _wireContextLossHandlers();
    }

    // Apply any current state (search + highlight) once nodes are live.
    _refreshGraph();
    _updateMatchCount();
  } catch(e) {
    console.error('Graph3D load failed:', e);
    // Full teardown — previously this just nulled the handle, leaving
    // the half-mounted canvas/renderer attached to the DOM. After a few
    // retries that orphaned chain would exhaust Chrome's per-tab WebGL
    // context budget and every subsequent mount would silently fail.
    _destroyGraph();
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
  bus.subscribe('recall:event', _onRecallEvent);

  // Keep the canvas matched to its container. ForceGraph3D does NOT observe
  // its host element, so without this the canvas keeps whatever size it had
  // at mount — widening the window left a narrow canvas in a wide pane.
  // A ResizeObserver catches every container size change from one place
  // (window resize, divider drag, layout-mode switch); resize() no-ops when
  // the graph isn't mounted. Deferred via rAF so the observed box has
  // settled before we read it and to avoid ResizeObserver feedback loops.
  // Replaces the old `live:layout` bus handler, which only covered the
  // divider/layout-mode subset and missed plain window resizes.
  const host = document.getElementById('graph-3d');
  if (host && 'ResizeObserver' in window) {
    new ResizeObserver(() => requestAnimationFrame(resize)).observe(host);
  }
}

export function activate() {
  // 300ms delay matches the legacy behavior: tab-content display:block
  // hasn't laid out yet by the time switchTab returns.
  setTimeout(() => {
    if (!graph3dData) loadGraph3D();
    else resize();
    // Pre-load the latest recall so the spotlight is immediately
    // visible on first open. Skips when the user has already pinned a
    // card (don't overwrite their selection).
    if (_highlightTier.size === 0 && _highlightMode !== 'pinned') {
      setHighlightMode('latest');
    }
  }, 300);
}

export function deactivate() {
  // 3D scene keeps animating in the background — cheap; re-activating
  // a paused scene introduces a visible re-warmup.
}

// Public teardown — frees the WebGL context, VRAM, and the render loop
// entirely (not paused — gone). Used by Live's graph-visibility toggle so
// a hidden graph costs zero GPU. Re-showing routes back through activate()
// → loadGraph3D(), which refetches since graph3dData is nulled here.
export function destroy() {
  _destroyGraph();
  graph3dData = null;
}

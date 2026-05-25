// ===========================================================================
// tabs/graph.js — 3D ForceGraph + search-driven node highlighter.
// ---------------------------------------------------------------------------
// Lifecycle contract (shared by every tabs/*.js module):
//
//   init()         called once on app boot, before any tab is shown.
//                  Wires polls + bus subs. Does NOT fetch data.
//
//   activate()     called when this tab becomes the visible one.
//                  Lazy-loads data + sizes graph to container.
//
//   deactivate()   called when leaving this tab. Most modules no-op
//                  (poll.js auto-gates on activeWhen + document.hidden).
//
// Since the P2.2 layout pivot the graph mounts inside Live's left pane;
// `activate()` is now driven by live.activate(). The standalone Graph
// tab is gone, and so is the community legend pane (P2.4) — the search
// input above the graph replaces both the discovery affordance (find a
// community by name) and the navigation affordance (zoom to its hub).
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

let graph3d = null;
let graph3dData = null;

// Search state. Lowercased query; nodes are matched against their name,
// type, and community_title (which is attached to every member of a
// community by the server). Empty query → everything matches.
let _searchQuery = '';

function _nodeMatches(node) {
  if (!_searchQuery) return true;
  const q = _searchQuery;
  return (node.name || '').toLowerCase().includes(q)
      || (node.type || '').toLowerCase().includes(q)
      || (node.community_title || '').toLowerCase().includes(q);
}

// Dim color for non-matched nodes. Dark enough to recede, light enough
// to still hint at structure. Matched nodes keep their assigned color.
const DIM_COLOR = '#1a1a1a';

// ── Recall pulse ──────────────────────────────────────────────────────
// Each recall surfaces three layers of nodes, in increasing causal
// importance. We pulse all three with distinct color + intensity so the
// operator can SEE the funnel narrow:
//
//   returned    ~25 nodes the recall pipeline surfaced (cosine + FTS5
//               + spread). Dim cool blue, weakest pulse — "considered".
//   activation  ≤30 nodes that lit up via spread-activation from the
//               judge's picks. These actually shape additionalContext.
//               Medium green pulse — "influenced this turn".
//   used        3–5 nodes the Haiku judge selected. Bright white pulse,
//               largest size bump — "Anchor literally read these".
//
// A node in multiple layers gets the strongest tier (used > activation
// > returned). _onRecallEvent enforces that ordering when populating
// the pulse map.

const PULSE_DURATION_MS = 5000;
const PULSE_SIZE_MULT   = 1.8;   // peak val multiplier at intensity=1.0

// tier name → { color, intensity }
// `intensity` scales both color-blend and size pulse. The tier ordering
// is encoded in PULSE_TIER_ORDER below — anything outside that list is
// ignored by _onRecallEvent.
const PULSE_TIERS = {
  used:       { color: '#ffffff', intensity: 1.00 },
  activation: { color: '#33ff88', intensity: 0.70 },
  returned:   { color: '#7eb8ff', intensity: 0.40 },
};
// Apply weakest → strongest so 'used' wins when a node appears in
// multiple layers (Map.set last-write semantics).
const PULSE_TIER_ORDER = ['returned', 'activation', 'used'];

// nodeId → { startedAt, tier }
const _pulses = new Map();
let _pulseRafScheduled = false;

function _pulseStrength(nodeId) {
  const p = _pulses.get(nodeId);
  if (!p) return 0;
  const elapsed = Date.now() - p.startedAt;
  if (elapsed >= PULSE_DURATION_MS) return 0;
  return 1 - (elapsed / PULSE_DURATION_MS);
}

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

function _colorFor(n) {
  const baseColor = _nodeMatches(n) ? (n.color || '#666') : DIM_COLOR;
  const p = _pulses.get(n.id);
  if (!p) return baseColor;
  const tier = PULSE_TIERS[p.tier];
  if (!tier) return baseColor;
  const strength = _pulseStrength(n.id);
  if (strength <= 0) return baseColor;
  return _blend(tier.color, baseColor, strength * tier.intensity);
}

function _valFor(n) {
  const base = n.hub ? n.val : 2;
  const p = _pulses.get(n.id);
  if (!p) return base;
  const tier = PULSE_TIERS[p.tier];
  if (!tier) return base;
  const strength = _pulseStrength(n.id);
  if (strength <= 0) return base;
  return base * (1 + (PULSE_SIZE_MULT - 1) * strength * tier.intensity);
}

function _scheduleAnimation() {
  if (_pulseRafScheduled || !graph3d) return;
  _pulseRafScheduled = true;
  const tick = () => {
    _pulseRafScheduled = false;
    if (!graph3d) return;
    const now = Date.now();
    let alive = 0;
    for (const [id, pulse] of _pulses) {
      if (now - pulse.startedAt >= PULSE_DURATION_MS) _pulses.delete(id);
      else alive++;
    }
    // Pass a fresh closure each tick. 3d-force-graph caches per-node
    // Three.js materials by color string; calling .nodeColor with the
    // SAME function reference is treated as a no-op by the library's
    // internal change detection, so the materials never refresh during
    // the pulse window. Wrapping in a new arrow each tick forces the
    // re-evaluation we need. Cheap — one closure per frame.
    graph3d.nodeColor(n => _colorFor(n));
    graph3d.nodeVal(n => _valFor(n));
    if (alive > 0) {
      _pulseRafScheduled = true;
      requestAnimationFrame(tick);
    }
  };
  requestAnimationFrame(tick);
}

function _onRecallEvent({ event }) {
  // Skip if the graph isn't loaded yet — buffered pulses for events that
  // arrived before the user opened Live would be confusing (out-of-time
  // flashes). The user sees activations going forward.
  if (!graph3d || !event) return;
  const now = Date.now();
  // Apply tiers weakest → strongest so a node in multiple layers ends up
  // tagged with its strongest tier (Map.set last-write).
  const idsByTier = {
    returned:   event.returned_ids   || [],
    activation: event.activation_ids || [],
    used:       event.used_ids       || [],
  };
  for (const tier of PULSE_TIER_ORDER) {
    for (const id of idsByTier[tier]) _pulses.set(id, { startedAt: now, tier });
  }
  _scheduleAnimation();
}

/** Apply current search to the live graph. Re-binding the nodeColor
 * function reference is what triggers ForceGraph3D to re-evaluate
 * material colors — passing the same fn ref is a no-op. */
function _refreshColors() {
  if (!graph3d) return;
  graph3d.nodeColor(_colorFor);
}

/** Pan the camera to the best match. Hub nodes win over members of the
 * same name, since the user typing "frame" usually means "show me the
 * Frame *community*", not one of its 20 leaf nodes. */
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

// Public — called by the inline `oninput` / `onkeydown` handlers via
// window.onGraphSearch / window.onGraphSearchKey.
export function setSearchQuery(q) {
  _searchQuery = (q || '').toLowerCase().trim();
  _refreshColors();
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

// ── Graph load + lifecycle ────────────────────────────────────────────

export async function loadGraph3D() {
  try {
    graph3dData = await api.graph3d();
    if (!graph3dData.nodes || !graph3dData.nodes.length) return;

    // Filter: only show nodes IN communities + community hub nodes.
    // Orphans get hidden — they clutter without adding structure.
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
    // Height comes from the parent .graph-container (100% of Live's
    // graph pane). Don't hard-set it.
    const w = container.offsetWidth || 800;
    const h = container.offsetHeight || 600;

    if (graph3d) {
      graph3d.graphData({nodes: visibleNodes, links: visibleLinks});
    } else {
      graph3d = ForceGraph3D()(container)
        .width(w).height(h)
        .graphData({nodes: visibleNodes, links: visibleLinks})
        .backgroundColor('#08080f')
        .nodeVal(_valFor)
        .nodeColor(_colorFor)
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

    // If the user had typed before the graph mounted, apply the filter
    // now that we have nodes.
    _refreshColors();
    _updateMatchCount();
  } catch(e) {
    console.error('Graph3D load failed:', e);
  }
}

/** Resize ForceGraph3D to the current container. Run after tab-switch or
 * layout drag — three.js doesn't observe its host. */
export function resize() {
  if (!graph3d) return;
  const c = document.getElementById('graph-3d');
  if (!c) return;
  // Trigger reflow before reading offset* so a hidden→visible transition
  // gives us the post-display size, not the pre-display zeros.
  void c.offsetHeight;
  const w = c.offsetWidth || 800;
  const h = c.offsetHeight || 600;
  graph3d.width(w).height(h);
  graph3d.renderer().setSize(w, h);
  graph3d.camera().aspect = w / h;
  graph3d.camera().updateProjectionMatrix();
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // No polls — the graph reloads on Refresh button or activate(). The 3D
  // scene's own animation loop keeps it moving once mounted.
  // Subscribe to layout drags from Live's divider — when the user resizes
  // the left pane, the renderer needs a setSize() pass.
  bus.subscribe('live:layout', () => {
    // Debounce-ish via rAF: mousemove fires every pixel; we don't want
    // to call renderer.setSize() that often. The browser coalesces.
    requestAnimationFrame(resize);
  });
  // Recall activations — pulse surfaced nodes for 5s when a new recall
  // lands. The bus event is published by live.js's pollRecallLog.
  bus.subscribe('recall:event', _onRecallEvent);
}

export function activate() {
  // 300ms delay matches the legacy behavior: the tab-content display:block
  // hasn't laid out yet by the time switchTab returns; size reads zero
  // without a beat.
  setTimeout(() => {
    if (!graph3dData) loadGraph3D();
    else resize();
  }, 300);
}

export function deactivate() {
  // 3D scene keeps animating in the background — cheap, and re-activating
  // a paused scene introduces a visible re-warmup.
}

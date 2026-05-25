// ===========================================================================
// tabs/graph.js — 3D ForceGraph + community legend.
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
// All cross-tab signals go through lib/bus.js, never direct imports.
// This module owns the 3D graph state — a single ForceGraph3D instance.
// In commit 2 the same instance gets mounted inside Live's left pane;
// the container ID lookup becomes parameterized at that point.
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

let graph3d = null;
let graph3dData = null;
let legendVisible = false;

export async function loadGraph3D() {
  try {
    graph3dData = await api.graph3d();
    if (!graph3dData.nodes || !graph3dData.nodes.length) return;

    // Filter: only show nodes IN communities + community hub nodes
    // Orphans get hidden — they clutter without adding structure
    const communityNodeIds = new Set();
    const hubIds = new Set();
    graph3dData.nodes.forEach(n => {
      if (n.hub) hubIds.add(n.id);
      if (n.community) communityNodeIds.add(n.id);
    });
    hubIds.forEach(id => communityNodeIds.add(id));

    const visibleNodes = graph3dData.nodes.filter(n => communityNodeIds.has(n.id) || hubIds.has(n.id));
    const visibleIds = new Set(visibleNodes.map(n => n.id));

    const visibleLinks = graph3dData.edges
      .filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
      .filter(e => e.relation !== 'co_accessed' && e.relation !== 'emergent_bridge')
      .map(e => ({source: e.source, target: e.target, relation: e.relation}));

    const container = document.getElementById('graph-3d');
    // Height comes from the parent .graph-container (100% of Live's
    // graph pane). Don't hard-set it — the layout has changed since
    // standalone-tab days.
    const w = container.offsetWidth || 800;
    const h = container.offsetHeight || 600;

    if (graph3d) {
      graph3d.graphData({nodes: visibleNodes, links: visibleLinks});
    } else {
      graph3d = ForceGraph3D()(container)
        .width(w).height(h)
        .graphData({nodes: visibleNodes, links: visibleLinks})
        .backgroundColor('#08080f')
        .nodeVal(n => n.hub ? n.val : 2)
        .nodeColor(n => n.color)
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

    const legendEl = document.getElementById('legend-items');
    if (graph3dData.communities && graph3dData.communities.length) {
      legendEl.innerHTML = graph3dData.communities.map(c =>
        '<div style="display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;cursor:pointer;transition:background 0.15s" ' +
        'onclick="focusCommunity(&quot;' + (c.hub_id || '') + '&quot;)" ' +
        'onmouseover="this.style.background=`rgba(255,255,255,0.08)`" onmouseout="this.style.background=`none`">' +
        '<div style="width:10px;height:10px;border-radius:50%;flex-shrink:0;background:' + c.color + ';box-shadow:0 0 4px ' + c.color + '"></div>' +
        '<span style="color:#aaa">' + c.name + ' (' + c.count + ')</span></div>'
      ).join('');
    } else {
      legendEl.innerHTML = '<div style="color:#555;padding:8px">No communities yet</div>';
    }
  } catch(e) {
    console.error('Graph3D load failed:', e);
  }
}

export function toggleLegend() {
  const el = document.getElementById('graph-legend');
  legendVisible = !legendVisible;
  el.style.transform = legendVisible ? 'translateX(0)' : 'translateX(220px)';
}

export function focusCommunity(hubId) {
  if (!graph3d || !hubId) return;
  const node = graph3d.graphData().nodes.find(n => n.id === hubId);
  if (node) {
    graph3d.cameraPosition({x: node.x + 120, y: node.y + 60, z: node.z + 120}, node, 1200);
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
    // Debounce-ish via rAF: the mousemove fires every pixel; we don't want
    // to call renderer.setSize() that often. The browser coalesces into
    // the next frame.
    requestAnimationFrame(resize);
  });
}

export function activate() {
  // 300ms delay matches the legacy behavior: the tab-content display:block
  // hasn't laid out yet by the time switchTab returns; size reads zero
  // without a beat. Once the layout-pivot lands and Live owns the graph,
  // this delay can probably drop.
  setTimeout(() => {
    if (!graph3dData) {
      loadGraph3D();
    } else {
      resize();
    }
  }, 300);
}

export function deactivate() {
  // 3D scene keeps animating in the background — cheap, and re-activating
  // a paused scene introduces a visible re-warmup.
}

// ===========================================================================
// lib/types.js — node type metadata. SINGLE SOURCE for type → color in JS.
// ---------------------------------------------------------------------------
// CSS reads from static/css/types.css (.type--<name>) for badges.
// JS reads from this module for code-time access (3D graph colorizing,
// dynamic legend, etc).
//
// The two are kept in sync MANUALLY for now — same names, same colors.
// Future improvement: serve a /api/type-colors endpoint that both CSS and
// JS consume so the brain-side aspect taxonomy is the only source of truth.
//
// Until then: ANY edit here must mirror static/css/types.css. The audit
// found three independent type→color maps (style.css, app.js TYPE_COLORS,
// queries/graph.py _TYPE_FALLBACK) that had drifted into three different
// subsets; killing that drift is the point of this file.
// ===========================================================================

// Hex literals so the 3D graph (which doesn't read CSS) gets a usable value.
// Matches static/css/types.css.
export const TYPE_COLORS = Object.freeze({
  // Lesson / insight family
  lesson:       '#4a9eff',
  insight:      '#88ddff',
  finding:      '#88ddff',
  decision:     '#aa66ff',
  mental_model: '#33dddd',
  mechanism:    '#dddd33',
  pattern:      '#ff66aa',
  architecture: '#33dddd',
  concept:      '#7eb8ff',
  design:       '#aadd33',
  fact:         '#ccc',

  // Identity-bearing
  principle:    '#ffaa33',
  rule:         '#ffaa33',
  identity:     '#ffaa33',
  vision:       '#ffcc44',
  directive:    '#ffaa33',

  // Episodic
  moment:       '#aa66ff',
  event:        '#aa66ff',
  interaction:  '#33ff88',
  context:      '#888',
  episode:      '#aa66ff',

  // Correction / divergence
  correction:   '#ff6666',
  bug:          '#ff8866',
  bug_lesson:   '#ff8866',
  tension:      '#ff4444',

  // Open / unresolved
  open:         '#aaaaff',
  uncertainty:  '#aaaaff',
  question:     '#aaaaff',
  constraint:   '#ff8833',
  impact:       '#ff6644',
  convention:   '#66aaff',

  // Communities + housekeeping
  community:    '#ffffff',
  vocabulary:   '#888',
  reflection:   '#ff99cc',
});

/** Color lookup with a fallback for unknown types. */
export function typeColor(name) {
  return TYPE_COLORS[name] || '#555';
}

/** CSS class fragment for a badge. Pair with the `.type-badge` base class. */
export function typeClass(name) {
  return name ? 'type--' + name : '';
}

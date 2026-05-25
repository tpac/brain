// ===========================================================================
// lib/scales.js — single source of truth for scale color codes.
// ---------------------------------------------------------------------------
// The fractal trace system tags every event with a scale (s0/s1/s2/s3/s4).
// Renderers across the dashboard color-code those scales identically:
//
//   traces.js     — per-chain accent stripe
//   node_detail.js — per-trace-ref color in the "Encoded from N traces"
//                    section
//   live.js       — S2 entries inherit s2 color
//
// Before this module, each renderer hard-coded its own {s0:'#888', ...}
// map, so adding a new scale meant chasing every consumer. Now: one map,
// one source.
//
// These hex codes mirror the design tokens in vars.css (--color-s0,
// --color-s1, ...). The duplication is intentional: callers who want
// a CSS class use `.text--s1` / `.card--s1`; callers who need an inline
// color for a `style="border-left-color:..."` attribute read from here.
// As inline-style usage drops, this map shrinks alongside.
// ===========================================================================

export const SCALE_COLORS = {
  s0: '#888',
  s1: '#7eb8ff',
  s2: '#ffaa33',
  s3: '#33ff88',
  s4: '#ff66aa',
};

/** Resolve a scale string to its accent color. Falls back to a neutral
 * gray so unknown scales render visibly rather than disappearing. */
export function scaleColor(scale) {
  return SCALE_COLORS[scale] || '#666';
}

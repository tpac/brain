// ===========================================================================
// lib/dom.js — DOM construction helpers (escaped by default).
// ---------------------------------------------------------------------------
// Replaces:
//   - 39 sites doing `el.innerHTML = '<div>...' + escapeHtml(x) + '...';`
//   - 23 manual escapeHtml() calls (one per interpolation; missing one was
//     an XSS-shaped bug waiting to happen)
//   - 162 inline `style="..."` attributes that should be CSS classes
//
// The core helper is `el(tag, attrs, ...children)`. Strings in `children`
// are auto-escaped. To insert raw HTML, use the `html()` marker.
//
//   el('div', { class: 'card card--s1' },
//     el('span', { class: 'badge badge--blue' }, 'HOOK'),
//     ' ',
//     evt.title,           // auto-escaped
//   )
//
// Returns an HTMLElement. Mount with parent.replaceChildren(el) or
// parent.appendChild(el) — no innerHTML assignment, no string concatenation.
//
// Why not just `template` elements + cloneNode? That's a fine pattern for
// fixed shapes, but the dashboard renders open-ended dynamic content (the
// activity feed varies per row type). `el()` is more compositional. We use
// <template> only for the largest fixed shapes (node-detail panel).
// ===========================================================================

const _HTML_MARKER = Symbol('html');

/** Mark a string as raw HTML so el() inserts it unescaped. Use sparingly —
 * always prefer composing with el() so escaping is automatic. */
export function html(str) {
  return { [_HTML_MARKER]: true, value: String(str) };
}

export function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Create an element. Strings in children are escaped; pass `html(...)` for
 * raw. Attrs:
 *   { class: 'foo bar' }                    → class attribute
 *   { onclick: () => ... }                  → addEventListener('click', ...)
 *   { dataset: { foo: 'bar' } }             → data-foo="bar"
 *   { style: { color: 'red', padding:'4px'}} → inline style (avoid; prefer class)
 *   { 'aria-label': '...' }                 → arbitrary attribute */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') {
        node.className = Array.isArray(v) ? v.filter(Boolean).join(' ') : String(v);
      } else if (k === 'dataset') {
        for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
      } else if (k === 'style' && typeof v === 'object') {
        for (const [sk, sv] of Object.entries(v)) node.style[sk] = sv;
      } else if (k.startsWith('on') && typeof v === 'function') {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else {
        node.setAttribute(k, v);
      }
    }
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    if (child instanceof Node) {
      node.appendChild(child);
    } else if (child && child[_HTML_MARKER]) {
      // Raw HTML — must trust the caller. Use sparingly.
      const tmp = document.createElement('div');
      tmp.innerHTML = child.value;
      while (tmp.firstChild) node.appendChild(tmp.firstChild);
    } else {
      node.appendChild(document.createTextNode(String(child)));
    }
  }
  return node;
}

/** Convert a server ISO timestamp to a human-readable local time. */
export function localTime(utcStr, mode) {
  if (!utcStr) return '';
  let s = utcStr;
  if (s.length >= 19 && !s.endsWith('Z') && !s.includes('+')) s += 'Z';
  const d = new Date(s);
  if (isNaN(d)) return utcStr;
  if (mode === 'time') {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  return d.toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** Speaker→responder identity chip used on recall + trace cards. Returns an
 * el() or null when neither side has identity (most pre-stamp traces). */
export function identityChip(human, agent) {
  if (!human && !agent) return null;
  const h = human || '?';
  const a = agent || '?';
  return el('span', { class: 'chip chip--identity', title: 'speaker → responder' },
    h, el('span', { style: { color: '#555', margin: '0 3px' } }, '→'), a);
}

/** Convenience: clear a node and append a list of children atomically.
 * Avoids the half-rendered flash that `node.innerHTML = ''` + repeated
 * appends causes. */
export function mount(parent, ...children) {
  parent.replaceChildren(...children.flat().filter(c => c != null));
}

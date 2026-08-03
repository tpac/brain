// ===========================================================================
// lib/api.js — single fetch wrapper.
// ---------------------------------------------------------------------------
// Two job:
//   1. Centralize fetch + JSON parsing + error handling. No more 14 sites
//      copy-pasting `await fetch(...) + .json()` with three different shape
//      conventions.
//   2. Dedupe concurrent identical requests so two panels asking for /api/X
//      at the same tick coalesce into one network hit.
//
// Returns the RAW response body (whatever the endpoint shipped). Callers
// keep their existing `.events` / `.runs` / `.nodes` access patterns. The
// shape-normalizing wrapper (`unwrap`) is opt-in for callers that want a
// uniform `{ok, data, warnings}` view — Phase 1+ code uses it; legacy
// inline code doesn't have to migrate yet.
//
// Why no auto-unwrap: there are 14 endpoints, 6 different shape conventions
// (bare array, {events:[]}, {runs:[]}, {nodes:[], total:N}, {aspects:[]},
// {status:{...}}). Forcing every caller through a normalized shape would
// require rewriting all 14 panels in one swing — a one-day risk for a
// one-week problem. Adoption stays incremental.
// ===========================================================================

const _inflight = new Map();   // url → in-flight Promise

/** Low-level GET. Returns the parsed JSON body, or throws on network/HTTP/
 * parse failure. Concurrent identical URLs coalesce. */
export async function get(url) {
  if (_inflight.has(url)) return _inflight.get(url);
  const p = (async () => {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error('HTTP ' + r.status + ' on ' + url);
      return await r.json();
    } finally {
      _inflight.delete(url);
    }
  })();
  _inflight.set(url, p);
  return p;
}

/** Low-level POST with a JSON body. Returns the parsed JSON body; throws on
 * network/HTTP/parse failure. NOT coalesced (writes aren't idempotent). This
 * is the dashboard's only write path — it reaches the daemon via the server,
 * never the DB directly. */
export async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  let parsed = null;
  try { parsed = await r.json(); } catch (_) { /* leave null */ }
  if (!r.ok) {
    const msg = (parsed && parsed.error) ? parsed.error : ('HTTP ' + r.status);
    throw new Error(msg);
  }
  return parsed;
}

/** Convert any of the legacy shapes into a uniform {ok, data, warnings}.
 * Opt-in — call this when you want consistency, leave it off when you want
 * the raw shape. */
export function unwrap(body) {
  if (body && body.status === 'error') {
    return { ok: false, data: null, warnings: [], error: body.error || 'unknown' };
  }
  if (body && body.status === 'success') {
    return { ok: true, data: body.data, warnings: body.warnings || [] };
  }
  if (Array.isArray(body)) {
    return { ok: true, data: body, warnings: [] };
  }
  if (body && typeof body === 'object') {
    const skip = new Set(['latest_ts', 'latest_id', 'count', 'total', 'status']);
    const arrayKeys = Object.keys(body).filter(
      k => !skip.has(k) && Array.isArray(body[k])
    );
    if (arrayKeys.length === 1) {
      return {
        ok: true,
        data: body[arrayKeys[0]],
        warnings: body.warnings || [],
        meta: Object.fromEntries(
          Object.keys(body).filter(k => skip.has(k)).map(k => [k, body[k]])
        ),
      };
    }
  }
  return { ok: true, data: body, warnings: [] };
}

// Query-string builder — skip empty values so URLs stay clean.
function _qs(params) {
  if (!params) return '';
  const parts = Object.entries(params)
    .filter(([_, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v));
  return parts.length ? '?' + parts.join('&') : '';
}

/** Named endpoint shortcuts. Each returns the raw body (legacy-compatible).
 * Wrap with `unwrap()` if you want the normalized shape. */
export const api = {
  stats:               ()       => get('/api/stats'),
  status:              ()       => get('/api/status'),
  systemStatus:        ()       => get('/api/system-status'),
  insights:            ()       => get('/api/insights'),
  insightsLive:        ()       => get('/api/insights/live'),
  aspects:             ()       => get('/api/aspects'),
  sessions:            ()       => get('/api/sessions'),
  recalls:             (p = {}) => get('/api/recalls' + _qs(p)),
  recallPrompt:        (p = {}) => get('/api/recall-prompt' + _qs(p)),
  encodingRuns:        (p = {}) => get('/api/encoding-runs' + _qs(p)),
  encodingPrompt:      (p = {}) => get('/api/encoding-prompt' + _qs(p)),
  encodingActivity:    (p = {}) => get('/api/encoding-activity' + _qs(p)),
  consolidationRuns:   (p = {}) => get('/api/consolidation-runs' + _qs(p)),
  communityRuns:       (p = {}) => get('/api/community-runs' + _qs(p)),
  healerRuns:          (p = {}) => get('/api/healer-runs' + _qs(p)),
  consolidationPrompt: (p = {}) => get('/api/consolidation-prompt' + _qs(p)),
  traces:              (p = {}) => get('/api/traces' + _qs(p)),
  errors:              (p = {}) => get('/api/errors' + _qs(p)),
  nodes:               (p = {}) => get('/api/nodes' + _qs(p)),
  graph3d:             ()       => get('/api/graph3d'),
  node:                (id)     => get('/api/node/' + encodeURIComponent(id)),
  nodeCorrections:     (id)     => get('/api/node/' + encodeURIComponent(id) + '/corrections'),
  nodeSourceRefs:      (id)     => get('/api/node/' + encodeURIComponent(id) + '/source-refs'),
  // Self-channel (Streams tab)
  selfMessages:        (p = {}) => get('/api/self-messages' + _qs(p)),
  bootRenders:         (p = {}) => get('/api/boot-renders' + _qs(p)),
  selfPresence:        (p = {}) => get('/api/self-presence' + _qs(p)),
  selfSend:            (body)   => post('/api/self-send', body),
};

export default api;

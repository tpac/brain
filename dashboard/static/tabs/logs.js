// ===========================================================================
// tabs/logs.js — one unified log feed: brain + hook + dashboard + client.
// ---------------------------------------------------------------------------
// Was three sub-feeds (Errors / Daemon / Dashboard) behind a toggle. Now a
// single stream with a SOURCE PICKER, identical error types GROUPED together
// (×N), and click-to-expand for the full story — message, context, and the
// traceback/stack that the old flat rows never showed.
//
// Still owns the browser-side error capture (window.onerror, console.error,
// unhandledrejection) — those are the "client" source — and the tab's unread
// badge (#logs-badge).
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime, relativeTime } from '/static/lib/dom.js';

const LEVEL_COLORS = { critical: '#ff4444', error: '#ff6644', warning: '#ffaa33', info: '#4a9eff' };
const _levelColor = (l) => LEVEL_COLORS[l] || '#888';

// Per-source identity for the filter + the source chip.
const SOURCE_META = {
  brain:     { label: 'brain',     color: '#7eb8ff' },
  hook:      { label: 'hook',      color: '#33d17a' },
  dashboard: { label: 'dashboard', color: '#ffaa33' },
  client:    { label: 'client',    color: '#ff6644' },
};

let _groups = [];                  // last computed groups (for expand re-render)
const _expanded = new Set();       // group keys currently expanded
let _lastLogFp = null;             // render signature — guards the 5s poll rebuild

// ── Client-side error capture (the "client" source) ────────────────────
const _clientErrors = [];          // ring of {ts, source, message, stack}
const MAX_CLIENT_ERRORS = 200;

function _captureClientError(source, message, stack) {
  _clientErrors.push({
    ts: new Date().toISOString(), source,
    message: String(message).slice(0, 300),
    stack: stack ? String(stack).split('\n').slice(0, 8).join('\n') : '',
  });
  while (_clientErrors.length > MAX_CLIENT_ERRORS) _clientErrors.shift();
  if (!_logsTabActive()) { _unreadDash++; _setLogsBadge(); }
}

let _clientErrorsWired = false;
function _wireClientErrorCapture() {
  if (_clientErrorsWired) return;
  _clientErrorsWired = true;
  const _legacyOnError = window.onerror;
  window.onerror = function (msg, src, line, col, err) {
    _captureClientError('window.onerror', msg + ' @ line ' + line, err && err.stack);
    if (_legacyOnError) return _legacyOnError(msg, src, line, col, err);
  };
  const _origConsoleError = console.error.bind(console);
  console.error = function (...args) {
    try {
      const text = args.map(a => (a && a.message ? a.message : String(a))).join(' ');
      _captureClientError('console.error', text, args[args.length - 1] && args[args.length - 1].stack);
    } catch (e) { /* don't recurse */ }
    _origConsoleError(...args);
  };
  window.addEventListener('unhandledrejection', e => {
    _captureClientError('unhandledrejection', (e.reason && e.reason.message) || String(e.reason),
      e.reason && e.reason.stack);
  });
}

// ── Fetch + normalize every source into one uniform entry shape ────────
// { source, level, component, message, context, detail, timestamp }
// `detail` is the traceback / stack — the full story shown on expand.
async function _fetchEntries() {
  const hours = document.getElementById('error-hours').value;
  const out = [];

  // brain + hook (one aggregated endpoint; `source` distinguishes them)
  try {
    const d = await api.errors({ hours, limit: 300 });
    for (const e of (d.errors || [])) {
      out.push({
        source: e.source || 'brain', level: e.level || 'error',
        component: e.component || '', message: e.error || '',
        context: e.context || '', detail: e.traceback || '',
        timestamp: e.timestamp,
      });
    }
  } catch (e) { console.error('[logs] errors fetch failed', e); }

  // dashboard server ring (Python warn() calls in the dashboard itself)
  try {
    const d = await fetch('/api/dashboard-errors').then(r => r.json());
    for (const e of (d.errors || [])) {
      out.push({
        source: 'dashboard', level: 'warning',
        component: e.component || '?', message: e.message || '',
        context: e.exc_type ? e.exc_type : '', detail: e.exc_text || '',
        timestamp: e.ts,
      });
    }
  } catch (e) { /* dashboard ring unavailable — leave it out */ }

  // client (browser) errors — the in-memory ring
  for (const e of _clientErrors) {
    out.push({
      source: 'client', level: 'error', component: e.source || 'browser',
      message: e.message || '', context: '', detail: e.stack || '',
      timestamp: e.ts,
    });
  }
  return out;
}

// Fingerprint: collapse occurrences of the SAME error type. Mask the volatile
// bits (numbers, hex ids, quoted strings, paths) so "failed for node a1b2" and
// "failed for node c3d4" group as one type. Keyed within (source, component).
function _fingerprint(e) {
  let m = (e.message || '').toLowerCase();
  m = m.replace(/0x[0-9a-f]+/g, '#')
       .replace(/\b[0-9a-f]{8,}\b/g, '#')
       .replace(/\d[\d.,:_/-]*/g, '#')   // any number run, incl. unit suffixes (352s → #s)
       .replace(/'[^']*'/g, "'…'").replace(/"[^"]*"/g, '"…"')
       .replace(/\/[^\s,)]+/g, '/…')
       .replace(/\s+/g, ' ').trim();
  return (e.source || '') + '|' + (e.component || '') + '|' + m.slice(0, 100);
}

function _group(entries) {
  const map = new Map();
  for (const e of entries) {
    const key = _fingerprint(e);
    let g = map.get(key);
    if (!g) { g = { key, rep: e, count: 0, occ: [] }; map.set(key, g); }
    g.count++;
    g.occ.push(e);
    if ((e.timestamp || '') > (g.rep.timestamp || '')) g.rep = e;   // newest is representative
  }
  const groups = [...map.values()];
  groups.sort((a, b) => (b.rep.timestamp || '').localeCompare(a.rep.timestamp || ''));
  return groups;
}

// ── Render ──────────────────────────────────────────────────────────────
export async function loadLogs() {
  const feed = document.getElementById('feed-logs');
  if (!feed) return;
  const src = document.getElementById('log-source').value;
  const level = document.getElementById('log-level').value;
  const hours = document.getElementById('error-hours').value;
  let entries = await _fetchEntries();
  if (src && src !== 'all') entries = entries.filter(e => e.source === src);
  // 'error' folds in critical (both error-class); 'warning' is exact.
  if (level && level !== 'all') {
    entries = entries.filter(e =>
      level === 'error' ? (e.level === 'error' || e.level === 'critical')
                        : e.level === level);
  }

  _groups = _group(entries);

  // Skip the rebuild when nothing changed — the 5s poll would otherwise wipe
  // the page scroll and the scroll position inside an expanded <pre> traceback
  // the operator is reading. Expand toggles re-render a single card directly
  // (not via loadLogs), so skipping here never strands them.
  const fp = [src, level, hours, entries.length, _groups.length,
    _groups[0] && _groups[0].rep.timestamp].join('|');
  if (fp === _lastLogFp) return;
  _lastLogFp = fp;

  document.getElementById('logs-count').textContent =
    entries.length + ' event' + (entries.length === 1 ? '' : 's') +
    ' · ' + _groups.length + ' type' + (_groups.length === 1 ? '' : 's');

  if (!_groups.length) {
    const qualifier = (level !== 'all' ? escapeHtml(level) + ' ' : '')
      + (src !== 'all' ? escapeHtml(src) + ' ' : '');
    feed.innerHTML = '<div style="color:#5a8a5a;text-align:center;padding:40px;font-size:12px">'
      + 'No ' + qualifier + 'logs in the last ' + escapeHtml(hours) + 'h. '
      + 'The substrate is loud-by-default — a blank panel here means quiet, not broken.</div>';
    return;
  }
  feed.innerHTML = _groups.map(_renderGroup).join('');
}

function _renderGroup(g) {
  const e = g.rep;
  const lc = _levelColor(e.level);
  const sm = SOURCE_META[e.source] || { label: e.source || '?', color: '#888' };
  const open = _expanded.has(g.key);

  let h = '<div class="log-group" data-key="' + escapeHtml(g.key) + '" '
    + 'style="background:#0b0c12;border:1px solid #1a1b26;border-left:3px solid ' + lc + ';border-radius:8px;margin:6px 0;padding:8px 11px;cursor:pointer">';

  // header
  h += '<div style="display:flex;align-items:center;gap:7px">';
  h += '<span style="background:' + lc + '22;color:' + lc + ';font-size:9px;font-weight:bold;text-transform:uppercase;padding:1px 6px;border-radius:3px">' + escapeHtml(e.level || 'error') + '</span>';
  h += '<span style="color:' + sm.color + ';font-size:10px;font-family:ui-monospace,monospace">' + escapeHtml(sm.label) + '</span>';
  if (e.component) h += '<span style="color:#778;font-size:10px;font-family:ui-monospace,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(e.component) + '</span>';
  h += '<span style="flex:1"></span>';
  if (g.count > 1) h += '<span title="' + g.count + ' occurrences" style="background:#1a1b26;color:#cdd;font-size:10px;font-weight:bold;padding:1px 7px;border-radius:10px">×' + g.count + '</span>';
  h += '<span style="color:#556;font-size:10px;white-space:nowrap" title="' + escapeHtml(e.timestamp || '') + '">' + escapeHtml(relativeTime(e.timestamp)) + '</span>';
  h += '<span style="color:#566;font-size:10px;width:10px;text-align:center">' + (open ? '▾' : '▸') + '</span>';
  h += '</div>';

  // message — clamped to 2 lines until expanded
  const clamp = open ? '' : 'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;';
  h += '<div style="color:#cdd;font-size:12px;margin-top:5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;' + clamp + '">' + escapeHtml(e.message || '(no message)') + '</div>';

  if (open) h += _renderDetail(g);
  h += '</div>';
  return h;
}

function _renderDetail(g) {
  // Show context/traceback from the RICHEST occurrence, not just the newest:
  // a grouped type whose latest firing is a bare marker shouldn't hide a full
  // traceback an earlier occurrence recorded.
  const _weight = (o) => (o.detail || '').length + (o.context || '').length;
  const e = g.occ.reduce((best, o) => (_weight(o) > _weight(best) ? o : best), g.rep);
  let h = '<div style="margin-top:8px;border-top:1px solid #15161f;padding-top:8px">';
  if (e.context) {
    h += '<div style="color:#667;font-size:9px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px">context</div>'
      + '<div style="color:#9ab;font-size:11px;white-space:pre-wrap;word-break:break-word;margin-bottom:6px">' + escapeHtml(e.context) + '</div>';
  }
  if (e.detail) {
    h += '<div style="color:#667;font-size:9px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px">traceback</div>'
      + '<pre style="margin:0 0 6px;padding:6px 8px;background:#0c0c16;border:1px solid #181826;border-radius:4px;color:#bcd;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto;font-family:ui-monospace,Menlo,monospace;font-size:11px">' + escapeHtml(e.detail) + '</pre>';
  }
  // occurrences — when grouped, list each time it fired (newest first)
  if (g.count > 1) {
    const times = g.occ.slice().sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || '')).slice(0, 25);
    h += '<div style="color:#667;font-size:9px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px">'
      + g.count + ' occurrences</div><div style="display:flex;flex-wrap:wrap;gap:4px">';
    h += times.map(o => '<span style="background:#101119;border:1px solid #1c1d28;border-radius:3px;color:#9ab;font-size:10px;padding:1px 6px" title="' + escapeHtml(o.timestamp || '') + '">' + escapeHtml(localTime(o.timestamp, 'time')) + '</span>').join('');
    if (g.count > 25) h += '<span style="color:#566;font-size:10px;align-self:center">+ ' + (g.count - 25) + ' more</span>';
    h += '</div>';
  } else if (!e.context && !e.detail) {
    h += '<div style="color:#566;font-size:11px">No further detail recorded.</div>';
  }
  h += '</div>';
  return h;
}

function _onFeedClick(e) {
  const card = e.target.closest('.log-group');
  if (!card) return;
  const key = card.getAttribute('data-key');
  if (!key) return;
  if (_expanded.has(key)) _expanded.delete(key);
  else _expanded.add(key);
  // re-render just this group in place
  const idx = _groups.findIndex(g => g.key === key);
  if (idx >= 0) card.outerHTML = _renderGroup(_groups[idx]);
}

// ── Unread badge (#logs-badge on the tab) ──────────────────────────────
let _seenErr = -1, _seenDash = -1, _unreadErr = 0, _unreadDash = 0;
function _logsTabActive() {
  const t = document.getElementById('tab-logs');
  return t && t.classList.contains('active');
}
function _setLogsBadge() {
  const b = document.getElementById('logs-badge');
  if (!b) return;
  const total = _unreadErr + _unreadDash;
  if (total > 0) { b.textContent = String(total); b.style.display = ''; }
  else b.style.display = 'none';
}

// ── Lifecycle ─────────────────────────────────────────────────────────
export function init() {
  _wireClientErrorCapture();

  const feed = document.getElementById('feed-logs');
  if (feed && !feed._logsClickBound) {
    feed.addEventListener('click', _onFeedClick);
    feed._logsClickBound = true;
  }

  // Refresh the feed while the tab is open.
  poll.register({
    key: 'logs-feed', interval: 5000,
    activeWhen: _logsTabActive,
    fetcher: loadLogs,
  });

  // Unread badge — counts new brain/hook + dashboard/client errors while the
  // tab ISN'T open; resets the baseline when it is. Always-on (no activeWhen);
  // poll.js still pauses on a hidden window.
  poll.register({
    key: 'logs-badge', interval: 10000,
    fetcher: async () => {
      try {
        const d = await api.errors({ hours: 24, limit: 200 });
        const errCount = d.count || 0;
        if (_seenErr < 0) _seenErr = errCount;
        let dashTotal = _clientErrors.length;
        try { dashTotal += (await fetch('/api/dashboard-errors?limit=1').then(r => r.json())).count || 0; } catch (_) {}
        if (_seenDash < 0) _seenDash = dashTotal;
        if (_logsTabActive()) {
          _seenErr = errCount; _seenDash = dashTotal; _unreadErr = 0; _unreadDash = 0;
        } else {
          _unreadErr = Math.max(0, errCount - _seenErr);
          _unreadDash = Math.max(0, dashTotal - _seenDash);
        }
        _setLogsBadge();
      } catch (e) { /* badge is best-effort */ }
    },
  });
}

export function activate() {
  _seenErr = -1; _seenDash = -1; _unreadErr = 0; _unreadDash = 0;
  _setLogsBadge();
  loadLogs();
}

export function deactivate() {}

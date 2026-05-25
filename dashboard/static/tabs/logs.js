// ===========================================================================
// tabs/logs.js — Errors / Daemon / Dashboard sub-feeds + client-error capture.
// ---------------------------------------------------------------------------
// Owns:
//   - Three sub-feeds (errors, daemon, dashboard) under the Logs tab.
//   - The badge that flashes when error count grows for the inactive feed.
//   - The browser-side error wraps (window.onerror, console.error,
//     unhandledrejection) that populate the Dashboard sub-feed.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime } from '/static/lib/dom.js';

let activeLogFeed = 'errors';

export function switchLogFeed(name) {
  activeLogFeed = name;
  // Find the button whose text matches `name` (case-insensitive) and mark
  // it active. Previously `event.target.classList.add('active')` — global
  // `event` only exists inside inline onclick handlers, so calling
  // switchLogFeed() programmatically threw.
  document.querySelectorAll('#tab-logs .feed-btn').forEach(b => {
    b.classList.toggle('active', b.textContent.trim().toLowerCase().startsWith(name));
  });
  ['errors','daemon','dashboard'].forEach(f => {
    document.getElementById('feed-' + f).style.display = f === name ? '' : 'none';
  });
  if (name === 'errors') document.getElementById('err-badge').style.display = 'none';
  if (name === 'dashboard') document.getElementById('dash-err-badge').style.display = 'none';
  loadLogs();
}

export async function loadLogs() {
  if (activeLogFeed === 'errors') loadErrors();
  else if (activeLogFeed === 'daemon') loadDaemonLogs();
  else if (activeLogFeed === 'dashboard') loadDashboardErrors();
}

// ── Dashboard self-monitoring ──────────────────────────────────────────
// The dashboard captures TWO error sources:
//   1. Python warn() calls — surfaced via /api/dashboard-errors (the ring
//      buffer in dashboard/log.py).
//   2. Browser-side errors — window.onerror + console.error wrap below.

const _clientErrors = [];  // ring of {ts, source, message, stack}
const MAX_CLIENT_ERRORS = 200;

function _captureClientError(source, message, stack) {
  _clientErrors.push({
    ts: new Date().toISOString(),
    source,
    message: String(message).slice(0, 300),
    stack: stack ? String(stack).split('\n').slice(0, 4).join(' | ') : '',
  });
  while (_clientErrors.length > MAX_CLIENT_ERRORS) _clientErrors.shift();
  // Flash the badge on the Dashboard sub-feed unless it's currently visible.
  if (activeLogFeed !== 'dashboard') {
    const b = document.getElementById('dash-err-badge');
    if (b) { b.textContent = _clientErrors.length; b.style.display = ''; }
  }
}

function _wireClientErrorCapture() {
  // Existing window.onerror handler still updates document.title for the
  // legacy at-a-glance indicator; we wrap it so capture still happens.
  const _legacyOnError = window.onerror;
  window.onerror = function(msg, src, line, col, err) {
    _captureClientError('window.onerror', msg + ' @ line ' + line, err && err.stack);
    if (_legacyOnError) return _legacyOnError(msg, src, line, col, err);
  };
  // console.error wrap — every error log lands in the ring too.
  const _origConsoleError = console.error.bind(console);
  console.error = function(...args) {
    try {
      const text = args.map(a => a && a.message ? a.message : String(a)).join(' ');
      _captureClientError('console.error', text, args[args.length-1] && args[args.length-1].stack);
    } catch (e) { /* don't recurse */ }
    _origConsoleError(...args);
  };
  // Promise rejections that nothing else caught.
  window.addEventListener('unhandledrejection', e => {
    _captureClientError('unhandledrejection', e.reason && e.reason.message || String(e.reason),
                        e.reason && e.reason.stack);
  });
}

async function loadDashboardErrors() {
  const feed = document.getElementById('feed-dashboard');
  let serverEntries = [];
  try {
    const d = await fetch('/api/dashboard-errors').then(r => r.json());
    serverEntries = (d.errors || []).map(e => ({
      ts: e.ts,
      source: 'server:' + (e.component || '?'),
      message: e.message + (e.exc_text ? ' — ' + e.exc_type + ': ' + e.exc_text : ''),
      stack: '',
    }));
  } catch (e) {
    _captureClientError('loadDashboardErrors', 'server fetch failed: ' + e);
  }
  const all = [...serverEntries, ..._clientErrors.slice().reverse()];
  all.sort((a, b) => (b.ts || '').localeCompare(a.ts || ''));
  document.getElementById('logs-count').textContent = all.length + ' dashboard events';

  if (!all.length) {
    feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">No dashboard errors. The substrate is loud-by-default; if a panel goes blank without entries here, that\'s a regression.</div>';
    return;
  }
  feed.innerHTML = '';
  for (const e of all) {
    const isServer = e.source && e.source.startsWith('server:');
    const color = isServer ? '#ffaa33' : '#ff6644';
    const div = document.createElement('div');
    div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + color + ';font-size:12px';
    const t = localTime(e.ts);
    div.innerHTML =
      '<span class="badge badge--ghost-amber" style="background:' + color + '22;color:' + color + '">' + (isServer ? 'PY' : 'JS') + '</span> ' +
      '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.source || '?') + '</span>' +
      '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.message || '') + '</div>' +
      (e.stack ? '<div style="color:#666;font-size:10px;margin-top:2px;font-family:monospace">' + escapeHtml(e.stack) + '</div>' : '') +
      '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
    feed.appendChild(div);
  }
}

async function loadErrors() {
  const hours = document.getElementById('error-hours').value;
  try {
    const d = await api.errors({ hours, limit: 100 });
    const feed = document.getElementById('feed-errors');
    document.getElementById('logs-count').textContent = d.count + ' errors';

    if (!d.errors || !d.errors.length) {
      feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">No errors in the last ' + hours + 'h</div>';
      return;
    }
    feed.innerHTML = '';
    for (const e of d.errors) {
      const div = document.createElement('div');
      div.dataset.source = e.source || '';
      const levelColor = {critical:'#ff4444',error:'#ff6644',warning:'#ffaa33',info:'#4a9eff'}[e.level] || '#888';
      div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + levelColor + ';font-size:12px';
      const t = localTime(e.timestamp);
      const sessionTag = e.session_id ? '<span style="color:#555;font-size:9px;margin-left:4px">' + e.session_id.substring(0,8) + '</span>' : '';
      div.innerHTML =
        '<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;text-transform:uppercase;background:' + levelColor + '22;color:' + levelColor + '">' + (e.level || 'error') + '</span> ' +
        '<span style="color:#888;font-size:10px">' + (e.source || '') + '</span> ' +
        '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.component || '') + '</span>' + sessionTag +
        '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.error || '') + '</div>' +
        (e.context ? '<div style="color:#666;font-size:10px;margin-top:2px">' + escapeHtml(e.context) + '</div>' : '') +
        '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
      feed.appendChild(div);
    }
  } catch(e) {
    document.getElementById('feed-errors').innerHTML = '<div style="color:#f66;padding:20px">Failed to load: ' + e + '</div>';
  }
}

async function loadDaemonLogs() {
  const hours = document.getElementById('error-hours').value;
  try {
    const d = await api.errors({ hours, limit: 200, source: 'daemon' });
    const feed = document.getElementById('feed-daemon');
    const d2 = await api.errors({ hours, limit: 50, source: 'hook' });

    const all = [...(d.errors || []), ...(d2.errors || [])];
    all.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
    document.getElementById('logs-count').textContent = all.length + ' daemon events';

    if (!all.length) {
      feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">No daemon events in the last ' + hours + 'h</div>';
      return;
    }
    feed.innerHTML = '';
    for (const e of all) {
      const div = document.createElement('div');
      const levelColor = {critical:'#ff4444',error:'#ff6644',warning:'#ffaa33',info:'#4a9eff'}[e.level] || '#888';
      const isRestart = (e.error || '').includes('restart') || (e.component || '').includes('restart');
      const borderColor = isRestart ? '#4a9eff' : levelColor;
      div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + borderColor + ';font-size:12px';
      const t = localTime(e.timestamp);
      div.innerHTML =
        '<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;text-transform:uppercase;background:' + borderColor + '22;color:' + borderColor + '">' + (isRestart ? 'restart' : e.level || 'error') + '</span> ' +
        '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.component || '') + '</span>' +
        '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.error || '') + '</div>' +
        '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
      feed.appendChild(div);
    }
  } catch(e) {
    document.getElementById('feed-daemon').innerHTML = '<div style="color:#f66;padding:20px">Failed to load: ' + e + '</div>';
  }
}

let lastSeenErrorCount = -1;
async function _checkErrBadge() {
  const logsTab = document.getElementById('tab-logs');
  const isViewing = logsTab && logsTab.classList.contains('active');
  try {
    const d = await api.errors({ hours: 1, limit: 1 });
    const errBadge = document.getElementById('err-badge');
    const logsBadge = document.getElementById('logs-badge');
    if (lastSeenErrorCount < 0) lastSeenErrorCount = d.count;
    if (isViewing && activeLogFeed === 'errors') {
      lastSeenErrorCount = d.count;
      errBadge.style.display = 'none';
      loadErrors();
    } else if (d.count > lastSeenErrorCount) {
      const diff = d.count - lastSeenErrorCount;
      errBadge.textContent = diff; errBadge.style.display = '';
      logsBadge.textContent = diff; logsBadge.style.display = '';
    } else {
      errBadge.style.display = 'none';
      logsBadge.style.display = 'none';
    }
  } catch(e) { console.error('[dashboard] err-badge check failed:', e); }
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  _wireClientErrorCapture();

  // Error badge — 10s background poll, always running (no activeWhen). The
  // badge counts errors for tabs that AREN'T visible; document.hidden still
  // gates poll.js so an unfocused window pauses.
  poll.register({ key: 'err-badge', interval: 10000, fetcher: _checkErrBadge });

  // Background poll for new dashboard errors — drives the badge so the
  // operator sees regression clusters without manually clicking the tab.
  poll.register({
    key: 'dash-err-badge',
    interval: 15000,
    fetcher: async () => {
      try {
        const d = await fetch('/api/dashboard-errors?limit=1').then(r => r.json());
        const totalServer = d.count || 0;
        const totalClient = _clientErrors.length;
        const total = totalServer + totalClient;
        const badge = document.getElementById('dash-err-badge');
        if (!badge) return;
        if (total > 0 && activeLogFeed !== 'dashboard') {
          badge.textContent = total;
          badge.style.display = '';
        } else {
          badge.style.display = 'none';
        }
      } catch (e) { /* don't recurse into _captureClientError here */ }
    },
  });
}

export function activate() {
  loadLogs();
}

export function deactivate() {}

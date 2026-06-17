// ===========================================================================
// tabs/streams.js — the self↔self channel observatory.
// ---------------------------------------------------------------------------
// Headline: a live ROSTER of streams of thought (rendered as panes by
// stream_roster.js). Click a pane to drill in — its full arc, its OWN boot
// context, and the messages it sent/received. Boot is per-stream context, so
// it lives inside the stream; there is no separate Boot tab.
//
// Below the roster: the cross-stream message log (the courier — self_inflight
// with delivery fan-out folded in) + the dashboard's one write path, a send
// composer. The operator sends attributed, never as a stream of thought.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, relativeTime } from '/static/lib/dom.js';
import { renderRoster } from '/static/lib/stream_roster.js';

// Roster state. _lastPresence / _lastMessages let the drill-down re-render
// synchronously without re-fetching; _streamOpen tracks which panes are drilled
// open (survives the 5s poll, which rebuilds the roster wholesale); _bootCache
// holds per-stream boot renders fetched lazily on first drill-in.
let _lastPresence = null;
let _lastMessages = [];
const _streamOpen = new Set();
const _bootCache = {};
const _msgExpanded = new Set();   // message ids whose body is expanded past the 4-line clamp
// Render signatures — the 5s poll calls _loadPresence/_loadMessages every tick;
// without these guards it rebuilt the roster + message feed wholesale every
// time, resetting page scroll and the scroll inside an open boot <pre> the
// operator is reading. We re-render only on a STRUCTURAL change (excluding raw
// updated_at, which ticks every poll). Toggles call _renderPresence /
// _paintMessages directly, so skipping here never strands an open pane.
let _lastPresenceFp = null;
let _lastMsgFp = null;

function _msgSignature() {
  return _lastMessages.length + '|' + (_lastMessages[0] && _lastMessages[0].id) + '|'
    + _lastMessages.reduce((n, m) => n + (m.delivered ? m.delivered.length : 0), 0);
}
function _presenceSignature() {
  const streams = (_lastPresence && _lastPresence.streams) || [];
  const lost = (_lastPresence && _lastPresence.lost) || [];
  return streams.map(s => [s.session_id, s.state, s.turn_count, s.pending_inbox_count,
    s.focus, (s.arc || '').length].join(':')).join('|')
    + '#lost:' + lost.length + '#msg:' + _msgSignature();
}

// Resolve a session id to its branch handle (via the live roster), so the
// message log reads "main → adoring-williams" instead of hex. Falls back to
// the 8-char short when the party isn't a currently-live stream.
function _handleForSession(sid) {
  if (!sid) return '?';
  const streams = (_lastPresence && _lastPresence.streams) || [];
  const s = streams.find(x => x.session_id === sid || x.short === sid.slice(0, 8));
  const b = s && s.branch && s.branch !== 'unknown' ? s.branch : '';
  if (b) return b.includes('/') ? b.slice(b.indexOf('/') + 1) : b;
  return sid.slice(0, 8);
}

export async function loadStreams() {
  await _loadMessages();    // load first so open drill-downs have the message list
  await _loadPresence();
}

// ── Presence roster ────────────────────────────────────────────────────
async function _loadPresence() {
  try {
    _lastPresence = await api.selfPresence();
    _syncSendDropdown(_lastPresence);   // cheap; safe to refresh every tick
    const fp = _presenceSignature();
    if (fp === _lastPresenceFp) return;  // nothing structural changed — don't stomp scroll
    _lastPresenceFp = fp;
    _renderPresence();
  } catch (e) { console.error('[streams] presence', e); }
}

function _renderPresence() {
  const host = document.getElementById('streams-presence');
  if (host) host.innerHTML = renderRoster(_lastPresence, {
    open: _streamOpen, boots: _bootCache, messages: _lastMessages,
  });
}

// Handle-first dropdown (branch · focus). Preserve selection across refreshes.
function _syncSendDropdown(p) {
  const sel = document.getElementById('streams-send-to');
  if (!sel) return;
  const current = sel.value;
  let html = '<option value="broadcast">broadcast (all live streams)</option>';
  for (const s of (p && p.streams) || []) {
    const handle = (s.branch && s.branch !== 'unknown') ? s.branch : (s.short || s.session_id.substring(0, 8));
    const focus = s.focus ? ' — ' + s.focus.substring(0, 40) : '';
    html += '<option value="' + escapeHtml(s.session_id) + '">' + escapeHtml(handle) + escapeHtml(focus) + '</option>';
  }
  sel.innerHTML = html;
  if (current) sel.value = current;
}

// Drill-down toggle — delegated on the persistent #streams-presence host.
// Opening lazily fetches that stream's boot captures (cached), then re-renders.
async function _onPresenceClick(e) {
  const bar = e.target.closest('[data-stream-toggle]');
  if (!bar) return;
  const sid = bar.getAttribute('data-stream-toggle');
  if (!sid) return;
  if (_streamOpen.has(sid)) {
    _streamOpen.delete(sid);
    _renderPresence();
    return;
  }
  _streamOpen.add(sid);
  _renderPresence();                 // immediate (shows "loading boot…")
  if (_bootCache[sid] === undefined) {
    try {
      const body = await api.bootRenders({ session: sid, limit: 3 });
      _bootCache[sid] = (body && body.renders) || [];
    } catch (_) { _bootCache[sid] = []; }
    if (_streamOpen.has(sid)) _renderPresence();
  }
}

// ── Cross-stream message log (courier + delivery fan-out) ──────────────
async function _loadMessages() {
  try {
    const hours = document.getElementById('streams-hours').value;
    const body = await api.selfMessages({ hours });
    _lastMessages = (body && body.messages) || [];
    const fp = _msgSignature();
    if (fp === _lastMsgFp) return;   // unchanged — don't stomp scroll/expanded bodies
    _lastMsgFp = fp;
    _paintMessages();
  } catch (e) { console.error('[streams] messages', e); }
}

// Render the message feed from cache (called by the load + the expand toggle).
function _paintMessages() {
  document.getElementById('streams-count').textContent =
    _lastMessages.length + ' message' + (_lastMessages.length === 1 ? '' : 's');
  const el = document.getElementById('feed-streams-messages');
  if (!el) return;
  if (!_lastMessages.length) {
    el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">' +
      'No messages between streams right now. Sends appear here; older ones reap ' +
      'after their TTL (then survive only as S0 traces).</div>';
    return;
  }
  el.innerHTML = _lastMessages.map(_renderMessage).join('');
}

function _renderMessage(m) {
  const broadcast = m.address === 'self:broadcast';
  const fromHandle = _handleForSession(m.from_full || m.from);
  const toSid = broadcast ? '' : (m.address || '').replace(/^self:/, '');
  const toHandle = broadcast ? 'broadcast' : _handleForSession(toSid);
  const expanded = _msgExpanded.has(m.id);

  const delivered = (m.delivered || []);
  const deliveredHtml = delivered.length
    ? '<div style="margin-top:6px;font-size:10px;color:#5a8a5a">✓ delivered → ' +
        delivered.map(d => '<span title="' + escapeHtml(d.to_full) + ' @ ' + escapeHtml(d.at || '') +
          '" style="background:#0e1a0e;border:1px solid #1e3a1e;border-radius:3px;padding:1px 6px;margin-right:4px">' +
          escapeHtml(_handleForSession(d.to_full) || d.to) + '</span>').join('') + '</div>'
    : '<div style="margin-top:6px;font-size:10px;color:#806a3a">○ pending — not yet consumed</div>';

  const refsHtml = (m.refs && m.refs.length)
    ? '<div style="margin-top:4px;font-size:10px;color:#666">refs: ' +
        m.refs.map(r => escapeHtml(String(r))).join(', ') + '</div>'
    : '';

  const bodyClass = expanded ? '' : ' msg-clamp';
  return '<div class="stream-msg" data-mid="' + escapeHtml(String(m.id || '')) + '">' +
    '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">' +
      '<div style="font-size:12px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace">' +
        '<span style="color:#ffd479">⚡</span> ' +
        '<span style="color:#7eb8ff;font-weight:600">' + escapeHtml(fromHandle) + '</span>' +
        '<span style="color:#556;margin:0 6px">→</span>' +
        '<span style="color:' + (broadcast ? '#c4a8f0' : '#cfd') + ';font-weight:600">' + escapeHtml(toHandle) + '</span>' +
      '</div>' +
      '<span style="color:#556;font-size:10px;white-space:nowrap;flex-shrink:0" title="' + escapeHtml(m.created_at || '') + '">' + relativeTime(m.created_at) + '</span>' +
    '</div>' +
    '<div class="msg-body' + bodyClass + '" style="color:#cdd;font-size:12px;margin-top:6px;line-height:1.5;white-space:pre-wrap;word-break:break-word">' + escapeHtml(m.body || '') + '</div>' +
    refsHtml +
    deliveredHtml +
  '</div>';
}

// Click a message → expand/collapse its body past the 4-line clamp.
function _onMessagesClick(e) {
  const card = e.target.closest('.stream-msg');
  if (!card) return;
  const mid = card.getAttribute('data-mid');
  if (!mid) return;
  if (_msgExpanded.has(mid)) _msgExpanded.delete(mid);
  else _msgExpanded.add(mid);
  _paintMessages();
}

// ── Send composer (the one write path) ─────────────────────────────────
export function onStreamsSendKey(event) {
  if (event && event.key === 'Enter') { event.preventDefault(); onStreamsSend(); }
}

export async function onStreamsSend() {
  const toEl = document.getElementById('streams-send-to');
  const bodyEl = document.getElementById('streams-send-body');
  const statusEl = document.getElementById('streams-send-status');
  const to = toEl ? toEl.value : '';
  const body = bodyEl ? bodyEl.value.trim() : '';
  if (!body) { if (statusEl) statusEl.textContent = 'type a message first'; return; }
  if (statusEl) statusEl.textContent = 'sending…';
  try {
    await api.selfSend({ to, body });
    if (bodyEl) bodyEl.value = '';
    if (statusEl) {
      statusEl.textContent = '✓ sent to ' + (to === 'broadcast' ? 'all streams' : to.substring(0, 8));
      setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 4000);
    }
    await loadStreams();   // refresh the log AND any open pane's message list
  } catch (e) {
    if (statusEl) statusEl.textContent = '✗ ' + (e.message || 'send failed');
    console.error('[streams] send', e);
  }
}

// ── Lifecycle ──────────────────────────────────────────────────────────
export function init() {
  // One delegated listener on the persistent presence host — survives the
  // poll-driven roster rebuilds (mirrors the traces tab pattern).
  const host = document.getElementById('streams-presence');
  if (host && !host._presenceClickBound) {
    host.addEventListener('click', _onPresenceClick);
    host._presenceClickBound = true;
  }
  const msgFeed = document.getElementById('feed-streams-messages');
  if (msgFeed && !msgFeed._msgClickBound) {
    msgFeed.addEventListener('click', _onMessagesClick);
    msgFeed._msgClickBound = true;
  }
  poll.register({
    key: 'streams',
    interval: 5000,
    activeWhen: () => {
      const tab = document.getElementById('tab-streams');
      return tab && tab.classList.contains('active');
    },
    fetcher: loadStreams,
  });
}

export function activate() { loadStreams(); }
export function deactivate() {}

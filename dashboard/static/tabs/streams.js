// ===========================================================================
// tabs/streams.js — the self↔self channel observatory.
// ---------------------------------------------------------------------------
// Two sub-views over the self-channel + boot, plus the dashboard's one write
// path (a send composer):
//
//   Messages — the courier log (self_inflight) with each message's delivery
//              fan-out folded in (self_delivered). "Who said what to whom, and
//              did it land." The delivery chips ARE the s0 `self_message`
//              marker events; the raw s0 rows also show in the Traces tab.
//   Boot     — the faithful per-session boot captures (boot_renders): the
//              exact text the daemon served at SessionStart. Collapsible.
//
// Presence (top line) is the live roster, read through the daemon. Sending
// goes POST → daemon self_send; the operator sends attributed, never as a
// stream of thought.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime, relativeTime } from '/static/lib/dom.js';

let _streamsView = 'messages';   // 'messages' | 'boot'

export function switchStreamsView(view) {
  _streamsView = view;
  document.querySelectorAll('#tab-streams .feed-btn').forEach(b => {
    b.classList.toggle('active', b.textContent.trim().toLowerCase().startsWith(view));
  });
  document.getElementById('feed-streams-messages').style.display = view === 'messages' ? '' : 'none';
  document.getElementById('feed-streams-boot').style.display     = view === 'boot' ? '' : 'none';
  // The send composer only makes sense on the Messages view.
  const compose = document.getElementById('streams-compose');
  if (compose) compose.style.display = view === 'messages' ? 'flex' : 'none';
  loadStreams();
}

export async function loadStreams() {
  await _loadPresence();
  if (_streamsView === 'messages') await _loadMessages();
  else await _loadBoot();
}

// ── Presence (live roster + send-to dropdown) ──────────────────────────
async function _loadPresence() {
  try {
    const p = await api.selfPresence();
    const line = document.getElementById('streams-presence');
    if (line) line.textContent = (p && p.line) ? '🧵 ' + p.line : 'no other streams of thought live right now';

    // Keep the send-to dropdown in sync with the live roster (broadcast
    // always first). Preserve the current selection across refreshes.
    const sel = document.getElementById('streams-send-to');
    if (sel) {
      const current = sel.value;
      let html = '<option value="broadcast">broadcast (all live streams)</option>';
      for (const s of (p && p.streams) || []) {
        const focus = s.focus ? ' — ' + s.focus.substring(0, 40) : '';
        html += '<option value="' + escapeHtml(s.session_id) + '">' +
          escapeHtml(s.short || s.session_id.substring(0, 8)) + escapeHtml(focus) + '</option>';
      }
      sel.innerHTML = html;
      if (current) sel.value = current;   // resets to broadcast if the stream went away
    }
  } catch (e) { console.error('[streams] presence', e); }
}

// ── Messages (courier log + delivery fan-out) ──────────────────────────
async function _loadMessages() {
  try {
    const hours = document.getElementById('streams-hours').value;
    const body = await api.selfMessages({ hours });
    const msgs = (body && body.messages) || [];
    document.getElementById('streams-count').textContent =
      msgs.length + ' message' + (msgs.length === 1 ? '' : 's');
    const el = document.getElementById('feed-streams-messages');
    if (!msgs.length) {
      el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">' +
        'No in-flight messages. Sends appear here; older ones reap after their TTL ' +
        '(then survive only as S0 traces).</div>';
      return;
    }
    el.innerHTML = msgs.map(_renderMessage).join('');
  } catch (e) { console.error('[streams] messages', e); }
}

function _renderMessage(m) {
  // letter = reflective (blue), signal = imperative (amber).
  const isLetter = m.intent === 'letter';
  const accent = isLetter ? '#7eb8ff' : '#ffaa33';
  const target = m.address === 'self:broadcast'
    ? 'broadcast'
    : (m.address || '').replace(/^self:/, '').substring(0, 8) || '—';

  const delivered = (m.delivered || []);
  const deliveredHtml = delivered.length
    ? '<div style="margin-top:4px;font-size:10px;color:#5a8a5a">✓ delivered → ' +
        delivered.map(d => '<span title="' + escapeHtml(d.to_full) + ' @ ' + escapeHtml(d.at || '') +
          '" style="background:#0e1a0e;border:1px solid #1e3a1e;border-radius:3px;padding:0 4px;margin-right:3px">' +
          escapeHtml(d.to) + '</span>').join('') + '</div>'
    : '<div style="margin-top:4px;font-size:10px;color:#806a3a">○ pending — not yet consumed</div>';

  const refsHtml = (m.refs && m.refs.length)
    ? '<div style="margin-top:3px;font-size:10px;color:#666">refs: ' +
        m.refs.map(r => escapeHtml(String(r))).join(', ') + '</div>'
    : '';

  return '<div class="stream-msg" style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid ' + accent + ';padding:8px 12px">' +
    '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">' +
      '<div style="font-size:12px;min-width:0">' +
        '<span style="color:' + accent + ';font-weight:bold">⚡ ' + escapeHtml(m.from || '?') + '</span>' +
        '<span style="color:#555;margin:0 5px">→</span>' +
        '<span style="color:#bbb">' + escapeHtml(target) + '</span>' +
        '<span style="background:#1a1a2a;color:' + accent + ';font-size:9px;padding:1px 5px;border-radius:3px;margin-left:8px">' + escapeHtml(m.intent || 'signal') + '</span>' +
      '</div>' +
      '<span style="color:#555;font-size:10px;white-space:nowrap" title="' + escapeHtml(m.created_at || '') + '">' + relativeTime(m.created_at) + '</span>' +
    '</div>' +
    '<div style="color:#ddd;font-size:12px;margin-top:5px;white-space:pre-wrap;word-break:break-word">' + escapeHtml(m.body || '') + '</div>' +
    refsHtml +
    deliveredHtml +
  '</div>';
}

// ── Boot (faithful per-session captures) ───────────────────────────────
async function _loadBoot() {
  try {
    const body = await api.bootRenders({ limit: 30 });
    const renders = (body && body.renders) || [];
    document.getElementById('streams-count').textContent =
      renders.length + ' boot' + (renders.length === 1 ? '' : 's');
    const el = document.getElementById('feed-streams-boot');
    if (!renders.length) {
      el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">' +
        'No boot captures yet. Each SessionStart records the exact text the ' +
        'daemon served — the next boot will appear here.</div>';
      return;
    }
    el.innerHTML = renders.map(_renderBoot).join('');
  } catch (e) { console.error('[streams] boot', e); }
}

function _renderBoot(b) {
  // The boot text is large (~2k tokens); collapse behind a native <details>.
  return '<details class="boot-render" style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid #9c7bd6;padding:6px 12px">' +
    '<summary style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:8px;list-style:none">' +
      '<span style="font-size:12px;color:#c4a8f0;font-weight:bold">boot · ' + escapeHtml(b.session_short || '?') + '</span>' +
      '<span style="color:#666;font-size:10px">' + (b.char_count || 0) + ' chars · ' +
        escapeHtml(b.user || '') + (b.project ? '/' + escapeHtml(b.project) : '') +
        ' · <span title="' + escapeHtml(b.created_at || '') + '">' + relativeTime(b.created_at) + '</span></span>' +
    '</summary>' +
    '<pre style="white-space:pre-wrap;word-break:break-word;color:#cdd;font-size:11px;line-height:1.45;margin:8px 0 4px;font-family:ui-monospace,Menlo,monospace">' +
      escapeHtml(b.text || '') + '</pre>' +
  '</details>';
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
    await _loadMessages();   // show it land in the log immediately
  } catch (e) {
    if (statusEl) statusEl.textContent = '✗ ' + (e.message || 'send failed');
    console.error('[streams] send', e);
  }
}

// ── Lifecycle ──────────────────────────────────────────────────────────
export function init() {
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

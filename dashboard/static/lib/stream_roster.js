// ===========================================================================
// lib/stream_roster.js — the live roster of streams of thought, as PANES.
// ---------------------------------------------------------------------------
// A stream of thought is a live process running in a worktree, so it reads as
// a terminal/window pane — a title bar (pulsing liveness light + ⎇ branch
// handle + state) over a body — NOT another flat memory card like Traces/Live.
// Identity = the branch/worktree handle (brain principle: "one stream, one
// worktree — your handle is your branch name"); hex is the subtitle.
//
// Click a pane's title bar → it drills open inline: full arc + the stream's
// OWN boot context (folded in — there is no separate Boot tab) + the messages
// it sent/received. Collapsed panes clamp the arc.
//
// Pure presentation. Structural styling lives in style.css (.stream-pane /
// .stream-titlebar / .live-light + the streamPulse keyframe); this returns an
// HTML string (matches streams.js's idiom). State (which panes are open, the
// per-stream boot cache, the global message list) is owned by streams.js and
// passed in — so the roster re-renders correctly under the 5s presence poll.
// ===========================================================================

import { escapeHtml, relativeTime } from '/static/lib/dom.js';
import { sessionLabel, sessionColor, sessionTooltip } from '/static/lib/sessions.js';

const _LIVE = {
  active:  { dot: 'active',  label: 'active',  color: '#33d17a' },
  dormant: { dot: 'dormant', label: 'dormant', color: '#ffaa33' },
  lost:    { dot: 'lost',    label: 'lost',    color: '#777' },
};

const _ARC_CLAMP = 150;

function _shortCwd(cwd) { return (cwd || '').replace(/^\/Users\/[^/]+/, '~'); }
function _dur(iso) { const r = relativeTime(iso); return r ? r.replace(/\s*ago$/, '') : ''; }
// Handle + hue + hover all come from the session registry (lib/sessions.js) —
// the same resolution the stream rail and every moment chip use, so one stream
// reads as one identity wherever it appears. This module used to derive the
// handle itself; two derivations meant two chances to drift.
function _handle(s) { return sessionLabel(s.session_id || '') || s.short || ''; }
// Transient = a freshly-spawned agent/shell with nothing to show. An active
// stream, or one with a focus/arc, is real even before its turn_count is
// stamped — don't demote it to the dim row.
function _isTransient(s) {
  if (s.state === 'active') return false;
  if (s.focus && s.focus.trim()) return false;
  if (s.arc && s.arc.trim()) return false;
  return !s.turn_count || s.turn_count === 0;
}

// ── drill-down sub-blocks ───────────────────────────────────────────────────
function _bootBlock(boots) {
  if (boots === undefined) return '<div style="color:#667;font-size:11px;margin-top:8px">loading boot context…</div>';
  if (!boots || !boots.length) return '<div style="color:#566;font-size:11px;margin-top:8px">No boot capture recorded for this stream.</div>';
  const b = boots[0];
  const more = boots.length > 1 ? ' <span style="color:#566">· ' + boots.length + ' boots</span>' : '';
  return '<details class="stream-boot" style="margin-top:8px">'
    + '<summary style="cursor:pointer;list-style:none;color:#c4a8f0;font-size:11px;font-weight:600">'
    + '🌅 boot context · ' + (b.char_count || 0) + ' chars · '
    + '<span style="color:#778;font-weight:400" title="' + escapeHtml(b.created_at || '') + '">' + escapeHtml(relativeTime(b.created_at)) + '</span>' + more + '</summary>'
    + '<pre style="white-space:pre-wrap;word-break:break-word;color:#cdd;font-size:11px;line-height:1.45;margin:6px 0 2px;max-height:340px;overflow:auto;'
    + 'background:#0c0c16;border:1px solid #181826;border-radius:4px;padding:8px 10px;font-family:ui-monospace,Menlo,monospace">'
    + escapeHtml(b.text || '') + '</pre></details>';
}

// This stream's sent + received messages, pulled from the global courier list.
function _msgBlock(sid, messages) {
  if (!messages) return '';
  const rows = [];
  for (const m of messages) {
    const out = m.from_full === sid;
    const directedTo = m.address === 'self:' + sid;
    const gotIt = (m.delivered || []).some(d => d.to_full === sid);
    const incoming = directedTo || (m.address === 'self:broadcast' && gotIt);
    if (!out && !incoming) continue;
    const clean = (m.body || '').replace(/\s+/g, ' ');
    const body = clean.length > 90 ? clean.slice(0, 90) + '…' : clean;
    if (out) {
      const tgt = m.address === 'self:broadcast' ? 'broadcast' : (m.address || '').replace(/^self:/, '').slice(0, 8);
      const ok = (m.delivered || []).length;
      rows.push('<div style="font-size:11px;margin:3px 0;color:#9ab"><span style="color:#7eb8ff">→ ' + escapeHtml(tgt) + '</span> '
        + '<span style="color:#cdd">' + escapeHtml(body) + '</span> '
        + (ok ? '<span style="color:#5a8a5a">✓</span>' : '<span style="color:#806a3a">○</span>') + '</div>');
    } else {
      rows.push('<div style="font-size:11px;margin:3px 0;color:#9ab"><span style="color:#c4a8f0">← ' + escapeHtml((m.from || '?').slice(0, 8)) + '</span> '
        + '<span style="color:#cdd">' + escapeHtml(body) + '</span></div>');
    }
    if (rows.length >= 8) break;
  }
  if (!rows.length) return '<div style="color:#566;font-size:11px;margin-top:8px">No messages to or from this stream.</div>';
  return '<div style="margin-top:8px"><div style="color:#667;font-size:9px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px">messages</div>' + rows.join('') + '</div>';
}

// ── pane ─────────────────────────────────────────────────────────────────
function _pane(s, open, boots, messages) {
  const live = _LIVE[s.state] || _LIVE.dormant;
  const sid = escapeHtml(s.session_id || '');
  const handle = escapeHtml(_handle(s));

  // The stream's hue — the same one its moments carry in the Live feed and
  // its activation carries in the graph, so a stream is one color everywhere.
  const hue = sessionColor(s.session_id || '');
  let h = '<div class="stream-pane ' + (s.state === 'active' ? 'is-active ' : '') + (open ? 'is-open' : '')
    + '" data-sid="' + sid + '" style="--stream-color:' + hue + '">';

  // title bar (the click target for drill-down)
  h += '<div class="stream-titlebar" data-stream-toggle="' + sid + '">'
    + '<span class="live-light ' + live.dot + '"></span>'
    + '<span class="stream-handle" style="color:' + hue + '" title="' + escapeHtml(sessionTooltip(s.session_id || '')) + '"><span class="glyph">⎇</span>' + handle + '</span>'
    + '<span class="stream-hex">' + escapeHtml(s.short || '') + '</span>'
    + '<span style="flex:1"></span>'
    + '<span class="stream-state" style="color:' + live.color + '" title="' + escapeHtml(s.updated_at || '') + '">'
    + live.label + (s.updated_at ? ' · ' + escapeHtml(relativeTime(s.updated_at)) : '') + '</span>'
    + '<span style="color:#566;font-size:10px;width:10px">' + (open ? '▾' : '▸') + '</span>'
    + '</div>';

  // body
  h += '<div class="stream-body">';

  // metrics
  const stats = [];
  if (s.cwd) stats.push('<span style="font-family:ui-monospace,monospace">' + escapeHtml(_shortCwd(s.cwd)) + '</span>');
  if (s.turn_count) stats.push(s.turn_count + ' turn' + (s.turn_count === 1 ? '' : 's'));
  const tenure = _dur(s.session_started_at);
  if (tenure) stats.push('up ' + escapeHtml(tenure));
  if (stats.length) {
    h += '<div style="color:#778;font-size:10px">' + stats.join('<span style="color:#445;margin:0 5px">·</span>');
    if (s.pending_inbox_count) h += '<span style="background:#2a1a00;border:1px solid #c83;border-radius:3px;color:#ffaa33;font-size:9px;padding:0 5px;margin-left:6px">📥 ' + s.pending_inbox_count + ' waiting</span>';
    h += '</div>';
  }

  // focus
  if (s.focus && s.focus.trim()) {
    h += '<div style="color:#cfd;font-size:12px;margin-top:6px;white-space:pre-wrap;word-break:break-word">' + escapeHtml(s.focus) + '</div>';
  }

  // arc — clamped when closed, full when drilled open
  const arc = (s.arc || '').trim();
  if (arc) {
    const shown = open ? arc : (arc.length > _ARC_CLAMP ? arc.slice(0, _ARC_CLAMP) + '…' : arc);
    h += '<div style="color:#8a8a9a;font-size:11px;margin-top:6px;line-height:1.5;white-space:pre-wrap;word-break:break-word;border-top:1px solid #15151f;padding-top:6px">'
      + escapeHtml(shown) + '</div>';
  }

  // drill-down: boot + messages
  if (open) {
    h += _bootBlock(boots);
    h += _msgBlock(s.session_id, messages);
  }

  h += '</div></div>';
  return h;
}

function _transientRow(s) {
  const live = _LIVE[s.state] || _LIVE.dormant;
  return '<div class="stream-transient" style="display:flex;gap:8px;align-items:center;padding:3px 12px;opacity:.5;font-size:11px">'
    + '<span class="live-light ' + live.dot + '" style="width:6px;height:6px"></span>'
    + '<span style="color:#8a8a9a;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace">'
    + escapeHtml(s.short || '') + ' · ' + escapeHtml(_shortCwd(s.cwd)) + ' · just spawned, no activity yet</span>'
    + '<span style="color:#556;font-size:10px;white-space:nowrap" title="' + escapeHtml(s.updated_at || '') + '">' + escapeHtml(relativeTime(s.updated_at)) + '</span>'
    + '</div>';
}

function _lostRow(s) {
  return '<div class="stream-lost" style="display:flex;gap:8px;align-items:center;padding:3px 12px;opacity:.4;font-size:11px">'
    + '<span class="live-light lost" style="width:6px;height:6px"></span>'
    + '<span style="color:#888;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
    + escapeHtml(_handle(s)) + (s.focus ? ' · ' + escapeHtml(s.focus.slice(0, 60)) : '') + '</span>'
    + '<span style="color:#556;font-size:10px;white-space:nowrap">lost · ' + escapeHtml(relativeTime(s.updated_at)) + '</span>'
    + '</div>';
}

/** Render the presence roster. `presence` = /api/self-presence payload
 *  ({streams, lost}); `opts` = { open:Set(sid), boots:{sid:[...]},
 *  messages:[...] } owned by streams.js. */
export function renderRoster(presence, opts = {}) {
  const streams = (presence && presence.streams) || [];
  const lost = (presence && presence.lost) || [];
  const open = opts.open || new Set();
  const boots = opts.boots || {};
  const messages = opts.messages || null;

  if (!streams.length && !lost.length) {
    return '<div style="color:#667;font-size:12px;padding:8px 4px">🧵 No other streams of thought live right now.</div>';
  }

  const real = streams.filter(s => !_isTransient(s));
  const transient = streams.filter(_isTransient);

  let h = '<div style="color:#778;font-size:11px;padding:4px 4px 2px;font-weight:600">🧵 '
    + real.length + ' stream' + (real.length === 1 ? '' : 's') + ' of thought live</div>';
  h += real.map(s => _pane(s, open.has(s.session_id), boots[s.session_id], messages)).join('');
  if (transient.length) h += transient.map(_transientRow).join('');
  if (lost.length) {
    h += '<div style="color:#556;font-size:10px;padding:6px 4px 2px">recently lost</div>';
    h += lost.map(_lostRow).join('');
  }
  return h;
}

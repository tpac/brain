"""Self-channel signal — the directed message (Phase 2a, pull courier).

The "hearing" half of self<->self: one stream sends a message addressed to
another live stream (self:<sid>) or to self:broadcast; the recipient consumes
it once. Presence is the passive look; a signal is a deliberate reach — Anchor
authors it, unlike the encoder-written letter.

Storage: the `self_inflight` courier in brain_logs.db (schema.LOG_TABLES), with
consumption tracked in `self_delivered` (one row per (message, recipient) — so
broadcast fans out and each recipient consumes exactly once). Phase 2a is PULL:
a recipient drains its inbox on demand. Phase 2b wires auto-delivery into
Observation at a hook (traced as the s0 `self_message` marker).

Writes go through `brain.write_lock` around `brain.logs_conn` — the canonical
shared-logs-writer pattern (see brain.discard_session_context). TTL is
per-message: send() resolves it by ADDRESS (broadcast vs directed, config-
tunable) and stamps `expires_at` via the clock contract (iso_now/iso_after);
readers filter `expires_at > now` and the reaper deletes `expires_at <= now`.
"""
import json
import uuid

from servers.clock import iso_now, iso_after
from servers.scales.self_channel import self_contract


def _resolve_ttl_hours(brain, address):
    """Per-message TTL in hours by address, with runtime config override.
    Defaults are documented in self_contract (BROADCAST/DIRECTED_TTL_HOURS);
    operators tune via brain.get_config('self_channel.{kind}_ttl_hours'). Coerced
    to float so a string-valued config ('2') is honored."""
    kind = self_contract.ttl_kind_for(address)
    default = (self_contract.BROADCAST_TTL_HOURS if kind == 'broadcast'
               else self_contract.DIRECTED_TTL_HOURS)
    return float(brain.get_config('self_channel.%s_ttl_hours' % kind, default))


def send(brain, from_session, address, body, intent=None, refs=None):
    """Place a directed/broadcast self-message in the courier. Returns its record.

    Stores the body and refs IN FULL — no truncation here. Per the self-channel
    truncation contract (self_contract), the SINGLE truncation point is delivery
    render (render_received_block), and it is loud; storage keeps everything so
    the dashboard always shows the message untruncated. The only guard is a
    non-empty body."""
    body = (body or '').strip()
    if not body:
        raise ValueError('self.signal.send: empty body')
    if intent not in self_contract.INTENTS:
        intent = self_contract.default_intent(address)
    refs_json = json.dumps(list(refs or []))
    mid = uuid.uuid4().hex[:12]
    created_at = iso_now()
    # Per-message TTL: resolved by address (broadcast ephemeral, directed waits)
    # and stamped as a future expires_at. Readers filter on it; nothing
    # recomputes a cutoff. The sub-ms skew vs created_at is irrelevant (TTL is
    # hours) and nothing compares the two columns.
    expires_at = iso_after(hours=_resolve_ttl_hours(brain, address))
    # Serialize the shared logs_conn write (mirrors brain.discard_session_context).
    with brain.write_lock:
        brain.logs_conn.execute(
            'INSERT INTO self_inflight '
            '(id, from_session, address, intent, body, refs, created_at, expires_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (mid, from_session or '', address, intent, body, refs_json,
             created_at, expires_at))
        brain.logs_conn.commit()
    return {'id': mid, 'address': address, 'intent': intent,
            'created_at': created_at, 'expires_at': expires_at}


def resolve_to(brain, to):
    """Resolve an MCP `to` to a delivery address, gracefully — returns
    (address, error) with exactly one non-None.

    'broadcast' and a full session UUID are canonical, honored directly (a UUID
    works even when the target isn't in the live roster this instant — it drains
    within TTL). A shorter form — the 8-char short you see in a message (an
    id-prefix) — is matched against the LIVE roster: unique → that stream;
    ambiguous → names the candidates; none → loud (which usefully says the stream
    is dormant/lost, so silence is never mistaken for delivery)."""
    to = (to or '').strip()
    if not to:
        return None, "self_send: empty 'to'"
    if to == 'broadcast':
        return self_contract.ADDR_BROADCAST, None
    if self_contract.is_session_id(to):
        return self_contract.address_for_stream(to), None
    window = self_contract.ROSTER_LIVE_WINDOW_MIN + self_contract.ROSTER_LOST_GRACE_MIN
    matches = []
    for r in brain.present_streams(window_min=window, limit=50):
        sid = r.get('session_id', '')
        if sid and sid.startswith(to):
            matches.append(sid)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return self_contract.address_for_stream(matches[0]), None
    if len(matches) > 1:
        return None, ("self_send: '%s' matches %d live streams (%s) — use the full "
                      "session id to disambiguate"
                      % (to, len(matches), ", ".join(s[:8] for s in matches)))
    return None, ("self_send: no live stream matches '%s' — it may be dormant or lost. "
                  "Use its full session id, or self_presence to see who's live." % to)


# The pending-inbox SELECT — the SINGLE source for drain (consume-once) and peek
# (read-only). A row is pending for `to_session` when it's addressed to it
# (directed or broadcast), unexpired, not its own broadcast, and not already
# delivered. drain wraps this in write_lock + a self_delivered stamp; peek just
# reads. One query means the two can't drift (they used to carry a "keep in
# lockstep" note — this removes the hazard).
_PENDING_INBOX_SQL = (
    'SELECT id, from_session, intent, body, created_at FROM self_inflight '
    'WHERE address IN (?, ?) AND expires_at > ? '
    'AND from_session != ? '                       # not your own broadcast
    'AND id NOT IN (SELECT message_id FROM self_delivered WHERE to_session = ?) '
    'ORDER BY created_at')


def _pending_rows(conn, to_session, now):
    """Pending-inbox rows for `to_session` at time `now`, oldest first
    (id, from_session, intent, body, created_at). Read-only; the single query
    shared by drain_inbox and peek_inbox."""
    directed, broadcast = self_contract.routes_at_turn(to_session)
    return conn.execute(
        _PENDING_INBOX_SQL,
        (directed, broadcast, now, to_session, to_session)).fetchall()


def drain_inbox(brain, to_session):
    """Consume-once: deliver undelivered, unexpired messages addressed to this
    stream (directed self:<id> + self:broadcast). Returns rendered messages,
    oldest first. Marking-delivered is atomic vs concurrent drains (write_lock +
    the self_delivered PK)."""
    if not to_session:
        return []
    now = iso_now()
    out = []
    with brain.write_lock:
        rows = _pending_rows(brain.logs_conn, to_session, now)
        for mid, from_session, intent, body, created_at in rows:
            brain.logs_conn.execute(
                'INSERT OR IGNORE INTO self_delivered (message_id, to_session, delivered_at) '
                'VALUES (?, ?, ?)', (mid, to_session, now))
            short = (from_session or '')[:8]
            out.append({
                'id': mid,
                'from': short,
                'intent': intent,
                'body': body,
                'created_at': created_at,
                'rendered': self_contract.render_signal(body, stream_short=short),
            })
        brain.logs_conn.commit()
    return out


def peek_inbox(brain, to_session):
    """Read-only twin of drain_inbox: return pending (undelivered, unexpired)
    messages addressed to this stream WITHOUT consuming them.

    Shares the pending-inbox query with drain_inbox (_pending_rows) — same
    filter (directed + broadcast routes, unexpired, not your own broadcast, not
    already delivered) — but NO self_delivered write and NO write_lock. The
    /watch-live poller calls this every ~1.5s to detect arrivals; the real
    consume-once drain still happens in drain_inbox at the Stop hook."""
    if not to_session:
        return []
    now = iso_now()
    rows = _pending_rows(brain.logs_conn, to_session, now)
    return [{'id': mid, 'from': (from_session or '')[:8], 'intent': intent,
             'body': body, 'created_at': created_at}
            for mid, from_session, intent, body, created_at in rows]


def drain_and_render(brain, to_session):
    """Phase 2b delivery primitive — drain this stream's inbox and render the
    pending messages into one budgeted block. Returns (block, n_drained), or
    ("", 0) when empty. Consume-once is inherited from drain_inbox.

    Callers wrap the block for their channel: PreToolUse prepends it to a tool's
    `reason` (tool feedback, the instant before a mutating action); Stop returns
    it as a `decision:block` reason (the backstop for no-tool turns). Delivery
    deliberately does NOT ride on_prompt — that channel is passive, competes with
    recall, and would win the consume-once race against these higher-salience
    hooks. The caller owns tracing (it holds the chain id)."""
    pending = drain_inbox(brain, to_session)
    if not pending:
        return "", 0
    return self_contract.render_received_block(pending), len(pending)


def reap_expired(brain):
    """Delete messages past their TTL (the dead-letter sweep) + orphan delivery
    rows. Returns count reaped. Wired into the daemon's idle-maintenance tick
    (daemon_server._run_idle_maintenance); safe to call anytime."""
    now = iso_now()
    with brain.write_lock:
        # expires_at <= now → dead. IS NULL → a pre-expires_at legacy row (column
        # added to the existing courier); send() always stamps it, so a NULL can
        # only be legacy — treat as dead and sweep it.
        cur = brain.logs_conn.execute(
            'DELETE FROM self_inflight WHERE expires_at <= ? OR expires_at IS NULL', (now,))
        reaped = cur.rowcount or 0
        brain.logs_conn.execute(
            'DELETE FROM self_delivered '
            'WHERE message_id NOT IN (SELECT id FROM self_inflight)')
        brain.logs_conn.commit()
    return reaped


def outbox(brain, from_session, limit=20):
    """Delivery status of messages THIS stream SENT — the sender-side view that
    until now only the dashboard had. Per recent in-flight message: which streams
    have drained (read) it and when, and — for a DIRECTED send — whether the named
    target is still pending. Closes the gap a sender hit when a reply never came:
    it tells "delivered, not acted on" apart from "never delivered". Read-only —
    pure SELECT, no write_lock.

    Broadcast has no fixed recipient set (any live stream may yet drain it), so
    `pending` is reported only for a directed address; broadcasts just list who has
    drained so far."""
    if not from_session:
        return {'messages': []}
    now = iso_now()
    rows = brain.logs_conn.execute(
        'SELECT id, address, intent, body, created_at FROM self_inflight '
        'WHERE from_session = ? AND expires_at > ? '
        'ORDER BY created_at DESC LIMIT ?',
        (from_session, now, limit)).fetchall()
    out = []
    for mid, address, intent, body, created_at in rows:
        delivered = brain.logs_conn.execute(
            'SELECT to_session, delivered_at FROM self_delivered '
            'WHERE message_id = ? ORDER BY delivered_at', (mid,)).fetchall()
        rec = {
            'id': mid,
            'address': address,
            'intent': intent,
            'created_at': created_at,
            'preview': (body or '')[:120] + (' …' if len(body or '') > 120 else ''),
            'delivered_to': [{'to': (ts or '')[:8], 'at': at} for ts, at in delivered],
        }
        target = address.split(':', 1)[1] if ':' in address else address
        if target == 'broadcast':
            rec['broadcast'] = True
        else:
            rec['target'] = target[:8]
            rec['pending'] = not any((ts or '') == target for ts, _ in delivered)
        out.append(rec)
    return {'messages': out}

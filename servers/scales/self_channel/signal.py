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
shared-logs-writer pattern (see brain.discard_session_context). TTL is enforced
by created_at + DEFAULT_SIGNAL_TTL_HOURS at drain/reap, using the clock contract
(iso_now to write, iso_cutoff to filter).
"""
import json
import uuid

from servers.clock import iso_now, iso_cutoff
from servers.scales.self_channel import self_contract


def send(brain, from_session, address, body, intent=None, refs=None, from_label=None):
    """Place a directed/broadcast self-message in the courier. Returns its record.

    Stores the body and refs IN FULL — no truncation here. Per the self-channel
    truncation contract (self_contract), the SINGLE truncation point is delivery
    render (render_received_block), and it is loud; storage keeps everything so
    the dashboard always shows the message untruncated. The only guard is a
    non-empty body. `from_label` (optional) persists a human display name for the
    sender so recipients see it and can reply by it — stored like session focus."""
    body = (body or '').strip()
    if not body:
        raise ValueError('self.signal.send: empty body')
    if from_label and from_label.strip():
        brain.set_config(self_contract.label_key(from_session or ''), from_label.strip())
    if intent not in self_contract.INTENTS:
        intent = self_contract.default_intent(address)
    refs_json = json.dumps(list(refs or []))
    mid = uuid.uuid4().hex[:12]
    created_at = iso_now()
    # Serialize the shared logs_conn write (mirrors brain.discard_session_context).
    with brain.write_lock:
        brain.logs_conn.execute(
            'INSERT INTO self_inflight '
            '(id, from_session, address, intent, body, refs, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (mid, from_session or '', address, intent, body, refs_json, created_at))
        brain.logs_conn.commit()
    return {'id': mid, 'address': address, 'intent': intent, 'created_at': created_at}


def resolve_to(brain, to):
    """Resolve an MCP `to` to a delivery address, gracefully — returns
    (address, error) with exactly one non-None.

    'broadcast' and a full session UUID are canonical, honored directly (a UUID
    works even when the target isn't in the live roster this instant — it drains
    within TTL). A shorter form (a label, case-insensitive, or an id-prefix) is
    matched against the LIVE roster: unique → that stream; ambiguous → names the
    candidates; none → loud (which usefully says the stream is dormant/lost, so
    silence is never mistaken for delivery)."""
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
        label = brain.get_config(self_contract.label_key(sid), '') or ''
        if (label and label.lower() == to.lower()) or (sid and sid.startswith(to)):
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


def drain_inbox(brain, to_session):
    """Consume-once: deliver undelivered, unexpired messages addressed to this
    stream (directed self:<id> + self:broadcast). Returns rendered messages,
    oldest first. Marking-delivered is atomic vs concurrent drains (write_lock +
    the self_delivered PK)."""
    if not to_session:
        return []
    directed, broadcast = self_contract.routes_at_turn(to_session)
    cutoff = iso_cutoff(hours=self_contract.DEFAULT_SIGNAL_TTL_HOURS)
    out = []
    with brain.write_lock:
        rows = brain.logs_conn.execute(
            'SELECT id, from_session, intent, body, created_at FROM self_inflight '
            'WHERE address IN (?, ?) AND created_at > ? '
            'AND from_session != ? '                     # don't hear your own broadcast
            'AND id NOT IN (SELECT message_id FROM self_delivered WHERE to_session = ?) '
            'ORDER BY created_at',
            (directed, broadcast, cutoff, to_session, to_session)).fetchall()
        now = iso_now()
        for mid, from_session, intent, body, created_at in rows:
            brain.logs_conn.execute(
                'INSERT OR IGNORE INTO self_delivered (message_id, to_session, delivered_at) '
                'VALUES (?, ?, ?)', (mid, to_session, now))
            short = (from_session or '')[:8]
            who = brain.get_config(self_contract.label_key(from_session or ''), '') or short
            out.append({
                'id': mid,
                'from': who,
                'intent': intent,
                'body': body,
                'created_at': created_at,
                'rendered': self_contract.render_signal(body, stream_short=who),
            })
        brain.logs_conn.commit()
    return out


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
    cutoff = iso_cutoff(hours=self_contract.DEFAULT_SIGNAL_TTL_HOURS)
    with brain.write_lock:
        cur = brain.logs_conn.execute(
            'DELETE FROM self_inflight WHERE created_at <= ?', (cutoff,))
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
    cutoff = iso_cutoff(hours=self_contract.DEFAULT_SIGNAL_TTL_HOURS)
    rows = brain.logs_conn.execute(
        'SELECT id, address, intent, body, created_at FROM self_inflight '
        'WHERE from_session = ? AND created_at > ? '
        'ORDER BY created_at DESC LIMIT ?',
        (from_session, cutoff, limit)).fetchall()
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

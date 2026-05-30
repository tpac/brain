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


def send(brain, from_session, address, body, intent=None, refs=None):
    """Place a directed/broadcast self-message in the courier. Returns its record."""
    body = (body or '').strip()
    if not body:
        raise ValueError('self.signal.send: empty body')
    body = body[:self_contract.SIGNAL_BODY_MAX]
    if intent not in self_contract.INTENTS:
        intent = self_contract.default_intent(address)
    refs_json = json.dumps(list(refs or [])[:self_contract.REFS_MAX])
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


def deliver_into_context(brain, to_session, context):
    """Phase 2b — delivery-into-Observation. Drain this stream's inbox and merge
    the messages into an Observation `context` string. Returns (new_context,
    n_delivered).

    A directed tap is imperative, so it PREPENDS — seen ahead of recall — but
    render_received_block budgets it so it can't crowd recall out. Consume-once
    is inherited from drain_inbox. The caller owns tracing (it holds the chain
    id) — this returns the count so the caller can decide whether to trace."""
    pending = drain_inbox(brain, to_session)
    if not pending:
        return context, 0
    block = self_contract.render_received_block(pending)
    merged = (block + "\n\n" + context) if context else block
    return merged, len(pending)


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

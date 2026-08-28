"""Thalamus mechanics — file / pull / resolve / withdraw / expire.

One producer door (`file()`), three entrances: Anchor's MCP tools
(dispatch_thalamus.py), agent toolsets (Phase 2), direct code calls. Routing
lives inside the door: a live-now FYI delegates to the courier broadcast and
goes terminal immediately; everything with a clock, a window, or an answer
queues. Delivery is PULL-ONLY at the two moments that provably land — the Stop
drain and the boot render — so the Thalamus never enumerates sessions, never
pushes, holds no roster: each session self-serves against the item list and
records its own delivery in the ledger at render time (annotate-at-render —
the only visibility mechanism that survives courier receipt expiry, brain
node 8a170558).

Storage: thalamus_items / thalamus_deliveries in brain_logs.db (DDL in
servers/schema.py). Reads via brain.logs_conn; writes serialize on
brain.logs_write_lock through brain.logs_conn_w — the self-channel pattern.
Time is WALL-CLOCK throughout (courier-class deadlines; see thalamus_contract).

Design: docs/THALAMUS-DESIGN.md
"""

import json
import uuid

from servers.clock import iso_now, iso_after
from servers.scales.thalamus import thalamus_contract as tc


def _row_to_item(row):
    """sqlite row tuple → item dict (column order = the SELECT below)."""
    return {
        'id': row[0], 'source': row[1], 'body': row[2],
        'refs': json.loads(row[3] or '[]'),
        'audience': row[4], 'target_session': row[5],
        'needs_answer': bool(row[6]), 'dedup_key': row[7],
        'deliver_at': row[8], 'expires_at': row[9],
        'state': row[10], 'answer': row[11],
        'created_at': row[12],
    }


_ITEM_COLS = ('id, source, body, refs, audience, target_session, '
              'needs_answer, dedup_key, deliver_at, expires_at, state, '
              'answer, created_at')


def _default_expires(needs_answer, audience, deliver_at):
    """The item's window when the producer names none. Asks get the long loud
    window; standing notices the notice window; once-items live until shortly
    after they fire."""
    if needs_answer:
        return iso_after(days=tc.ASK_EXPIRES_DAYS)
    if audience == tc.AUDIENCE_ALL:
        return iso_after(days=tc.NOTICE_EXPIRES_DAYS)
    if deliver_at:
        from datetime import datetime as _dt, timedelta as _td
        base = _dt.fromisoformat(deliver_at)
        return (base + _td(days=tc.REMIND_GRACE_DAYS)).isoformat()
    return iso_after(days=tc.REMIND_GRACE_DAYS)


def file(brain, source, body, *, needs_answer=False, when=None, for_whom=None,
         refs=None, dedup_key=None, expires=None, session_id=''):
    """The single producer door. Returns {'filed': True, ...} or a LOUD
    synchronous rejection {'filed': False, 'error': <guidance>} — the budget
    guard lives here, at the write boundary, where the rejection lands in the
    caller's loop while it can still adapt (never in a sweeper hours later).

    Routing: for_whom 'live' delegates to the courier broadcast (requires a
    filing session — the locked stream-speech render must stay honest) and the
    row goes terminal 'sent'. Everything else queues for pull delivery.
    A repeat (source, dedup_key) UPDATES the open item instead of inserting —
    identity is producer-owned or absent, never derived from text.
    """
    body = (body or '').strip()
    if not body:
        return {'filed': False, 'error': 'thalamus.file: empty body'}
    if not source:
        return {'filed': False, 'error': 'thalamus.file: source is required'}

    try:
        route, audience, target_session = tc.resolve_for_whom(
            for_whom, needs_answer)
        deliver_at = tc.resolve_when(when)
        expires_at = (tc.resolve_when(expires) if expires
                      else _default_expires(needs_answer, audience, deliver_at))
    except ValueError as e:
        return {'filed': False, 'error': str(e)}

    now = iso_now()
    refs_json = json.dumps(list(refs or []))

    if route == 'live':
        # Fire-and-forget to whoever is alive NOW — the courier's contract.
        # Terminal 'sent' row kept for observability (thalamus_list), no
        # ledger lifecycle: the courier owns delivery and death (1h TTL).
        if not session_id:
            return {'filed': False, 'error':
                    "thalamus.file: for_whom='live' requires a filing session "
                    "(the stream-speech render needs an honest sender); "
                    "machine live-now is Phase 3"}
        from servers.scales.self_channel import signal, self_contract
        sent = signal.send(brain, from_session=session_id,
                           address=self_contract.ADDR_BROADCAST,
                           body=body, refs=refs)
        item_id = 'th_%s' % uuid.uuid4().hex[:8]
        with brain.logs_write_lock:
            brain.logs_conn_w.execute(
                'INSERT INTO thalamus_items (id, source, body, refs, audience,'
                ' target_session, needs_answer, dedup_key, deliver_at,'
                ' expires_at, state, answer, created_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?, ?, ?, ?)',
                (item_id, source, body, refs_json, '', '', dedup_key or '',
                 sent['expires_at'], tc.STATE_SENT, '', now))
            brain.logs_conn_w.commit()
        return {'filed': True, 'id': item_id, 'route': 'live',
                'courier_id': sent['id']}

    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        if dedup_key:
            cur = conn.execute(
                'SELECT id FROM thalamus_items WHERE source = ? '
                'AND dedup_key = ? AND state = ?',
                (source, dedup_key, tc.STATE_OPEN))
            existing = cur.fetchone()
            if existing:
                conn.execute(
                    'UPDATE thalamus_items SET body = ?, refs = ?, '
                    'deliver_at = ?, expires_at = ?, updated_at = ? '
                    'WHERE id = ?',
                    (body, refs_json, deliver_at, expires_at, now,
                     existing[0]))
                conn.commit()
                return {'filed': True, 'id': existing[0], 'updated': True,
                        'route': 'queue'}

        cur = conn.execute(
            'SELECT COUNT(*) FROM thalamus_items WHERE source = ? '
            'AND state = ?', (source, tc.STATE_OPEN))
        open_count = cur.fetchone()[0]
        if open_count >= tc.MAX_OPEN_PER_SOURCE:
            return {'filed': False, 'error':
                    'thalamus budget: %r has %d open items (cap %d) — '
                    'resolve, update (same dedup_key), or withdraw before '
                    'filing new ones; thalamus_list shows them'
                    % (source, open_count, tc.MAX_OPEN_PER_SOURCE)}

        item_id = 'th_%s' % uuid.uuid4().hex[:8]
        conn.execute(
            'INSERT INTO thalamus_items (id, source, body, refs, audience,'
            ' target_session, needs_answer, dedup_key, deliver_at,'
            ' expires_at, state, answer, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (item_id, source, body, refs_json, audience, target_session,
             1 if needs_answer else 0, dedup_key or '', deliver_at,
             expires_at, tc.STATE_OPEN, '', now))
        conn.commit()
    return {'filed': True, 'id': item_id, 'route': 'queue',
            'audience': audience, 'deliver_at': deliver_at,
            'expires_at': expires_at}


def pull(brain, session_id, via):
    """Due items for this session at a delivery moment ('boot' | 'stop') —
    rendered block + count. Writes the ledger at render (INSERT OR IGNORE:
    the PK makes re-render idempotent per session). Asks (needs_answer)
    deliver at BOOT ONLY — an architecture question arriving mid-thread
    trains reflex-deferral; at boot there is no thread to protect.

    The caller owns tracing (the Stop hook holds the chain); at boot the
    ledger + boot_renders row are the record. Two sessions racing a
    once-item can each deliver it — rare and harmless (a reminder seen
    twice beats one lost); the ledger stays truthful about both.
    """
    if not session_id:
        return '', 0
    now = iso_now()
    params = [now, now, session_id]
    sql = ('SELECT %s FROM thalamus_items WHERE state = \'open\''
           ' AND (deliver_at IS NULL OR deliver_at <= ?)'
           ' AND expires_at > ?'
           ' AND (target_session = \'\' OR target_session = ?)'
           % _ITEM_COLS)
    if via != 'boot':
        sql += ' AND needs_answer = 0'
    sql += (' AND ((audience = \'once\' AND NOT EXISTS ('
            '   SELECT 1 FROM thalamus_deliveries d WHERE d.item_id = thalamus_items.id))'
            '  OR (audience = \'all\' AND NOT EXISTS ('
            '   SELECT 1 FROM thalamus_deliveries d WHERE d.item_id = thalamus_items.id'
            '   AND d.session_id = ?)))'
            ' ORDER BY needs_answer DESC, COALESCE(deliver_at, created_at) ASC'
            ' LIMIT ?')
    params += [session_id, tc.PULL_MAX_ITEMS + 1]
    rows = brain.logs_conn.execute(sql, params).fetchall()
    if not rows:
        return '', 0
    overflow = max(0, len(rows) - tc.PULL_MAX_ITEMS)
    items = [_row_to_item(r) for r in rows[:tc.PULL_MAX_ITEMS]]
    delivered_at = iso_now()
    with brain.logs_write_lock:
        brain.logs_conn_w.executemany(
            'INSERT OR IGNORE INTO thalamus_deliveries '
            '(item_id, session_id, delivered_at, via) VALUES (?, ?, ?, ?)',
            [(i['id'], session_id, delivered_at, via) for i in items])
        brain.logs_conn_w.commit()
    return tc.render_block(items, brain=brain, overflow=overflow), len(items)


def list_items(brain, include_closed=False, limit=50):
    """The pullable view — open items (default) with their delivery counts,
    newest first. include_closed adds terminal items for audit."""
    where = '' if include_closed else "WHERE state = 'open'"
    rows = brain.logs_conn.execute(
        'SELECT %s, (SELECT COUNT(*) FROM thalamus_deliveries d'
        '            WHERE d.item_id = thalamus_items.id) AS deliveries'
        ' FROM thalamus_items %s ORDER BY created_at DESC LIMIT ?'
        % (_ITEM_COLS, where), (int(limit),)).fetchall()
    items = []
    for r in rows:
        item = _row_to_item(r[:-1])
        item['deliveries'] = r[-1]
        items.append(item)
    return {'items': items, 'count': len(items)}


def _get_item(conn, item_id):
    row = conn.execute(
        'SELECT %s FROM thalamus_items WHERE id = ?' % _ITEM_COLS,
        (item_id,)).fetchone()
    return _row_to_item(row) if row else None


def resolve(brain, item_id, answer=None, defer_until=None, dismiss=False):
    """Anchor's exit verb — exactly one of answer / defer_until / dismiss.

    answer      → state 'answered'; the payload rides the item for the
                  producer's return surface (Phase 2 render-join).
    defer_until → deliver_at moves forward, the item RE-ARMS (its ledger rows
                  are cleared so it delivers again when due) and the window
                  stretches to cover the new date.
    dismiss     → state 'dismissed' (closed without an answer).
    """
    actions = [a for a in (answer, defer_until, True if dismiss else None)
               if a is not None]
    if len(actions) != 1:
        return {'ok': False, 'error':
                'thalamus_resolve: pass exactly one of answer / defer_until / '
                'dismiss'}
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        item = _get_item(conn, item_id)
        if not item:
            return {'ok': False, 'error': 'thalamus_resolve: no item %r'
                    % item_id}
        if item['state'] != tc.STATE_OPEN:
            return {'ok': False, 'error':
                    'thalamus_resolve: %s is already %r'
                    % (item_id, item['state'])}
        now = iso_now()
        if answer is not None:
            conn.execute(
                'UPDATE thalamus_items SET state = ?, answer = ?, '
                'answered_at = ?, updated_at = ? WHERE id = ?',
                (tc.STATE_ANSWERED, str(answer), now, now, item_id))
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_ANSWERED}
        elif defer_until is not None:
            try:
                new_deliver = tc.resolve_when(defer_until)
            except ValueError as e:
                return {'ok': False, 'error': str(e)}
            if new_deliver is None:
                new_deliver = iso_now()
            # Window must outlive the new due date by the grace period.
            from datetime import datetime as _dt, timedelta as _td
            new_expires = max(
                item['expires_at'] or '',
                (_dt.fromisoformat(new_deliver)
                 + _td(days=tc.REMIND_GRACE_DAYS)).isoformat())
            conn.execute(
                'UPDATE thalamus_items SET deliver_at = ?, expires_at = ?, '
                'updated_at = ? WHERE id = ?',
                (new_deliver, new_expires, now, item_id))
            conn.execute('DELETE FROM thalamus_deliveries WHERE item_id = ?',
                         (item_id,))
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_OPEN,
                      'deliver_at': new_deliver, 'expires_at': new_expires}
        else:
            conn.execute(
                'UPDATE thalamus_items SET state = ?, updated_at = ? '
                'WHERE id = ?', (tc.STATE_DISMISSED, now, item_id))
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_DISMISSED}
        conn.commit()
    return result


def withdraw(brain, source, item_id=None, dedup_key=None):
    """Producer retraction — a producer may close ITS OWN open item (by id or
    dedup_key). Without this, a condition that resolved itself waits as a
    stale ask: the wallpaper defect one level up. Source must match."""
    if not (item_id or dedup_key):
        return {'ok': False, 'error':
                'thalamus.withdraw: pass item_id or dedup_key'}
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        if item_id:
            row = conn.execute(
                'SELECT source, state FROM thalamus_items WHERE id = ?',
                (item_id,)).fetchone()
        else:
            row = conn.execute(
                'SELECT source, state, id FROM thalamus_items WHERE '
                'source = ? AND dedup_key = ? AND state = ?',
                (source, dedup_key, tc.STATE_OPEN)).fetchone()
            item_id = row[2] if row else None
        if not row:
            return {'ok': False, 'error': 'thalamus.withdraw: no such item'}
        if row[0] != source:
            return {'ok': False, 'error':
                    'thalamus.withdraw: %s belongs to %r, not %r — only the '
                    'filing producer may withdraw' % (item_id, row[0], source)}
        if row[1] != tc.STATE_OPEN:
            return {'ok': False, 'error': 'thalamus.withdraw: %s is already %r'
                    % (item_id, row[1])}
        conn.execute(
            'UPDATE thalamus_items SET state = ?, updated_at = ? WHERE id = ?',
            (tc.STATE_WITHDRAWN, iso_now(), item_id))
        conn.commit()
    return {'ok': True, 'id': item_id, 'state': tc.STATE_WITHDRAWN}


def expire_due(brain):
    """The window sweep (daemon idle maintenance). An expired NOTICE is a
    natural death; an expired ASK is the dead-letter case — logged LOUDLY to
    the errors table (which has a reader: query_logs) so an unanswered
    question never dissolves silently. Returns count expired."""
    now = iso_now()
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        dead_asks = conn.execute(
            'SELECT id, source, body FROM thalamus_items '
            "WHERE state = 'open' AND expires_at <= ? AND needs_answer = 1",
            (now,)).fetchall()
        cur = conn.execute(
            "UPDATE thalamus_items SET state = ?, updated_at = ? "
            "WHERE state = 'open' AND expires_at <= ?",
            (tc.STATE_EXPIRED, now, now))
        expired = cur.rowcount or 0
        conn.commit()
    for item_id, source, body in dead_asks:
        brain._log_error(
            'thalamus_ask_expired',
            RuntimeError('ask %s from %s expired unanswered: %.200s'
                         % (item_id, source, body)),
            'thalamus.expire_due — dead-letter (unanswered ask hit its window)')
    return expired

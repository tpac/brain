"""Thalamus mechanics — file / pull / resolve / withdraw / expire.

Every verb answers in ONE envelope — {'ok': bool, 'id': …} — so a producer
learns one result contract for the whole module ('filed' survives on file()
as a compatibility alias for one release).

One producer door (`file()`), three entrances: Anchor's MCP tools
(dispatch_thalamus.py), agent toolsets (Phase 2), direct code calls. Routing
lives inside the door, which validates and resolves the grammars and then
hands off to one storage branch: a live-now FYI delegates to the courier
broadcast and goes terminal immediately (`_file_live`); everything with a
clock, a window, or an answer queues (`_file_queued`). Delivery is PULL-ONLY
at the two moments that provably land — the Stop
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

from servers.clock import iso_now
from servers.channels.thalamus import thalamus_contract as tc


# The item's columns, declared ONCE as names. Everything else derives: the
# SELECT text, the INSERT text, and _row_to_item's mapping — which is BY NAME,
# so a column may be added anywhere in this tuple without silently shifting
# every field after it into the wrong key.
_ITEM_COLS = ('id', 'source', 'body', 'refs', 'audience', 'target_session',
              'needs_answer', 'dedup_key', 'deliver_at', 'expires_at',
              'state', 'answer', 'created_at', 'armed_epoch')

_ITEM_SELECT = ', '.join(_ITEM_COLS)

# The door supplies every column it writes; armed_epoch is the exception —
# it takes its DDL default, generation 0. Derived rather than re-listed so a
# new column reaches the INSERT too, where _insert_item's KeyError makes the
# author decide what to write instead of defaulting silently.
_INSERT_COLS = tuple(c for c in _ITEM_COLS if c != 'armed_epoch')


def _row_to_item(row):
    """A SELECT of _ITEM_COLS → item dict, mapped by NAME. Extra trailing
    columns (list_items' delivery counts) are ignored — zip stops at the
    names — and the caller attaches them itself."""
    item = dict(zip(_ITEM_COLS, row))
    item['refs'] = json.loads(item['refs'] or '[]')
    item['needs_answer'] = bool(item['needs_answer'])
    return item


def _ok(**fields):
    """The door's success envelope — `ok` + `id`, the shape resolve() and
    withdraw() already return, so a producer learns ONE result contract.
    `filed` is a compatibility alias kept for one release."""
    return {'ok': True, 'filed': True, **fields}


def _reject(error):
    """The door's LOUD synchronous rejection — same envelope, inverted."""
    return {'ok': False, 'filed': False, 'error': error}


def _insert_item(conn, **fields):
    """The one INSERT and the one id-minting site, shared by both routes.
    Every column is named by the caller: a missing field is a KeyError here,
    not a silent NULL in the row. The caller holds the write lock and
    commits."""
    item_id = 'th_%s' % uuid.uuid4().hex[:8]
    row = dict(fields, id=item_id, answer='')
    conn.execute(
        'INSERT INTO thalamus_items (%s) VALUES (%s)'
        % (', '.join(_INSERT_COLS), ', '.join('?' * len(_INSERT_COLS))),
        tuple(row[col] for col in _INSERT_COLS))
    return item_id


def file(brain, source, body, *, needs_answer=False, when=None, for_whom=None,
         refs=None, dedup_key=None, expires=None, session_id=''):
    """The single producer door: validate → resolve the grammars → route.

    Returns {'ok': True, 'id': …} or a LOUD synchronous rejection
    {'ok': False, 'error': <guidance>} — every guard fires HERE, at the write
    boundary, where the rejection lands in the caller's loop while it can
    still adapt (never in a sweeper hours later).

    Routing: for_whom 'live' delegates to the courier broadcast (requires a
    filing session — the locked stream-speech render must stay honest) and the
    row goes terminal 'sent'. Everything else queues for pull delivery.

    Identity: a repeat (source, dedup_key) UPDATES the open item instead of
    inserting — identity is producer-owned or absent, never derived from text.
    Both forms return updated=True; they differ in `rearmed`. A repeat that
    CHANGES any producer-controlled delivery attribute — body, refs, when,
    needs_answer, for_whom — rewrites the item to what this call describes
    and re-arms delivery (rearmed=True), so an edited item delivers again to
    sessions that already saw the old one. An identical repeat only refreshes
    the window (rearmed=False) — a cyclic producer re-asserting its standing
    item must never re-notify every session on every cycle.
    """
    body = (body or '').strip()
    if not body:
        return _reject('thalamus.file: empty body')
    # `source` is the budget key AND the withdraw-ownership key — a typo'd
    # one gets a fresh budget and orphans its own items, so it is vocabulary
    # (the repo-wide encoding_source grammar), not free text.
    from servers.contract import validate_encoding_source
    ok, err = validate_encoding_source(source)
    if not ok:
        return _reject('thalamus.file: %s' % err)

    try:
        route, audience, target_session = tc.resolve_for_whom(
            for_whom, needs_answer)
        deliver_at = tc.resolve_when(when)
        expires_at = (tc.resolve_when(expires) if expires
                      else tc.window_for(needs_answer, deliver_at))
    except ValueError as e:
        return _reject(str(e))
    if route == 'queue' and audience not in tc.AUDIENCES:
        # Internal drift guard at the write boundary: an audience outside the
        # closed set matches neither pull-predicate branch — the row would
        # stay open forever and die silently at expiry.
        return _reject('thalamus.file: internal audience %r is not one of %s'
                       % (audience, tc.AUDIENCES))
    if needs_answer and target_session:
        # Undeliverable by construction: asks render at BOOT only, boot fires
        # once per fresh session, and a UUID you can name has already had its
        # boot — the item would wait out its window and dead-letter, guaranteed.
        return _reject(
            'thalamus.file: a directed ask cannot deliver — asks render at '
            'boot only and the named session has already booted; use '
            'self_send to reach a live stream, or file for_whom=\'all\'')
    if expires_at and deliver_at and expires_at <= deliver_at:
        # window_for anchors the DEFAULT window at deliver_at exactly to
        # prevent expiry-before-due; an explicit `expires` must meet the same
        # bar or the item never becomes due (a false loud dead-letter for asks).
        return _reject(
            'thalamus.file: expires (%s) is not after when (%s) — the item '
            'would expire before it ever becomes due' % (expires_at, deliver_at))

    now = iso_now()
    refs_json = json.dumps(list(refs or []))

    if route == 'live':
        # A queue-shaped param on a live send is an intent the route cannot
        # honor — reject loudly rather than silently dropping it. Routing
        # validation stays at the door; the branch only stores.
        if deliver_at or needs_answer or dedup_key or expires:
            return _reject(
                "thalamus.file: for_whom='live' is fire-and-forget — it "
                "cannot honor when/needs_answer/dedup_key/expires; drop "
                "them, or queue instead (for_whom='all' or omit)")
        if not session_id:
            return _reject(
                "thalamus.file: for_whom='live' requires a filing session "
                "(the stream-speech render needs an honest sender); "
                "machine live-now is Phase 3")
        return _file_live(brain, source, body, refs, refs_json, session_id,
                          now)

    return _file_queued(brain, source, body, refs_json, now,
                        audience=audience, target_session=target_session,
                        needs_answer=needs_answer, dedup_key=dedup_key,
                        deliver_at=deliver_at, expires_at=expires_at)


def _file_live(brain, source, body, refs, refs_json, session_id, now):
    """Route 'live' — fire-and-forget to whoever is alive NOW, the courier's
    contract. The terminal 'sent' row is kept for observability
    (thalamus_list) and has no ledger lifecycle: the courier owns delivery and
    death (1h TTL). Phase 3's machine live-now rewrites exactly this."""
    from servers.channels.self_channel import signal, self_contract
    sent = signal.send(brain, from_session=session_id,
                       address=self_contract.ADDR_BROADCAST,
                       body=body, refs=refs)
    with brain.logs_write_lock:
        item_id = _insert_item(
            brain.logs_conn_w, source=source, body=body, refs=refs_json,
            audience='', target_session='', needs_answer=0, dedup_key='',
            deliver_at=None, expires_at=sent['expires_at'],
            state=tc.STATE_SENT, created_at=now)
        brain.logs_conn_w.commit()
    return _ok(id=item_id, route='live', courier_id=sent['id'])


def _file_queued(brain, source, body, refs_json, now, *, audience,
                 target_session, needs_answer, dedup_key, deliver_at,
                 expires_at):
    """Route 'queue' — the durable path, in order: dedup upsert, then budget,
    then insert. The door owns the identity contract; the epoch mechanics are
    below."""
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        if dedup_key:
            # Unbound one-step lookup + LIMIT 1 — the dal_logs.py write-conn
            # invariant: a bound, unexhausted cursor on logs_conn_w holds a
            # read snapshot and the next write fails INSTANTLY on a concurrent
            # commit (SQLITE_BUSY_SNAPSHOT, busy_timeout bypassed).
            existing = conn.execute(
                'SELECT id, body, refs, deliver_at, needs_answer, audience,'
                ' target_session FROM thalamus_items '
                'WHERE source = ? AND dedup_key = ? AND state = ? LIMIT 1',
                (source, dedup_key, tc.STATE_OPEN)).fetchone()
            if existing:
                # The gate compares EVERY producer-controlled delivery
                # attribute, and the UPDATE sets each one it compares: a
                # re-file leaves the row in the state the producer just
                # described, or it is a no-op. Comparing a subset silently
                # keeps stale attributes — a note re-filed as an ask kept
                # needs_answer=0 (still a note, still Stop-delivered, never
                # dead-lettering) while taking the ask's longer window.
                # A CHANGED re-file is a re-arm: bump the generation (the
                # same mechanism as resolve's defer) so an item already
                # delivered at the old epoch delivers again with the updated
                # content — without the bump the current-epoch ledger row
                # suppresses the update forever while file() reports success.
                # An UNCHANGED re-file (a cyclic producer re-asserting its
                # standing item) is idempotent: it refreshes the window and
                # nothing else — bumping on a no-op would re-deliver the same
                # text to every session on every producer cycle, unbounded.
                needs_answer_col = 1 if needs_answer else 0
                changed = tuple(existing[1:]) != (
                    body, refs_json, deliver_at, needs_answer_col, audience,
                    target_session)
                if changed:
                    conn.execute(
                        'UPDATE thalamus_items SET body = ?, refs = ?, '
                        'deliver_at = ?, expires_at = ?, needs_answer = ?, '
                        'audience = ?, target_session = ?, '
                        'armed_epoch = armed_epoch + 1, updated_at = ? '
                        'WHERE id = ?',
                        (body, refs_json, deliver_at, expires_at,
                         needs_answer_col, audience, target_session, now,
                         existing[0]))
                else:
                    conn.execute(
                        'UPDATE thalamus_items SET expires_at = ?, '
                        'updated_at = ? WHERE id = ?',
                        (expires_at, now, existing[0]))
                conn.commit()
                return _ok(id=existing[0], updated=True, rearmed=changed,
                           route='queue')

        # expires_at filter: expired-but-unswept items must not wedge a
        # producer at its cap while being invisible to delivery.
        open_count = conn.execute(
            'SELECT COUNT(*) FROM thalamus_items WHERE source = ? '
            'AND state = ? AND expires_at > ?',
            (source, tc.STATE_OPEN, now)).fetchone()[0]
        if open_count >= tc.MAX_OPEN_PER_SOURCE:
            return _reject(
                'thalamus budget: %r has %d open items (cap %d) — '
                'resolve, update (same dedup_key), or withdraw before '
                'filing new ones; thalamus_list shows them'
                % (source, open_count, tc.MAX_OPEN_PER_SOURCE))

        item_id = _insert_item(
            conn, source=source, body=body, refs=refs_json, audience=audience,
            target_session=target_session,
            needs_answer=1 if needs_answer else 0, dedup_key=dedup_key or '',
            deliver_at=deliver_at, expires_at=expires_at,
            state=tc.STATE_OPEN, created_at=now)
        conn.commit()
    return _ok(id=item_id, route='queue', audience=audience,
               deliver_at=deliver_at, expires_at=expires_at)


def _due_filter(session_id, via, now):
    """The due predicate — WHERE clause + params, shared by pull()'s fetch and
    its overflow count so the two cannot drift (the signal._PENDING_INBOX_SQL
    pattern). Asks (needs_answer) are due at tc.ASK_MOMENTS only."""
    sql = (' WHERE state = ?'
           ' AND (deliver_at IS NULL OR deliver_at <= ?)'
           ' AND expires_at > ?'
           " AND (target_session = '' OR target_session = ?)")
    params = [tc.STATE_OPEN, now, now, session_id]
    if via not in tc.ASK_MOMENTS:
        sql += ' AND needs_answer = 0'
    # The ledger is append-only across re-arms: only CURRENT-epoch rows
    # block delivery — a defer bumps the item's armed_epoch, so prior
    # generations' deliveries stay as history without suppressing the re-arm.
    sql += (' AND ((audience = ? AND NOT EXISTS ('
            '   SELECT 1 FROM thalamus_deliveries d WHERE d.item_id = thalamus_items.id'
            '   AND d.armed_epoch = thalamus_items.armed_epoch))'
            '  OR (audience = ? AND NOT EXISTS ('
            '   SELECT 1 FROM thalamus_deliveries d WHERE d.item_id = thalamus_items.id'
            '   AND d.armed_epoch = thalamus_items.armed_epoch'
            '   AND d.session_id = ?)))')
    params.extend([tc.AUDIENCE_FIRST, tc.AUDIENCE_EVERY, session_id])
    return sql, params


def _attach_ref_lines(brain, items, session_id):
    """Resolve every item's refs in ONE veil-aware batch and attach
    'ref_lines' for the render — the contract only FORMATS (its own rule);
    pull() holds both `brain` and `session_id`, so resolution lives here.
    Routes through filter_nodes (the existing veil door, default-deny): a
    globally-filed item can ref a walled node, and its title must not print
    into another session's boot — an unreturned ref (walled, retired, or a
    bad id) renders bare. A ref whose node was ABSORBED rides filter_nodes'
    canonical redirect: the survivor comes back stamped `_redirected_from`
    and renders as `old ↦ new · Title (absorbed)`.
    Failure-isolated and LOUD: refs left untitled by a
    failed batch render bare AND the failure is logged — bare ids are also
    the normal walled/bad-id output, so an unlogged dead resolution path
    would be invisible forever."""
    want = []
    for item in items:
        for ref in (item['refs'] or [])[:tc.RENDER_REFS_MAX]:
            if ref not in want:
                want.append(ref)
    titles = {}
    redirects = {}  # stored ref id -> (survivor id, survivor title)
    if want:
        try:
            res = brain.filter_nodes(
                field='id', include=want, rich=False,
                session_id=session_id, limit=len(want))
            from servers.contract import REDIRECTED_FROM_KEY
            for n in res.get('nodes') or []:
                titles[n['id']] = n.get('title') or ''
                # Canonical-pull redirect: a stored ref whose node was
                # absorbed resolves to its live survivor — render both ids
                # so the consumer knows the ref migrated.
                for src in n.get(REDIRECTED_FROM_KEY) or ():
                    redirects[src] = (n['id'], n.get('title') or '')
        except Exception as e:
            brain._log_error('thalamus_ref_resolve', e,
                             'pull ref batch failed — refs render bare')
    for item in items:
        refs = item['refs'] or []
        lines = []
        for ref in refs[:tc.RENDER_REFS_MAX]:
            if ref in redirects:
                from servers.contract import redirect_ref_line
                sid, title = redirects[ref]
                lines.append(redirect_ref_line(ref, sid, title))
                continue
            title = titles.get(ref)
            lines.append('%s · %s' % (ref[:8], title) if title else ref[:8])
        extra = len(refs) - tc.RENDER_REFS_MAX
        if extra > 0:
            lines.append('(+%d more refs — get_nodes)' % extra)
        item['ref_lines'] = lines


def pull(brain, session_id, via):
    """Due items for this session at a delivery moment ('boot' | 'stop') —
    rendered block + count of items actually shown. Writes the ledger at
    render for exactly those items (INSERT OR IGNORE: the PK makes re-render
    idempotent per session and epoch). Asks (needs_answer)
    deliver at BOOT ONLY — an architecture question arriving mid-thread
    trains reflex-deferral; at boot there is no thread to protect.

    The caller owns tracing (the Stop hook holds the chain); at boot the
    ledger + boot_renders row are the record. Two sessions racing a
    once-item can each deliver it — rare and harmless (a reminder seen
    twice beats one lost); the ledger stays truthful about both.
    """
    if via not in tc.MOMENTS:
        # via is written to the ledger verbatim — vocabulary, not free text.
        # A typo'd moment would behave as Stop (asks silently withheld) and
        # ledger the typo.
        raise ValueError('thalamus.pull: via=%r is not a delivery moment %s'
                         % (via, tc.MOMENTS))
    if not session_id:
        return '', 0
    now = iso_now()
    where, params = _due_filter(session_id, via, now)
    # True due count first, so the render's overflow tail names the real
    # number — a "+1" hiding fifteen is the wallpaper defect one level down.
    # Predicate shared with the item fetch below via _due_filter, so the two
    # cannot drift.
    total = brain.logs_conn.execute(
        'SELECT COUNT(*) FROM thalamus_items' + where, params).fetchone()[0]
    if not total:
        return '', 0
    rows = brain.logs_conn.execute(
        'SELECT %s FROM thalamus_items%s'
        ' ORDER BY needs_answer DESC, COALESCE(deliver_at, created_at) ASC'
        ' LIMIT ?' % (_ITEM_SELECT, where),
        params + [tc.PULL_MAX_ITEMS]).fetchall()
    overflow = total - len(rows)
    items = [_row_to_item(r) for r in rows]
    _attach_ref_lines(brain, items, session_id)
    # Render FIRST, then ledger only what the block actually shows — a
    # cap-dropped item was never delivered, and a ledger row would suppress
    # it forever; unledgered, it stays armed for the next moment.
    block, kept = tc.render_block(items, overflow=overflow)
    delivered_at = iso_now()
    with brain.logs_write_lock:
        brain.logs_conn_w.executemany(
            'INSERT OR IGNORE INTO thalamus_deliveries '
            '(item_id, session_id, delivered_at, via, armed_epoch) '
            'VALUES (?, ?, ?, ?, ?)',
            [(i['id'], session_id, delivered_at, via, i['armed_epoch'])
             for i in items[:kept]])
        brain.logs_conn_w.commit()
    return block, kept


def list_items(brain, include_closed=False, limit=50):
    """The pullable view — open items (default) with their delivery counts,
    newest first. include_closed adds terminal items for audit."""
    where, params = ('', []) if include_closed else ('WHERE state = ?',
                                                     [tc.STATE_OPEN])
    rows = brain.logs_conn.execute(
        'SELECT %s,'
        ' (SELECT COUNT(*) FROM thalamus_deliveries d'
        '  WHERE d.item_id = thalamus_items.id) AS deliveries,'
        ' (SELECT COUNT(*) FROM thalamus_deliveries d'
        '  WHERE d.item_id = thalamus_items.id'
        '  AND d.armed_epoch = thalamus_items.armed_epoch) AS deliveries_epoch'
        ' FROM thalamus_items %s ORDER BY created_at DESC LIMIT ?'
        % (_ITEM_SELECT, where), params + [int(limit)]).fetchall()
    items = []
    for r in rows:
        item = _row_to_item(r)            # the two aggregates fall off the zip
        item['deliveries'] = r[-2]        # all-time, across re-arms
        item['deliveries_epoch'] = r[-1]  # since the last re-arm
        items.append(item)
    return {'items': items, 'count': len(items)}


def _get_item(conn, item_id):
    row = conn.execute(
        'SELECT %s FROM thalamus_items WHERE id = ?' % _ITEM_SELECT,
        (item_id,)).fetchone()
    return _row_to_item(row) if row else None


def resolve(brain, item_id, answer=None, defer_until=None, dismiss=False):
    """Anchor's exit verb — exactly one of answer / defer_until / dismiss.

    answer      → state 'answered'; the payload rides the item for the
                  producer's return surface (Phase 2 render-join).
    defer_until → deliver_at moves forward and the item RE-ARMS as a new
                  generation (armed_epoch bump — prior deliveries stay in the
                  ledger as history but stop blocking); the window stretches
                  to cover the new date.
    dismiss     → state 'dismissed' (closed without an answer).
    """
    actions = [a for a in (answer, defer_until, True if dismiss else None)
               if a is not None]
    if len(actions) != 1:
        return {'ok': False, 'error':
                'thalamus_resolve: pass exactly one of answer / defer_until / '
                'dismiss'}
    if answer is not None and not str(answer).strip():
        # Same guard file() puts on the body — an ask must not close on an
        # empty payload; the producer's return surface would render nothing.
        return {'ok': False, 'error':
                'thalamus_resolve: empty answer — pass real text, or dismiss'}
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
            new_expires = tc.extend_window(item['needs_answer'], new_deliver,
                                           item['expires_at'])
            # Re-arm is a GENERATION, not a deletion: bumping armed_epoch
            # makes the pull predicate ignore prior-epoch ledger rows, so the
            # item delivers again when due — while the ledger keeps truthful
            # history ("delivered, then deferred" ≠ "never delivered";
            # Phase 3 retry gates on unacked).
            conn.execute(
                'UPDATE thalamus_items SET deliver_at = ?, expires_at = ?, '
                'armed_epoch = armed_epoch + 1, updated_at = ? WHERE id = ?',
                (new_deliver, new_expires, now, item_id))
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
    """The window sweep (Brain.sweep_channels_if_due, hourly, ahead of the
    S2 fire gate). An expired NOTICE is a
    natural death; an expired ASK is the dead-letter case — logged LOUDLY to
    the errors table (which has a reader: query_logs) so an unanswered
    question never dissolves silently. Returns count expired."""
    now = iso_now()
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        dead_asks = conn.execute(
            'SELECT id, source, body FROM thalamus_items '
            'WHERE state = ? AND expires_at <= ? AND needs_answer = 1',
            (tc.STATE_OPEN, now)).fetchall()
        cur = conn.execute(
            'UPDATE thalamus_items SET state = ?, updated_at = ? '
            'WHERE state = ? AND expires_at <= ?',
            (tc.STATE_EXPIRED, now, tc.STATE_OPEN, now))
        expired = cur.rowcount or 0
        conn.commit()
    for item_id, source, body in dead_asks:
        brain._log_error(
            'thalamus_ask_expired',
            RuntimeError('ask %s from %s expired unanswered: %.200s'
                         % (item_id, source, body)),
            'thalamus.expire_due — dead-letter (unanswered ask hit its window)')
    return expired

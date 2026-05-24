#!/usr/bin/env python3
"""Backfill identity metadata on historical trace_events (one-off).

After 2026-05-23, new trace_events carry human_identity / agent_identity
in their metadata JSON (TraceDAL.set_identity stamps every write). All
trace_events written before that have no identity in metadata.

This migration stamps every existing trace_events row with
  human_identity = 'Tom'
  agent_identity = 'Anchor'
based on the architectural assumption (Tom, 2026-05-23): for this
brain's current cast, every user_message is Tom and every agent
event is Anchor. That assumption is true for every historical trace
because no other partner has ever interacted with this brain.

Side effect: pre-fix tool_result rows that stored double-encoded JSON
(the dispatch-decode bug fixed in commit 65bf483) get re-encoded to
clean single-layer JSON.

Idempotent — re-running is a no-op for rows already carrying both
identity keys.

Daemon coordination: REQUIRES daemon DOWN. Migration touches every
trace_events row; concurrent writes during migration would race with
the rewrites. Run via:

    touch /tmp/brain-maintenance-$(id -u).lock
    launchctl unload ~/Library/LaunchAgents/com.brain.daemon.plist
    ./dev python3 scripts/migrate_trace_identity.py
    launchctl load ~/Library/LaunchAgents/com.brain.daemon.plist
    rm /tmp/brain-maintenance-$(id -u).lock

After migration, trace_embeddings is wiped so the embed_queue worker
re-embeds with the new (concrete-identity) render templates. Without
the wipe, traces embedded under sentinel labels would stay stale.
"""

import json
import os
import sqlite3
import sys
import time

OPERATOR = 'Tom'
AGENT = 'Anchor'
BATCH_SIZE = 1000
DEFAULT_DB = os.path.expanduser('~/AgentsContext/brain/brain_logs.db')


def decode_metadata(raw):
    """Defensive decode handling single-encoded, double-encoded, and
    garbage. Returns (meta_dict, was_double_encoded).
      - meta_dict is None when the cell can't be parsed at all
      - was_double_encoded is True only when json.loads(raw) yields a
        string that itself parses as JSON
    """
    if not raw:
        return ({}, False)
    try:
        first = json.loads(raw)
    except Exception:
        return (None, False)
    if isinstance(first, str):
        try:
            second = json.loads(first)
        except Exception:
            return (None, False)
        if isinstance(second, dict):
            return (second, True)
        return (None, False)
    if isinstance(first, dict):
        return (first, False)
    return (None, False)


def needs_stamping(meta):
    """True when the dict is missing either identity key."""
    return ('human_identity' not in meta) or ('agent_identity' not in meta)


def main(db_path):
    print('Opening %s' % db_path)
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')

    total = conn.execute('SELECT COUNT(*) FROM trace_events').fetchone()[0]
    print('Total trace_events: %d' % total)

    # SQL pre-filter to avoid pulling rows that are obviously already
    # stamped (LIKE on the JSON text is cheap; we re-verify per-row).
    cur = conn.execute(
        "SELECT id, metadata FROM trace_events "
        "WHERE metadata IS NULL "
        "   OR NOT (metadata LIKE '%human_identity%' "
        "          AND metadata LIKE '%agent_identity%') "
        "ORDER BY id ASC")

    updated = 0
    double_decoded_fixed = 0
    already_ok = 0
    skipped_unparseable = 0
    batch = []
    t0 = time.time()

    for row_id, raw in cur.fetchall():
        meta, was_double = decode_metadata(raw)
        if meta is None:
            skipped_unparseable += 1
            continue
        # Final correctness check — the LIKE pre-filter is approximate
        if not needs_stamping(meta) and not was_double:
            already_ok += 1
            continue
        meta.setdefault('human_identity', OPERATOR)
        meta.setdefault('agent_identity', AGENT)
        if was_double:
            double_decoded_fixed += 1
        new_raw = json.dumps(meta)
        batch.append((new_raw, row_id))
        if len(batch) >= BATCH_SIZE:
            conn.executemany(
                "UPDATE trace_events SET metadata = ? WHERE id = ?", batch)
            conn.commit()
            updated += len(batch)
            print('  ... updated %d (took %.1fs so far)' %
                  (updated, time.time() - t0))
            batch = []

    if batch:
        conn.executemany(
            "UPDATE trace_events SET metadata = ? WHERE id = ?", batch)
        conn.commit()
        updated += len(batch)

    elapsed = time.time() - t0
    print()
    print('Summary:')
    print('  Total trace_events:           %d' % total)
    print('  Updated (identity added):     %d' % updated)
    print('     of which double-decoded:  %d' % double_decoded_fixed)
    print('  Skipped (already stamped):    %d' % already_ok)
    print('  Skipped (unparseable):        %d' % skipped_unparseable)
    print('  Elapsed:                      %.1fs' % elapsed)

    # Wipe trace_embeddings so the worker re-embeds against the new
    # concrete-identity render. Without this, embeddings produced under
    # the OPERATOR/ANCHOR sentinel render would stay stale in the
    # neighborhood. Cheap to rebuild — worker drains 5/tick.
    embed_count = conn.execute(
        'SELECT COUNT(*) FROM trace_embeddings').fetchone()[0]
    print()
    print('trace_embeddings before wipe: %d' % embed_count)
    conn.execute('DELETE FROM trace_embeddings')
    conn.commit()
    after = conn.execute(
        'SELECT COUNT(*) FROM trace_embeddings').fetchone()[0]
    print('trace_embeddings after wipe:  %d' % after)
    print()
    print('Done. Restart daemon to resume; embed_queue will repopulate.')
    conn.close()


if __name__ == '__main__':
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    main(db)

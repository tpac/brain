#!/usr/bin/env python3
"""One-shot: re-embed all active, non-noise edge_relations after dropping the
aspect-family `meaning` from compose_edge_text.

Existing edge embeddings were computed from "[rel] desc family: <meaning>".
Recall compares query<->edge cosine in a single pass, so a mix of meaning-laden
and meaning-free vectors is INCONSISTENT geometry. This recomputes every active,
non-noise edge embedding with the new "[rel] desc" composer.

Run with the daemon STOPPED — bulk edge writes must not race the daemon's
writers (same discipline as scripts/scrub_archived_edge_embeddings.py):

    touch /tmp/brain-maintenance-$(id -u).lock
    launchctl unload ~/Library/LaunchAgents/com.brain.daemon.plist
    ./dev python3 scripts/reembed_edges_drop_meaning.py
    rm /tmp/brain-maintenance-$(id -u).lock
    launchctl load ~/Library/LaunchAgents/com.brain.daemon.plist

Idempotent: NULLs stale embeddings, then re-embeds in chunks (batched) until no
active non-noise row is NULL. Re-running after completion is a near no-op.
"""
from __future__ import annotations
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tie the noise set to the contract (matches backfill_edge_embeddings' own
# filter) rather than re-typing it — see CLAUDE.md "Contract-first".
from servers.dal import DEFAULT_EXCLUDED_RELATIONS  # noqa: E402
NOISE = tuple(sorted(DEFAULT_EXCLUDED_RELATIONS))
CHUNK = 800


def _count(brain, where):
    ph = ','.join('?' * len(NOISE))
    return brain.conn.execute(
        "SELECT COUNT(*) FROM edge_relations "
        "WHERE archived=0 AND relation NOT IN (%s) %s" % (ph, where),
        NOISE).fetchone()[0]


def main():
    from servers.brain import Brain
    from servers import embedder

    db_path = os.path.join(
        os.environ.get('BRAIN_DB_DIR') or os.path.expanduser('~/AgentsContext/brain'),
        'brain.db')
    brain = Brain.get_instance(db_path)
    if not embedder.is_ready():
        embedder.embed_batch(['warm'], kind='document')

    # Backup before the destructive NULL pass (CLAUDE.md: "Backup before
    # destructive DB operations ... No exceptions."). WAL-safe snapshot via the
    # same path the daemon's scheduler uses; cheap insurance for re-runs.
    from servers import db_backup
    backup = db_backup.backup_database(
        db_path, os.path.join(os.path.dirname(db_path), 'backups'))
    print('backup: %s (%d bytes gz)' % (backup.get('dest'), backup.get('gz_bytes', 0)),
          flush=True)

    ph = ','.join('?' * len(NOISE))
    n_active = _count(brain, '')
    n_embedded = _count(brain, 'AND embedding IS NOT NULL')
    print('active non-noise relations: %d (embedded before: %d)' % (n_active, n_embedded),
          flush=True)

    # 1. Invalidate the stale (meaning-laden) embeddings.
    t0 = time.monotonic()
    with brain.write_lock:
        cur = brain.conn.execute(
            "UPDATE edge_relations SET embedding=NULL, embedding_model=NULL "
            "WHERE archived=0 AND relation NOT IN (%s) AND embedding IS NOT NULL" % ph,
            NOISE)
        nulled = cur.rowcount
        brain.conn.commit()
    print('NULLed %d stale embeddings in %dms' % (nulled, (time.monotonic() - t0) * 1000),
          flush=True)

    # 2. Re-embed in chunks (backfill_edge_embeddings batches the embed call and
    #    locks only the write). Self-draining: written rows leave the NULL set.
    t1 = time.monotonic()
    total = rounds = 0
    while True:
        ids = [r[0] for r in brain.conn.execute(
            "SELECT DISTINCT edge_id FROM edge_relations "
            "WHERE archived=0 AND relation NOT IN (%s) AND embedding IS NULL "
            "LIMIT %d" % (ph, CHUNK), NOISE).fetchall()]
        if not ids:
            break
        wrote = brain.backfill_edge_embeddings(ids)
        total += wrote
        rounds += 1
        print('  round %d: +%d (total %d)' % (rounds, wrote, total), flush=True)
        if wrote == 0:
            remaining = _count(brain, 'AND embedding IS NULL')
            print('  no progress; %d rows remain NULL (empty-text edges?) — stopping'
                  % remaining, flush=True)
            break

    final_embedded = _count(brain, 'AND embedding IS NOT NULL')
    still_null = _count(brain, 'AND embedding IS NULL')
    print('DONE in %.1fs: re-embedded %d; %d/%d active non-noise embedded, %d NULL'
          % (time.monotonic() - t1, total, final_embedded, n_active, still_null), flush=True)


if __name__ == '__main__':
    main()

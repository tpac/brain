"""Scrub stale embedding blobs from already-archived edge_relations rows.

One-shot: run once after the v26-archive-symmetry fix lands to clear
embeddings that were written before archive paths started NULLing them.

Why: pre-fix, three archive paths (`delete_node_edges`, `decay_edges`,
`remove_relation`) flipped `archived = 1` but left the embedding blob
in place. With ~3 KB per archived row, that's wasted storage that's
never read (every recall path filters `archived = 0`). The fix updated
those paths going forward; this script cleans up the historical state.

Idempotent: scrubs only `archived = 1 AND embedding IS NOT NULL` rows.
Re-running is a no-op once the table is clean.

Run with the daemon stopped (or under maintenance lock) to avoid
racing the daemon's writers:

    touch /tmp/brain-maintenance-$(id -u).lock
    launchctl unload ~/Library/LaunchAgents/com.brain.daemon.plist
    ./dev python3 scripts/scrub_archived_edge_embeddings.py
    launchctl load ~/Library/LaunchAgents/com.brain.daemon.plist
    rm /tmp/brain-maintenance-$(id -u).lock

Args:
    --dry-run    Report counts but don't UPDATE.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Report counts but do not UPDATE')
    args = parser.parse_args()

    from servers.brain import Brain
    db_path = os.path.join(
        os.environ.get('BRAIN_DB_DIR') or
        os.path.expanduser('~/AgentsContext/brain'),
        'brain.db')
    brain = Brain.get_instance(db_path)

    # Count what we'd scrub
    n_dirty = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations '
        'WHERE archived = 1 AND embedding IS NOT NULL'
    ).fetchone()[0]
    n_archived = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations WHERE archived = 1'
    ).fetchone()[0]
    n_active = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations WHERE archived = 0'
    ).fetchone()[0]
    print(f'edge_relations: {n_active} active, {n_archived} archived '
          f'({n_dirty} archived rows still hold an embedding)')
    if n_dirty == 0:
        print('Nothing to scrub.')
        return

    if args.dry_run:
        print(f'[dry run] would NULL embedding + embedding_model on '
              f'{n_dirty} archived rows.')
        return

    t0 = time.monotonic()
    with brain.write_lock:
        cur = brain.conn.execute(
            'UPDATE edge_relations '
            'SET embedding = NULL, embedding_model = NULL '
            'WHERE archived = 1 AND embedding IS NOT NULL')
        scrubbed = cur.rowcount
        brain.conn.commit()
    elapsed = time.monotonic() - t0
    print(f'Scrubbed {scrubbed} rows in {elapsed*1000:.0f}ms.')

    # Sanity-check we didn't touch active rows.
    n_active_after = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations '
        'WHERE archived = 0 AND embedding IS NOT NULL'
    ).fetchone()[0]
    print(f'Active rows with embedding (should match pre-scrub): '
          f'{n_active_after} / {n_active}')


if __name__ == '__main__':
    main()

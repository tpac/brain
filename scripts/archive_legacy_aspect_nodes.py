#!/usr/bin/env python3
"""One-shot: archive legacy `type='aspect'` nodes after JSON-source migration.

Context: 2026-05-08 migration moved AspectRegistry from reading brain
aspect-nodes to reading aspects_v1.json directly. The 60 brain aspect-nodes
(14 required + 46 emergent) became orphaned data — nothing reads them
anymore. This script archives them (soft-delete via archived=1) so they
stop showing up in recall/Frame/listings.

REVERSIBLE — archived nodes can be unarchived if anything goes wrong.

REQUIREMENTS BEFORE RUNNING:
  1. Backup brain.db (cp brain.db brain.db.bak-{ts})
  2. Maintenance lock set: touch /tmp/brain-maintenance-$(id -u).lock
  3. Daemon stopped: launchctl unload ~/Library/LaunchAgents/com.brain.daemon.plist

The script verifies #2 and refuses to run without the lock.
"""

import os
import sqlite3
import sys
from datetime import datetime, timezone


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/tpac/AgentsContext/brain/brain.db'
    uid = os.getuid()
    lock_path = '/tmp/brain-maintenance-%d.lock' % uid

    if not os.path.exists(lock_path):
        print('ERROR: maintenance lock not set at %s' % lock_path, file=sys.stderr)
        print('Run: touch %s' % lock_path, file=sys.stderr)
        return 1

    if not os.path.exists(db_path):
        print('ERROR: brain.db not found at %s' % db_path, file=sys.stderr)
        return 1

    print('Connecting to %s' % db_path)
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')

    # Find all live aspect nodes
    rows = conn.execute(
        "SELECT id, title, locked FROM nodes WHERE archived = 0 AND type = 'aspect'"
    ).fetchall()
    print('Found %d live aspect nodes to archive' % len(rows))
    if not rows:
        print('Nothing to do.')
        conn.close()
        return 0

    # Show what we're about to do
    locked_count = sum(1 for r in rows if r[2])
    print('  - %d locked (required aspects)' % locked_count)
    print('  - %d unlocked (emergent aspects)' % (len(rows) - locked_count))
    print('  Sample titles: %s' % ', '.join(r[1] for r in rows[:5]))

    ts = datetime.now(timezone.utc).isoformat()
    aspect_ids = [r[0] for r in rows]

    try:
        # Archive node rows
        conn.execute("""
            UPDATE nodes
            SET archived = 1, updated_at = ?
            WHERE id IN (%s)
        """ % ','.join('?' * len(aspect_ids)), [ts] + aspect_ids)
        n_nodes = conn.total_changes
        print('Archived %d node rows' % n_nodes)

        # Audit metadata via node_metadata_kv (mirrors archive_node behavior)
        for nid in aspect_ids:
            for key, value in [
                ('_sys_archived_at', ts),
                ('_sys_archived_by', 'migration:json_source_2026-05-08'),
                ('_sys_archived_reason',
                 'AspectRegistry migrated to read aspects_v1.json directly; '
                 'brain aspect-nodes are no longer the source of truth'),
            ]:
                conn.execute("""
                    INSERT INTO node_metadata_kv (node_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(node_id, key) DO UPDATE SET value = excluded.value
                """, (nid, key, value))

        # Archive edge_relations that touch these nodes. edge_relations is
        # keyed by (edge_id, relation); the connecting edge_id comes from
        # the edges table where source/target match an aspect node.
        cur = conn.execute("""
            UPDATE edge_relations
            SET archived = 1, archived_at = ?, archived_by = ?
            WHERE archived = 0
              AND edge_id IN (
                SELECT edge_id FROM edges
                WHERE source_id IN (%s) OR target_id IN (%s)
              )
        """ % (','.join('?' * len(aspect_ids)), ','.join('?' * len(aspect_ids))),
            [ts, 'migration:json_source_2026-05-08'] + aspect_ids + aspect_ids)
        print('Archived %d edge_relation rows' % cur.rowcount)

        conn.commit()
        print('Committed.')
    except Exception as e:
        conn.rollback()
        print('ERROR: %s — rolled back' % e, file=sys.stderr)
        conn.close()
        return 1

    # Verify
    remaining = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0 AND type = 'aspect'"
    ).fetchone()[0]
    archived_total = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 1 AND type = 'aspect'"
    ).fetchone()[0]
    print('Verification: %d live aspects remaining, %d archived total'
          % (remaining, archived_total))

    conn.close()

    if remaining > 0:
        print('WARNING: %d aspect nodes still live — partial archive' % remaining,
              file=sys.stderr)
        return 1

    print('\nDone. To restart daemon:')
    print('  rm %s' % lock_path)
    print('  launchctl load ~/Library/LaunchAgents/com.brain.daemon.plist')
    return 0


if __name__ == '__main__':
    sys.exit(main())

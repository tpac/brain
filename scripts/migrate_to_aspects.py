#!/usr/bin/env python3
"""Migrate the live brain to the unified aspects system — one-shot.

What it does:
  1. Sends a 'save' command to the daemon (flushes WAL).
  2. Backs up brain.db to brain.db.bak-{timestamp}.
  3. Sends 'migrate_to_aspects' to the daemon, which:
     - Seeds the 14 required aspect-nodes from aspects_v1.json (locked=True
       via encoding_source='anchor:seed_aspects').
     - Imports emergent (non-required) families from existing
       s2_node_families + s2_edge_families interactions as unlocked
       aspect-nodes.
  4. Prints the result.

Idempotent. Safe to re-run — already-existing aspect-nodes are skipped.

Usage:
    ./dev python3 scripts/migrate_to_aspects.py
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.daemon_client import send_command


def _resolve_db_dir() -> str:
    """Find the brain DB directory — env var or standard $HOME/AgentsContext/brain."""
    db_dir = os.environ.get('BRAIN_DB_DIR')
    if db_dir:
        return db_dir
    return os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')


def main():
    db_dir = _resolve_db_dir()
    brain_db = os.path.join(db_dir, 'brain.db')

    if not os.path.isfile(brain_db):
        print('ERROR: brain.db not found at %s' % brain_db, file=sys.stderr)
        sys.exit(1)

    # 1. Flush WAL so the backup is consistent
    print('[migrate] Flushing WAL via daemon save command...')
    save_resp = send_command('save', {})
    if not save_resp.get('ok'):
        print('ERROR: save command failed: %s' % save_resp.get('error'),
              file=sys.stderr)
        sys.exit(1)

    # 2. Backup
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = '%s.bak-%s' % (brain_db, ts)
    print('[migrate] Backing up brain.db to %s...' % backup_path)
    try:
        shutil.copy2(brain_db, backup_path)
    except Exception as e:
        print('ERROR: backup failed: %s' % e, file=sys.stderr)
        sys.exit(1)

    backup_size = os.path.getsize(backup_path)
    print('[migrate]   backup size: %d bytes' % backup_size)

    # 3. Run migration
    print('[migrate] Running migrate_to_aspects via daemon...')
    resp = send_command('migrate_to_aspects', {}, timeout=60.0)
    if not resp.get('ok'):
        print('ERROR: migration failed: %s' % resp.get('error'), file=sys.stderr)
        print('       Backup preserved at: %s' % backup_path, file=sys.stderr)
        sys.exit(1)

    result = resp.get('result', {})

    # 4. Print summary
    print()
    print('=== Migration complete ===')
    print()
    print(json.dumps(result, indent=2))
    print()
    print('Backup: %s' % backup_path)
    print('Total aspect-nodes in brain: %d' % result.get('aspect_node_count', 0))


if __name__ == '__main__':
    main()

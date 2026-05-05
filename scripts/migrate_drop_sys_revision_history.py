#!/usr/bin/env python3
"""Stage 1A migration: drop legacy _sys_revision_history KV blobs.

What it does:
  1. Counts nodes with the _sys_revision_history key (dry-run by default).
  2. With --commit: flushes WAL, backs up brain.db, deletes the keys.

Why:
  Stage 1A moved node revision history from a per-node JSON blob (capped at
  5 entries, content-only) to standard trace events. The legacy KV blob is
  no longer written by revise() (see brain_remember.py) and is never read
  by anything (audit confirmed). This script removes the existing keys.

  Per Tom's direction: drop the data, no retroactive trace conversion.

Usage:
    ./dev python3 scripts/migrate_drop_sys_revision_history.py            # dry-run
    ./dev python3 scripts/migrate_drop_sys_revision_history.py --commit   # delete

Safety:
  - Backs up brain.db before any write
  - Uses daemon dispatch (no second Brain() spawn)
  - Idempotent — safe to re-run
"""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--commit', action='store_true',
                        help='Actually delete (default: dry-run reports count)')
    args = parser.parse_args()

    db_dir = _resolve_db_dir()
    brain_db = os.path.join(db_dir, 'brain.db')

    if not os.path.isfile(brain_db):
        print('ERROR: brain.db not found at %s' % brain_db, file=sys.stderr)
        sys.exit(1)

    # Step 1: Count without deleting (always do this first)
    print('[migrate] Counting _sys_revision_history keys...')
    resp = send_command('drop_sys_revision_history', {'commit': False}, timeout=30.0)
    if not resp.get('ok'):
        print('ERROR: count failed: %s' % resp.get('error'), file=sys.stderr)
        sys.exit(1)

    count = resp['result']['count_found']
    print('[migrate]   nodes with _sys_revision_history: %d' % count)

    if count == 0:
        print('[migrate] Nothing to do — already clean.')
        return

    if not args.commit:
        print()
        print('Dry run complete. Re-run with --commit to delete.')
        return

    # Step 2: Backup before delete (writes need backup per CLAUDE.md discipline)
    print('[migrate] Flushing WAL via daemon save command...')
    save_resp = send_command('save', {})
    if not save_resp.get('ok'):
        print('ERROR: save failed: %s' % save_resp.get('error'), file=sys.stderr)
        sys.exit(1)

    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = '%s.bak-pre-stage1a-%s' % (brain_db, ts)
    print('[migrate] Backing up brain.db to %s...' % backup_path)
    try:
        shutil.copy2(brain_db, backup_path)
    except Exception as e:
        print('ERROR: backup failed: %s' % e, file=sys.stderr)
        sys.exit(1)

    backup_size = os.path.getsize(backup_path)
    print('[migrate]   backup size: %d bytes' % backup_size)

    # Step 3: Delete via daemon dispatch
    print('[migrate] Deleting _sys_revision_history keys...')
    resp = send_command('drop_sys_revision_history', {'commit': True}, timeout=60.0)
    if not resp.get('ok'):
        print('ERROR: delete failed: %s' % resp.get('error'), file=sys.stderr)
        print('       Backup preserved at: %s' % backup_path, file=sys.stderr)
        sys.exit(1)

    result = resp['result']

    # Step 4: Summary
    print()
    print('=== Migration complete ===')
    print()
    print(json.dumps(result, indent=2))
    print()
    print('Backup: %s' % backup_path)


if __name__ == '__main__':
    main()

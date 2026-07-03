#!/usr/bin/env python3
"""One-off migration: nodes.project column → node_metadata_kv['project'].

Project became system-stamped kv provenance on 2026-07-03 (contract
PROMOTED_FIELDS entry; stamped at the write boundaries, read by the LAF proj
lane and dict filters). This moves the legacy column values into kv with the
canonical slug mapping, then NULLs the column. The column itself is dropped
at the next schema bump.

Mapping (approved by the operator, 2026-07-03): the historical values are
brain-repo work under topical costume names ('S1Scribe', 'aspects_refactor',
'brain-daemon', ...) → 'brain', except the EX.CO trio → 'ex.co'. Any value
this script has never seen still maps to 'brain' but is listed loudly first —
run --dry-run to review before writing.

MUST run with the daemon STOPPED (two writers corrupt indexes):
    touch /tmp/brain-maintenance-$(id -u).lock
    launchctl bootout gui/$(id -u)/com.brain.daemon
    ./dev python3 -m servers.tools.migrate_project_to_kv --dry-run
    ./dev python3 -m servers.tools.migrate_project_to_kv
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.brain.daemon.plist
    rm /tmp/brain-maintenance-$(id -u).lock

Idempotent: already-migrated rows (column NULL) are skipped; re-running after
a partial failure completes the remainder. Makes its own timestamped backup
before writing.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from servers import db_backends                                   # noqa: E402
from servers.dal_metadata import MetadataDAL                      # noqa: E402

# ── the approved slug mapping ──
EXCO_VALUES = {'EX.CO CTV kit', 'ex.co', 'CTVOnboarding'}

# The 41 distinct values in the inventory the operator approved (2026-07-03).
# A value NOT in this set appeared after approval — mapped to 'brain' like the
# rest, but flagged loudly in the summary so it gets eyeballs before commit.
APPROVED_INVENTORY = EXCO_VALUES | {
    'brain', 'S1Scribe', 'dashboard', 'S2Aspect', 'awareness', 'brain-daemon',
    'aspects_refactor', 'anchor-brain', 'Anchor', 'EPISODIC-REFERENCES', 'S2',
    's2_community', 'S1S', 'S2CD', 'brain-dashboard', 'brain-s2',
    'fractal-brain', 's1scribe', 'S2 Community', 'benchmark', 'brain-recall',
    'S2CE', 'brain-eval', 'embedding', 'Frame', 'S1S prompt rewrite', 'boot',
    'brain-encoding', 'trace-unification', 'brain-architecture',
    'brain-cleanup', 'brain-core', 'brain-encoding-prompt-v3', 'daemon',
    'encoding prompt v3', 'encoding-prompt', 'graph_architecture',
    'longmem-eval',
}


def canonical_slug(legacy: str) -> str:
    return 'ex.co' if legacy in EXCO_VALUES else 'brain'


def db_path() -> str:
    d = os.environ.get('BRAIN_DB_DIR') or os.path.expanduser(
        '~/AgentsContext/brain')
    return os.path.join(d, 'brain.db')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='print the mapping summary, write nothing')
    args = ap.parse_args()

    path = db_path()
    if not os.path.exists(path):
        print('FATAL: %s not found' % path)
        return 1

    import sqlite3
    conn = sqlite3.connect(path)
    db_backends.current.apply_pragmas(conn)

    rows = conn.execute(
        "SELECT id, project FROM nodes "
        "WHERE project IS NOT NULL AND project != ''").fetchall()
    if not rows:
        print('nothing to migrate — column already clean')
        return 0

    # Summary, unexpected values first (loud before any write)
    by_value = {}
    for _, legacy in rows:
        by_value[legacy] = by_value.get(legacy, 0) + 1
    print('%d nodes carry a legacy project value (%d distinct):'
          % (len(rows), len(by_value)))
    for legacy, count in sorted(by_value.items(),
                                key=lambda kv: (kv[0] in APPROVED_INVENTORY,
                                                -kv[1])):
        tag = ('' if legacy in APPROVED_INVENTORY
               else '   <-- NOT in the approved inventory, maps to brain')
        print('  %-28r x%-4d -> %r%s'
              % (legacy, count, canonical_slug(legacy), tag))

    if args.dry_run:
        print('dry run — nothing written')
        return 0

    backup = '%s.bak-%s' % (path, time.strftime('%Y%m%d-%H%M%S'))
    shutil.copyfile(path, backup)
    print('backup: %s' % backup)

    mdal = MetadataDAL(conn)
    conn.execute('BEGIN IMMEDIATE')
    try:
        for node_id, legacy in rows:
            mdal.set_many(node_id, {'project': canonical_slug(legacy)})
        conn.execute(
            "UPDATE nodes SET project = NULL "
            "WHERE project IS NOT NULL AND project != ''")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    kv_count = conn.execute(
        "SELECT COUNT(*) FROM node_metadata_kv WHERE key = 'project'"
    ).fetchone()[0]
    col_count = conn.execute(
        "SELECT COUNT(*) FROM nodes "
        "WHERE project IS NOT NULL AND project != ''").fetchone()[0]
    print('migrated %d nodes -> kv (kv project rows now: %d; '
          'column non-null remaining: %d)' % (len(rows), kv_count, col_count))
    return 0 if col_count == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())

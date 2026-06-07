"""Backfill `edge_relations.embedding` for the live brain (schema v26+).

Why: schema v26 added `embedding` BLOB + `embedding_model` TEXT columns to
`edge_relations` so that surface_spread reads stored vectors instead of
running fastembed at recall time. Existing rows have NULL until backfilled.
This script populates them in batches, idempotent and resumable.

Workflow:
  1. Read all (edge_id, relation, description) where embedding IS NULL
  2. Compose enriched text via brain.aspects.compose_edge_text(rel, desc)
  3. Embed in fastembed batches of 256
  4. UPDATE in transactions of 500 rows
  5. Print progress + final stats

Run with the daemon STOPPED, or via the maintenance lock so writes don't
race the daemon's own writers:

    touch /tmp/brain-maintenance-$(id -u).lock
    launchctl unload ~/Library/LaunchAgents/com.brain.daemon.plist
    ./dev python3 scripts/backfill_edge_embeddings.py
    launchctl load ~/Library/LaunchAgents/com.brain.daemon.plist
    rm /tmp/brain-maintenance-$(id -u).lock

Idempotent: running twice is safe (only NULL rows are touched). Limit
arg useful for dev/test runs.

Args:
    --limit N        Backfill at most N rows (default: all NULL)
    --dry-run        Compute but don't write
"""
from __future__ import annotations
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.dispatch import load_env  # noqa: E402
load_env()


def _open_brain():
    from servers.brain import Brain
    db_path = os.path.join(
        os.environ.get('BRAIN_DB_DIR') or
        os.path.expanduser('~/AgentsContext/brain'),
        'brain.db')
    return Brain.get_instance(db_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Backfill at most N rows (default: all NULL)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute embeddings but skip the UPDATE')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='fastembed batch size (default 256)')
    args = parser.parse_args()

    brain = _open_brain()

    # Count NULL rows
    n_null = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations '
        'WHERE embedding IS NULL AND archived = 0'
    ).fetchone()[0]
    n_total = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations WHERE archived = 0'
    ).fetchone()[0]
    print(f'edge_relations: {n_total} active, {n_null} need backfill')
    if n_null == 0:
        print('Nothing to do.')
        return

    # Distinct edges needing backfill. The compose→embed→write is DELEGATED to
    # Brain.backfill_edge_embeddings — the single source of truth (it owns the
    # excluded-relation filter, stale-model re-embed, embedder-not-ready guard,
    # and the concurrency-safe description-guarded write). This script is just
    # the bulk driver: gather ids, chunk, report. Keeping the logic here would
    # drift from the runtime path (the bug this dedup closes).
    sql = ('SELECT DISTINCT edge_id FROM edge_relations '
           'WHERE embedding IS NULL AND archived = 0')
    if args.limit:
        sql += f' LIMIT {int(args.limit)}'
    edge_ids = [r[0] for r in brain.conn.execute(sql).fetchall()]
    print(f'Loaded {len(edge_ids)} edges to backfill')

    if args.dry_run:
        print('  [DRY RUN — Brain.backfill_edge_embeddings IS the write path; '
              'nothing written]')
        return

    t_start = time.monotonic()
    chunk_size = max(args.batch_size, 1)
    n_done = 0
    for i in range(0, len(edge_ids), chunk_size):
        chunk = edge_ids[i:i + chunk_size]
        n_done += brain.backfill_edge_embeddings(chunk)
        seen = min(i + chunk_size, len(edge_ids))
        elapsed = time.monotonic() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f'  {seen}/{len(edge_ids)} edges  '
              f'({n_done} relations embedded, {rate:.0f}/s)')

    elapsed = time.monotonic() - t_start
    print(f'\nDone. Embedded {n_done} relations in {elapsed:.1f}s')
    n_remaining = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations '
        'WHERE embedding IS NULL AND archived = 0').fetchone()[0]
    print(f'  Remaining NULL rows: {n_remaining}')


if __name__ == '__main__':
    main()

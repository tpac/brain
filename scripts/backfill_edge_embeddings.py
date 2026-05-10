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
    from servers import embedder

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

    # Pull rows needing backfill
    sql = ('SELECT edge_id, relation, COALESCE(description, "") '
           'FROM edge_relations WHERE embedding IS NULL AND archived = 0')
    if args.limit:
        sql += f' LIMIT {int(args.limit)}'
    rows = brain.conn.execute(sql).fetchall()
    print(f'Loaded {len(rows)} rows to embed')

    # Compose texts
    texts = []
    for edge_id, relation, description in rows:
        text = brain.aspects.compose_edge_text(relation, description)
        texts.append(text)
    nonempty = sum(1 for t in texts if t)
    print(f'Composed {nonempty} non-empty enriched texts '
          f'({len(rows) - nonempty} were empty — skipped)')

    # Embed in batches
    model_name = embedder.stats.get('model_name') or ''
    t_start = time.monotonic()
    n_done = 0
    batch_size = args.batch_size
    transaction_size = 500
    pending_updates = []

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Skip empty texts but keep alignment with rows
        nonempty_pairs = [(i, t) for i, t in enumerate(batch_texts) if t]
        if nonempty_pairs:
            indices, only_texts = zip(*nonempty_pairs)
            blobs = embedder.embed_batch(list(only_texts), kind='document')
        else:
            indices, blobs = (), ()

        # Build UPDATE list, keeping indices straight
        for i, blob in zip(indices, blobs):
            edge_id, relation, _desc = batch[i]
            if blob:
                pending_updates.append((blob, model_name, edge_id, relation))

        n_done += len(batch)
        elapsed = time.monotonic() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (len(rows) - n_done) / rate if rate > 0 else 0
        print(f'  {n_done}/{len(rows)} embedded  '
              f'({rate:.0f}/s, ~{remaining:.0f}s remaining)')

        # Flush in transactions of 500
        while len(pending_updates) >= transaction_size:
            chunk = pending_updates[:transaction_size]
            pending_updates = pending_updates[transaction_size:]
            if not args.dry_run:
                with brain.write_lock:
                    brain.conn.executemany(
                        'UPDATE edge_relations '
                        'SET embedding = ?, embedding_model = ? '
                        'WHERE edge_id = ? AND relation = ?', chunk)
                    brain.conn.commit()

    # Flush remainder
    if pending_updates and not args.dry_run:
        with brain.write_lock:
            brain.conn.executemany(
                'UPDATE edge_relations '
                'SET embedding = ?, embedding_model = ? '
                'WHERE edge_id = ? AND relation = ?', pending_updates)
            brain.conn.commit()

    elapsed = time.monotonic() - t_start
    print(f'\nDone. Embedded {n_done} rows in {elapsed:.1f}s '
          f'({n_done/elapsed:.0f}/s)')
    if args.dry_run:
        print('  [DRY RUN — no rows were written]')
    else:
        n_remaining = brain.conn.execute(
            'SELECT COUNT(*) FROM edge_relations '
            'WHERE embedding IS NULL AND archived = 0').fetchone()[0]
        print(f'  Remaining NULL rows: {n_remaining}')


if __name__ == '__main__':
    main()

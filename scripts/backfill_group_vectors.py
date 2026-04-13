#!/usr/bin/env python3
"""Backfill group vectors (title, high_meta, other_meta) for nodes missing them.

920 pre-April nodes were created before _compute_group_vectors() existed.
They have content embeddings but no group vectors, disadvantaging them in
recall's z-weighted scoring (STEP 3.5).

Also cleans orphaned enrichment rows for archived/deleted nodes.

Usage:
    python3 scripts/backfill_group_vectors.py --check     # Show current gaps
    python3 scripts/backfill_group_vectors.py              # Backfill + clean
    python3 scripts/backfill_group_vectors.py --dry-run    # Count only, no writes
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def check_coverage(brain):
    """Report group vector coverage."""
    from servers.health_check import check_group_vector_coverage
    result = check_group_vector_coverage(brain)

    print(f"Group vector coverage:")
    print(f"  Active non-community nodes: {result['total']}")
    print(f"  Title vectors:  {result['coverage']['title_count']}/{result['total']} "
          f"({result['coverage']['title_pct']:.0f}%)")
    print(f"  High meta:      {result['coverage']['high_meta_count']}/{result['total']} "
          f"({result['coverage']['high_meta_pct']:.0f}%)")
    print(f"  Other meta:     {result['coverage']['other_meta_count']}/{result['total']} "
          f"({result['coverage']['other_meta_pct']:.0f}%)")
    print(f"  Orphaned rows:  {result['orphaned']}")
    print(f"  OK: {result['ok']}")

    if result['gaps']:
        print(f"\n  Gaps:")
        for gap in result['gaps']:
            print(f"    - {gap}")

    return result


def backfill(brain, dry_run=False):
    """Compute group vectors for all nodes missing title vectors."""
    from servers.dal_metadata import MetadataDAL

    # Find nodes missing title vector
    missing = brain.conn.execute(
        "SELECT n.id, n.title, n.content "
        "FROM nodes n "
        "JOIN node_embeddings ne ON ne.node_id = n.id "
        "WHERE n.archived = 0 AND n.type != 'community' "
        "AND n.id NOT IN ("
        "  SELECT node_id FROM node_enrichments WHERE vector_type = 'title'"
        ")"
    ).fetchall()

    print(f"\nNodes missing title vectors: {len(missing)}")

    if dry_run:
        print("(dry run — no writes)")
        return 0

    if not missing:
        return 0

    mdal = MetadataDAL(brain.conn)
    computed = 0
    errors = 0
    t0 = time.time()

    for i, (node_id, title, content) in enumerate(missing):
        try:
            # Load metadata for this node
            meta = mdal.get(node_id)
            situation_row = brain.conn.execute(
                "SELECT situation_text FROM node_embeddings WHERE node_id = ?",
                (node_id,)
            ).fetchone()
            situation = situation_row[0] if situation_row and situation_row[0] else None

            # Call the existing function
            brain._compute_group_vectors(
                node_id, title, content or '',
                situation=situation,
                reasoning=meta.get('reasoning', ''),
                user_raw_quote=meta.get('user_raw_quote', ''),
                anchor_raw_quote=meta.get('anchor_raw_quote', ''),
                correction_pattern=meta.get('correction_pattern', ''),
                source_context=meta.get('source_context', ''),
            )
            computed += 1

            # Commit every 50 nodes
            if (i + 1) % 50 == 0:
                brain.conn.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i + 1}/{len(missing)} ({rate:.0f} nodes/s)")

        except Exception as e:
            errors += 1
            print(f"  ERROR on {node_id[:8]}: {e}")

    brain.conn.commit()
    elapsed = time.time() - t0
    print(f"\nComputed group vectors for {computed} nodes in {elapsed:.1f}s ({errors} errors)")
    return computed


def clean_orphans(brain, dry_run=False):
    """Delete enrichment rows for archived or non-existent nodes."""
    count = brain.conn.execute(
        "SELECT COUNT(*) FROM node_enrichments "
        "WHERE node_id NOT IN (SELECT id FROM nodes WHERE archived = 0)"
    ).fetchone()[0]

    print(f"\nOrphaned enrichment rows: {count}")

    if dry_run or count == 0:
        return count

    brain.conn.execute(
        "DELETE FROM node_enrichments "
        "WHERE node_id NOT IN (SELECT id FROM nodes WHERE archived = 0)"
    )
    brain.conn.commit()
    print(f"Deleted {count} orphaned rows")
    return count


def main():
    parser = argparse.ArgumentParser(description='Backfill group vectors')
    parser.add_argument('--check', action='store_true', help='Show coverage only')
    parser.add_argument('--dry-run', action='store_true', help='Count only, no writes')
    args = parser.parse_args()

    from servers.brain import Brain

    db_dir = os.environ.get('BRAIN_DB_DIR',
                            os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain'))
    db_path = os.path.join(db_dir, 'brain.db')

    if not os.path.exists(db_path):
        print(f"ERROR: brain.db not found at {db_path}")
        sys.exit(1)

    # Load .env
    env_path = os.path.join(ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    brain = Brain(db_path)
    try:
        if args.check:
            check_coverage(brain)
            return

        print("BEFORE:")
        check_coverage(brain)

        backfill(brain, dry_run=args.dry_run)
        clean_orphans(brain, dry_run=args.dry_run)

        print("\nAFTER:")
        check_coverage(brain)
    finally:
        brain.save()
        brain.close()


if __name__ == '__main__':
    main()

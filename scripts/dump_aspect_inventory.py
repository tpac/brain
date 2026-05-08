#!/usr/bin/env python3
"""Dump every distinct node type + edge relation with counts and 3 example records.

Direct sqlite queries — no Brain instance, no daemon. Useful for:
  - Building eval/aspects_ground_truth.json (look at every string + examples,
    decide where it belongs).
  - Diagnostics — see what's actually in the brain's vocabulary.

Output: eval/aspect_inventory.json (overwritable). Mirrors the shape the
decoder builds, so visual review here matches what the encoder will see.

Usage:
    ./dev python3 scripts/dump_aspect_inventory.py
    ./dev python3 scripts/dump_aspect_inventory.py --db /path/to/brain.db
    ./dev python3 scripts/dump_aspect_inventory.py --min-count 1
"""

import argparse
import json
import os
import sqlite3
import sys


def default_db_path():
    env = os.environ.get('BRAIN_DB_DIR', '')
    if env and os.path.exists(os.path.join(env, 'brain.db')):
        return os.path.join(env, 'brain.db')
    home_default = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain', 'brain.db')
    if os.path.exists(home_default):
        return home_default
    return None


def pick_diverse(rows, n):
    if not rows:
        return []
    if len(rows) <= n:
        tiers = ['strong', 'typical', 'edge'][:len(rows)]
        return list(zip(tiers, rows))
    if n == 3:
        return [
            ('strong', rows[0]),
            ('typical', rows[len(rows) // 2]),
            ('edge', rows[-1]),
        ]
    step = (len(rows) - 1) / (n - 1)
    indices = [int(round(i * step)) for i in range(n)]
    tiers = ['strong'] + ['typical'] * (n - 2) + ['edge']
    return list(zip(tiers, [rows[i] for i in indices]))


def dump_node_types(conn, min_count, examples_per):
    distinct = conn.execute("""
        SELECT type, COUNT(*) AS count
        FROM nodes
        WHERE archived = 0 AND type IS NOT NULL AND type != ''
        GROUP BY type
        ORDER BY count DESC
    """).fetchall()

    out = []
    for row in distinct:
        if row['count'] < min_count:
            continue
        ex_rows = conn.execute("""
            SELECT n.type, n.title, n.content, n.confidence, n.access_count,
                   (SELECT value FROM node_metadata_kv
                    WHERE node_id = n.id AND key = 'situation' LIMIT 1) AS situation
            FROM nodes n
            WHERE n.archived = 0 AND n.type = ?
            ORDER BY n.access_count DESC, LENGTH(n.content) DESC
        """, (row['type'],)).fetchall()
        examples = [
            {
                'tier': tier,
                'type': r['type'] or '',
                'title': r['title'] or '',
                'content_snippet': (r['content'] or '')[:400],
                'situation': (r['situation'] or '')[:300],
                'access_count': r['access_count'],
                'confidence': r['confidence'],
            }
            for tier, r in pick_diverse(ex_rows, examples_per)
        ]
        out.append({
            'value': row['type'],
            'count': row['count'],
            'examples': examples,
        })
    return out


def dump_edge_relations(conn, min_count, examples_per):
    distinct = conn.execute("""
        SELECT relation, COUNT(*) AS count
        FROM edge_relations
        WHERE archived = 0 AND relation IS NOT NULL AND relation != ''
        GROUP BY relation
        ORDER BY count DESC
    """).fetchall()

    out = []
    for row in distinct:
        if row['count'] < min_count:
            continue
        ex_rows = conn.execute("""
            SELECT er.description, er.weight,
                   src.title AS src_title, src.type AS src_type,
                   src.content AS src_content,
                   tgt.title AS tgt_title, tgt.type AS tgt_type,
                   tgt.content AS tgt_content
            FROM edge_relations er
            JOIN edges e ON er.edge_id = e.edge_id
            JOIN nodes src ON e.source_id = src.id
            JOIN nodes tgt ON e.target_id = tgt.id
            WHERE er.archived = 0 AND er.relation = ?
              AND src.archived = 0 AND tgt.archived = 0
            ORDER BY er.weight DESC, LENGTH(er.description) DESC
        """, (row['relation'],)).fetchall()
        examples = [
            {
                'tier': tier,
                'src_title': r['src_title'] or '',
                'src_type': r['src_type'] or '',
                'src_content_snippet': (r['src_content'] or '')[:150],
                'tgt_title': r['tgt_title'] or '',
                'tgt_type': r['tgt_type'] or '',
                'tgt_content_snippet': (r['tgt_content'] or '')[:150],
                'description': (r['description'] or '')[:300],
                'weight': r['weight'],
            }
            for tier, r in pick_diverse(ex_rows, examples_per)
        ]
        out.append({
            'value': row['relation'],
            'count': row['count'],
            'examples': examples,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument('--db', default=default_db_path(),
                        help='Path to brain.db (default: ~/AgentsContext/brain/brain.db)')
    parser.add_argument('--min-count', type=int, default=1,
                        help='Filter to types/relations with count >= N (default: 1)')
    parser.add_argument('--examples', type=int, default=3,
                        help='Number of example records per string (default: 3)')
    parser.add_argument('--output', default=None,
                        help='Output path (default: eval/aspect_inventory.json)')
    args = parser.parse_args()

    if not args.db or not os.path.exists(args.db):
        print('ERROR: brain.db not found. Pass --db or set BRAIN_DB_DIR.', file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output = os.path.join(repo_root, 'eval', 'aspect_inventory.json')

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print('reading from %s' % args.db)
    node_types = dump_node_types(conn, args.min_count, args.examples)
    edge_relations = dump_edge_relations(conn, args.min_count, args.examples)
    conn.close()

    inventory = {
        'min_count': args.min_count,
        'examples_per_string': args.examples,
        'node_types': node_types,
        'edge_relations': edge_relations,
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(inventory, f, indent=2)
        f.write('\n')

    print('wrote %s' % output)
    print('  node_types:     %d distinct (counts: %d total uses)' % (
        len(node_types), sum(r['count'] for r in node_types)))
    print('  edge_relations: %d distinct (counts: %d total uses)' % (
        len(edge_relations), sum(r['count'] for r in edge_relations)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

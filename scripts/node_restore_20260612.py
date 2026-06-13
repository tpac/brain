#!/usr/bin/env python3
"""Restore the ~3,896 nodes silently dropped on 2026-06-12 13:02Z.

Incident: a March-era code snapshot ran its schema rebuild against the live
brain.db — `INSERT OR IGNORE` into a table with March's closed node-type
CHECK constraint silently dropped every node with a post-March type, then
DROP/RENAME committed it. Proven: zero type overlap between survivors and
dead (see incident node 182cece5).

What this script does (additive only — never deletes, never updates rows
that exist):
  1. NODES   — insert rows present in the source backup but missing from the
               target, ALL types, shared-columns mapping. Loud count checks.
  2. VECTORS — copy node_vectors + node_enrichments rows for restored ids
               (same embedder model — blobs are valid; no re-embed needed).
  3. REFS    — copy node_source_refs rows for restored ids.
  4. EDGES   — rebuild deleted parent `edges` rows for the orphaned
               edge_relations (backup match, fingerprint-verified). The er
               and node_metadata_kv rows survived in the live db and
               reconnect by themselves.
  5. FTS     — rebuild the nodes FTS index.
  6. VERIFY  — counts, orphan residue, integrity_check.

Tier-3 archive of still-orphaned relations is NOT done here — run
scripts/orphan_edge_recovery.py afterwards for the residue report.

June 5→12 created-then-killed nodes are NOT here either (in no backup) —
they need the trace-replay pass (separate, best-effort).

SAFETY: dry-run by default. --apply on the LIVE db requires the maintenance
lock (daemon stopped/quiesced). Rehearse on a copy first.

USE
---
  ./dev python3 scripts/node_restore_20260612.py --db /tmp/rehearsal-brain.db
  ./dev python3 scripts/node_restore_20260612.py --db /tmp/rehearsal-brain.db --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

SOURCE_BACKUP = os.path.expanduser(
    '~/AgentsContext/brain/brain.db.bak-20260605-162833')


def edge_hash(s, t):
    return 'edg_' + hashlib.md5((s + ':' + t).encode()).hexdigest()[:8]


def open_ro(path):
    return sqlite3.connect('file:%s?mode=ro&immutable=1' % path, uri=True)


def table_cols(conn, table):
    return [r[1] for r in conn.execute('PRAGMA table_info(%s)' % table)]


def copy_missing_rows(src, dst, table, key_col, restored_ids, apply,
                      id_filter=None):
    """Copy rows from src whose key is in restored_ids and absent in dst.
    Shared-column mapping. Returns rows copied (or would-copy in dry run)."""
    s_cols = table_cols(src, table)
    d_cols = set(table_cols(dst, table))
    shared = [c for c in s_cols if c in d_cols]
    col_list = ', '.join(shared)
    existing = {r[0] for r in dst.execute(
        'SELECT %s FROM %s' % (key_col, table))}
    n = 0
    for row in src.execute('SELECT %s, %s FROM %s' % (key_col, col_list, table)):
        key, vals = row[0], row[1:]
        if key not in restored_ids or (key, ) and key in existing:
            if key not in restored_ids:
                continue
        if id_filter and not id_filter(key):
            continue
        # key may repeat (multi-row tables like vectors/refs) — `existing`
        # check applies only to single-row-per-key tables; for multi-row
        # tables the caller pre-deletes nothing and we rely on INSERT OR
        # IGNORE + table PKs.
        if apply:
            dst.execute(
                'INSERT OR IGNORE INTO %s (%s) VALUES (%s)' % (
                    table, col_list, ','.join('?' * len(shared))), vals)
            n += dst.execute('SELECT changes()').fetchone()[0]
        else:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--source', default=SOURCE_BACKUP)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--report', default='/tmp/node_restore_report.json')
    args = ap.parse_args()

    live_db = os.path.realpath(os.path.expanduser('~/AgentsContext/brain/brain.db'))
    if args.apply and os.path.realpath(args.db) == live_db:
        lock = '/tmp/brain-maintenance-%d.lock' % os.getuid()
        if not os.path.exists(lock):
            sys.exit('REFUSED: --apply on LIVE requires maintenance lock %s' % lock)

    t0 = time.time()
    src = open_ro(args.source)
    dst = sqlite3.connect(args.db) if args.apply else open_ro(args.db)
    report = {'source': args.source, 'target': args.db, 'apply': args.apply}

    # ── 1. NODES ──
    cur_ids = {r[0] for r in dst.execute('SELECT id FROM nodes')}
    src_nodes = {r[0] for r in src.execute('SELECT id FROM nodes')}
    to_restore = src_nodes - cur_ids
    report['nodes_missing_from_target'] = len(to_restore)
    print('nodes in source not in target: %d' % len(to_restore))

    s_cols = table_cols(src, 'nodes')
    d_cols = set(table_cols(dst, 'nodes'))
    shared = [c for c in s_cols if c in d_cols]
    dropped_cols = [c for c in s_cols if c not in d_cols]
    print('node columns: %d shared, source-only dropped: %s' % (
        len(shared), dropped_cols or 'none'))

    n_nodes = 0
    if args.apply:
        dst.execute('BEGIN IMMEDIATE')
        ph = ','.join('?' * len(shared))
        col_list = ', '.join(shared)
        for row in src.execute(
                'SELECT %s FROM nodes WHERE id IN (%s)' % (
                    col_list, ','.join('?' * len(to_restore))),
                list(to_restore)):
            dst.execute('INSERT INTO nodes (%s) VALUES (%s)' % (col_list, ph), row)
            n_nodes += 1
    else:
        n_nodes = len(to_restore)
    report['nodes_restored'] = n_nodes
    print('nodes restored: %d' % n_nodes)

    # ── 2+3. VECTORS / ENRICHMENTS / SOURCE REFS ──
    for table, key in [('node_vectors', 'node_id'),
                       ('node_enrichments', 'node_id'),
                       ('node_source_refs', 'node_id')]:
        try:
            n = 0
            s_cols_t = table_cols(src, table)
            d_cols_t = set(table_cols(dst, table))
            sh = [c for c in s_cols_t if c in d_cols_t]
            if not sh:
                print('%s: no shared columns, skipped' % table)
                continue
            col_list = ', '.join(sh)
            ph = ','.join('?' * len(sh))
            for row in src.execute(
                    'SELECT %s FROM %s WHERE %s IN (%s)' % (
                        col_list, table, key, ','.join('?' * len(to_restore))),
                    list(to_restore)):
                if args.apply:
                    dst.execute('INSERT OR IGNORE INTO %s (%s) VALUES (%s)'
                                % (table, col_list, ph), row)
                n += 1
            report[table] = n
            print('%s rows: %d' % (table, n))
        except sqlite3.Error as e:
            print('%s: %s (skipped)' % (table, e))
            report[table] = 'skipped: %s' % e

    # ── 4. EDGES for orphaned relations (backup match + hash gate) ──
    orphans = {r[0] for r in dst.execute(
        "SELECT DISTINCT er.edge_id FROM edge_relations er "
        "LEFT JOIN edges e ON er.edge_id = e.edge_id "
        "WHERE e.edge_id IS NULL AND er.archived = 0")}
    node_set = cur_ids | to_restore
    edge_cols = ('edge_id', 'source_id', 'target_id', 'weight',
                 'co_access_count', 'last_strengthened', 'created_at')
    n_edges, skipped_fp, skipped_endpoint = 0, 0, 0
    for row in src.execute('SELECT %s FROM edges' % ', '.join(edge_cols)):
        eid, s, t = row[0], row[1], row[2]
        if eid not in orphans:
            continue
        if edge_hash(s or '', t or '') != eid:
            skipped_fp += 1
            continue
        if s not in node_set or t not in node_set:
            skipped_endpoint += 1
            continue
        if args.apply:
            dst.execute(
                'INSERT OR IGNORE INTO edges (%s) VALUES (%s)' % (
                    ', '.join(edge_cols), ','.join('?' * len(edge_cols))), row)
        n_edges += 1
    report['edges_restored'] = n_edges
    report['edges_skipped_fingerprint'] = skipped_fp
    report['edges_skipped_dead_endpoint'] = skipped_endpoint
    print('edges restored: %d (fingerprint-fail %d, dead-endpoint %d)' % (
        n_edges, skipped_fp, skipped_endpoint))

    if args.apply:
        dst.commit()
        # ── 5. FTS rebuild ──
        fts = [r[0] for r in dst.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE '%fts%' AND sql LIKE '%fts5%'")]
        for t in fts:
            try:
                dst.execute("INSERT INTO %s(%s) VALUES('rebuild')" % (t, t))
                print('FTS rebuilt: %s' % t)
            except sqlite3.Error as e:
                print('FTS %s: %s' % (t, e))
        dst.commit()

        # ── 6. VERIFY ──
        total, active = dst.execute(
            "SELECT COUNT(*), SUM(CASE WHEN archived=0 THEN 1 ELSE 0 END) "
            "FROM nodes").fetchone()
        left = dst.execute(
            "SELECT COUNT(DISTINCT er.edge_id) FROM edge_relations er "
            "LEFT JOIN edges e ON er.edge_id = e.edge_id "
            "WHERE e.edge_id IS NULL AND er.archived = 0").fetchone()[0]
        ic = dst.execute('PRAGMA integrity_check(5)').fetchone()[0]
        report.update({'post_total': total, 'post_active': active,
                       'orphan_edge_ids_remaining': left,
                       'integrity_check': ic})
        print('POST: nodes=%d active=%d orphan_edge_ids_left=%d integrity=%s'
              % (total, active, left, ic))

    report['elapsed_s'] = round(time.time() - t0, 1)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print('report: %s (%.1fs)%s' % (args.report, report['elapsed_s'],
                                    '' if args.apply else '  DRY RUN'))


if __name__ == '__main__':
    main()

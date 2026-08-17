#!/usr/bin/env python3
"""Orphan edge_relations recovery — rebuild deleted parent `edges` rows.

Context (2026-06-12 graph-integrity incident): dal.py's orphan cleanup
deleted `edges` rows without their `edge_relations` rows for ~2 months,
stranding 16,889 relation rows (84% of active) with no parent — invisible
to every JOIN-based read. The leak is fixed; this script recovers the
stranded rows by rebuilding their parent edges.

Recovery tiers (evidence quality order):
  1. BACKUP MATCH — scan dated brain.db backups for the deleted edges rows
     (exact source/target/weight/created_at). New-schema backups match by
     edge_id; pre-v22 backups match by recomputing edge_id from their
     (source, target) pairs.
  2. HASH ENUMERATION — edge_id is deterministic: 'edg_' + md5(src:tgt)[:8].
     Enumerate ordered pairs of nodes in the target db and match remaining
     orphans. Ambiguous ids (hash collision: >1 candidate pair) are flagged,
     never guessed.
  3. RESIDUE — orphans with no backup row and no enumerable pair (endpoint
     hard-deleted everywhere) are ARCHIVED (er.archived=1), never deleted.

Every recovered edge must pass the fingerprint check
(edg_ + md5(source:target)[:8] == edge_id) regardless of tier, and both
endpoints must exist in the target's nodes table (else residue).

SAFETY: dry-run by default; --apply writes. Applying to the LIVE db
requires the maintenance lock (/tmp/brain-maintenance-{uid}.lock) so the
daemon is stopped/quiesced first. Rehearse on a copy.

USE
---
  # rehearsal (dry-run plan)
  ./dev python3 scripts/orphan_edge_recovery.py --db /tmp/rehearsal.db
  # rehearsal (apply to the copy)
  ./dev python3 scripts/orphan_edge_recovery.py --db /tmp/rehearsal.db --apply --archive-residue
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_BACKUP_GLOBS = [
    os.path.expanduser('~/AgentsContext/brain/brain.db.bak*'),
    os.path.expanduser('~/AgentsContext/brain/brain.db.backup*'),
    os.path.expanduser('~/AgentsContext/brain/brain.db.v*.bak'),
    os.path.expanduser('~/AgentsContext/brain/brain.db.pre-*'),
    # Central-mechanism artifacts: tagged pre-destructive backups beside the
    # DB (raw .bak from boot-path callers, .bak.gz from scripts) and the
    # rolling GFS snapshots (.gz, materialized on open).
    os.path.expanduser('~/AgentsContext/brain/brain.db.*.bak'),
    os.path.expanduser('~/AgentsContext/brain/brain.db.*.bak.gz'),
    os.path.expanduser('~/AgentsContext/brain/backups/brain.db.*.gz'),
]
EDGE_COLS = ('edge_id', 'source_id', 'target_id', 'weight',
             'co_access_count', 'last_strengthened', 'created_at')


def edge_hash(source_id: str, target_id: str) -> str:
    return 'edg_' + hashlib.md5(
        (source_id + ':' + target_id).encode()).hexdigest()[:8]


def open_ro(path: str) -> sqlite3.Connection:
    # immutable=1: never touches -wal/-shm siblings. Backups with non-empty
    # WALs lose their last uncheckpointed frames — acceptable for recovery.
    # Gzipped backups (central mechanism) are decompressed to a sibling
    # .materialized file first; harmless residue, reused across runs.
    if path.endswith('.gz'):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from servers.db_backup import materialize_backup
        path = materialize_backup(path)
    return sqlite3.connect('file:%s?mode=ro&immutable=1' % path, uri=True)


def load_target_state(db_path: str):
    conn = open_ro(db_path)
    orphans = {}
    for eid, relation, weight, created in conn.execute(
            "SELECT er.edge_id, er.relation, er.weight, er.created_at "
            "FROM edge_relations er LEFT JOIN edges e ON er.edge_id = e.edge_id "
            "WHERE e.edge_id IS NULL AND er.archived = 0"):
        o = orphans.setdefault(eid, {'relations': [], 'min_created': created,
                                     'max_weight': weight or 0.5})
        o['relations'].append(relation)
        if created and (o['min_created'] is None or created < o['min_created']):
            o['min_created'] = created
        o['max_weight'] = max(o['max_weight'], weight or 0.5)
    node_ids = [r[0] for r in conn.execute("SELECT id FROM nodes")]
    conn.close()
    return orphans, node_ids


def scan_backup(path: str, orphan_ids: set):
    """Return {edge_id: row-dict} for orphan ids found in this backup."""
    found = {}
    try:
        conn = open_ro(path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)")}
        if not cols:
            conn.close()
            return found
        has_eid = 'edge_id' in cols
        sel = ('edge_id, ' if has_eid else '') + \
            'source_id, target_id, weight, co_access_count, last_strengthened, created_at'
        for row in conn.execute('SELECT %s FROM edges' % sel):
            if has_eid:
                eid, src, tgt, w, cac, ls, created = row
                if eid is None:
                    continue
            else:
                src, tgt, w, cac, ls, created = row
                eid = edge_hash(src or '', tgt or '')
            if eid in orphan_ids and eid not in found:
                # fingerprint check — a backup row that fails it is corrupt
                if edge_hash(src or '', tgt or '') != eid:
                    continue
                found[eid] = dict(zip(EDGE_COLS, (eid, src, tgt, w, cac, ls, created)))
        conn.close()
    except sqlite3.Error as e:
        print('  [skip] %s: %s' % (os.path.basename(path), e))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True, help='target brain.db (copy for rehearsal)')
    ap.add_argument('--backups', nargs='*', help='extra backup db paths')
    ap.add_argument('--apply', action='store_true', help='write (default: dry-run)')
    ap.add_argument('--archive-residue', action='store_true',
                    help='with --apply: archive tier-3 orphan relations')
    ap.add_argument('--report', default='/tmp/orphan_recovery_report.json')
    args = ap.parse_args()

    live_db = os.path.expanduser('~/AgentsContext/brain/brain.db')
    target = os.path.realpath(args.db)
    if args.apply and target == os.path.realpath(live_db):
        lock = '/tmp/brain-maintenance-%d.lock' % os.getuid()
        if not os.path.exists(lock):
            sys.exit('REFUSED: --apply on the LIVE db requires the maintenance '
                     'lock (%s) — stop/quiesce the daemon first.' % lock)

    t0 = time.time()
    orphans, node_ids = load_target_state(args.db)
    orphan_ids = set(orphans)
    node_set = set(node_ids)
    print('orphan edge_ids: %d  (relation rows: %d)  nodes: %d' % (
        len(orphan_ids), sum(len(o['relations']) for o in orphans.values()),
        len(node_ids)))

    # ── Tier 1: backups (oldest → newest; newest occurrence wins) ──
    backup_paths = sorted(set(
        p for g in DEFAULT_BACKUP_GLOBS for p in glob.glob(g)
        if not p.endswith(('-shm', '-wal'))) | set(args.backups or []))
    tier1 = {}
    for path in backup_paths:  # sorted = roughly chronological; later overwrites
        hits = scan_backup(path, orphan_ids)
        fresh = sum(1 for k in hits if k not in tier1)
        tier1.update(hits)
        print('  %-55s +%d (tier1 total %d)' % (
            os.path.basename(path)[:55], fresh, len(tier1)))
    # endpoint check — both nodes must still exist in target, else residue
    tier1 = {k: v for k, v in tier1.items()
             if v['source_id'] in node_set and v['target_id'] in node_set}
    print('tier 1 (backup match, endpoints alive): %d' % len(tier1))

    # ── Tier 2: hash enumeration over ordered node pairs ──
    remaining = orphan_ids - set(tier1)
    candidates = defaultdict(list)
    if remaining:
        for src in node_ids:
            prefix = src + ':'
            for tgt in node_ids:
                if src == tgt:
                    continue
                eid = 'edg_' + hashlib.md5(
                    (prefix + tgt).encode()).hexdigest()[:8]
                if eid in remaining:
                    candidates[eid].append((src, tgt))
    tier2, ambiguous = {}, {}
    for eid, pairs in candidates.items():
        if len(pairs) == 1:
            src, tgt = pairs[0]
            o = orphans[eid]
            tier2[eid] = dict(zip(EDGE_COLS, (
                eid, src, tgt, o['max_weight'], 0, None, o['min_created'])))
        else:
            ambiguous[eid] = pairs
    print('tier 2 (hash enumeration): %d   ambiguous (flagged): %d' % (
        len(tier2), len(ambiguous)))

    # ── Tier 3: residue ──
    residue = orphan_ids - set(tier1) - set(tier2) - set(ambiguous)
    print('tier 3 (residue → archive): %d' % len(residue))

    # community_member visibility recovered
    cm_recovered = sum(
        1 for eid in list(tier1) + list(tier2)
        if 'community_member' in orphans[eid]['relations'])
    print('community memberships made visible again: %d' % cm_recovered)

    # ── Final fingerprint gate on the whole plan ──
    plan = {**tier1, **tier2}
    bad = [eid for eid, row in plan.items()
           if edge_hash(row['source_id'], row['target_id']) != eid]
    assert not bad, 'fingerprint check failed for %d rows — aborting' % len(bad)

    report = {
        'target': args.db, 'apply': args.apply,
        'orphan_edge_ids': len(orphan_ids),
        'tier1': len(tier1), 'tier2': len(tier2),
        'ambiguous': {k: v for k, v in ambiguous.items()},
        'residue': len(residue),
        'community_member_recovered': cm_recovered,
        'backups_scanned': len(backup_paths),
        'elapsed_s': round(time.time() - t0, 1),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print('report: %s  (%.1fs)' % (args.report, report['elapsed_s']))

    if not args.apply:
        print('DRY RUN — nothing written.')
        return

    conn = sqlite3.connect(args.db)
    conn.execute('BEGIN IMMEDIATE')
    for row in plan.values():
        conn.execute(
            'INSERT OR IGNORE INTO edges (%s) VALUES (?,?,?,?,?,?,?)' %
            ', '.join(EDGE_COLS),
            tuple(row[c] for c in EDGE_COLS))
    n_archived = 0
    if args.archive_residue and residue:
        ph = ','.join('?' * len(residue))
        cur = conn.execute(
            'UPDATE edge_relations SET archived = 1 '
            'WHERE edge_id IN (%s) AND archived = 0' % ph, list(residue))
        n_archived = cur.rowcount
    conn.commit()
    # post-verify
    left = conn.execute(
        "SELECT COUNT(DISTINCT er.edge_id) FROM edge_relations er "
        "LEFT JOIN edges e ON er.edge_id = e.edge_id "
        "WHERE e.edge_id IS NULL AND er.archived = 0").fetchone()[0]
    conn.close()
    print('APPLIED: %d edges inserted, %d residue relations archived' % (
        len(plan), n_archived))
    print('orphan edge_ids remaining: %d (expect ambiguous count: %d)' % (
        left, len(ambiguous)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Pass 2 + Pass 3 of the June-12 recovery — reconstruct killed week-nodes
from trace content, then forward-apply revisions to true 13:02 state.

NOT command replay. Both passes are pure data:
  PASS 2 (reconstruct): for nodes created June 5→12 then killed (absent from
    the June-5 backup, so the node-restore couldn't cover them), pull the
    full node spec from encoding_run action_details `input` (remember_batch
    nodes[] / brain_batch remember-ops[]), map to the original id via the
    `created[]` array (positional, remember-ops only), and INSERT the row
    with its ORIGINAL id. Surviving kv (question/situation/reasoning) and
    edges reconnect for free. Each (id,title) is cross-checked against the
    encoding journal's `- <id> "<title>"` line; a mismatch is SKIPPED, never
    inserted (guards against created[]↔ops[] misalignment).
  PASS 3 (forward-apply revisions): for every killed+restored node with
    node_revised delta traces in the window, take the latest `new` value per
    real field and UPDATE the row to its 13:02 state. Fixes nodes restored
    at stale (June-5) content. Churn/control fields are ignored
    (skip_embedding, community_size/maturity/internal_fraction/is_corridor).

SAFETY: dry-run by default; --apply on LIVE needs the maintenance lock
(daemon stopped). Additive — only inserts ids ABSENT from the target, only
updates fields with a real delta. Rehearse on a copy.

USE
---
  ./dev python3 scripts/trace_node_reconstruct_20260612.py --db /tmp/copy.db
  ./dev python3 scripts/trace_node_reconstruct_20260612.py --db <live> --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

LOGS = os.path.expanduser(
    '~/AgentsContext/brain/brain_logs.db.bak-20260612-orphan-audit')
WINDOW_START = '2026-06-05T18:20'
KILL_TS = '2026-06-12T13:02'

# nodes-table structural columns we can set from a remember input
NODE_COLS = {'type', 'title', 'content', 'confidence', 'emotion',
             'emotion_label', 'project', 'personal', 'personal_context',
             'critical', 'evolution_status', 'encoding_source',
             'source_turn_id'}
# promoted metadata-kv fields a remember input may carry
KV_FIELDS = {'situation', 'reasoning', 'user_raw_quote', 'anchor_raw_quote',
             'correction_pattern', 'source_context', 'question'}
# revise delta fields that are real knowledge (forward-apply); the rest is churn
REVISE_REAL = NODE_COLS | KV_FIELDS
REVISE_IGNORE = {'skip_embedding', 'community_size', 'community_maturity',
                 'community_internal_fraction', 'community_is_corridor',
                 'community_dominant_type'}


def open_ro(p):
    return sqlite3.connect('file:%s?mode=ro&immutable=1' % p, uri=True)


def parse_journal_titles(logs):
    """id -> title from every encoding journal ENCODED line in the window."""
    titles = {}
    for (md,) in logs.execute(
            "SELECT metadata FROM trace_events WHERE ref_type='encoding_run' "
            "AND created_at > ?", (WINDOW_START,)):
        d = json.loads(md or '{}')
        for m in re.finditer(r'-\s+([0-9a-f]{8})\s+"([^"]+)"',
                             d.get('journal_entry', '') or ''):
            titles[m.group(1)] = m.group(2)
    return titles


def collect_reconstructions(logs):
    """id -> spec dict, from action_details.input across encoding runs.
    Newest occurrence wins (a node created then re-specified keeps the last)."""
    specs = {}
    rows = logs.execute(
        "SELECT created_at, metadata FROM trace_events WHERE ref_type='encoding_run' "
        "AND created_at > ? ORDER BY created_at", (WINDOW_START,)).fetchall()
    for created_at, md in rows:
        d = json.loads(md or '{}')
        for ad in d.get('action_details', []) or []:
            if not isinstance(ad, dict):
                continue
            tool, inp, created = ad.get('tool'), ad.get('input'), ad.get('created') or []
            if not inp or not created:
                continue
            if tool == 'remember_batch':
                node_specs = inp.get('nodes', [])
            elif tool == 'brain_batch':
                node_specs = [o for o in inp.get('operations', [])
                              if isinstance(o, dict) and o.get('op') == 'remember']
            else:
                continue
            # positional map: created[i] <-> i-th remember spec
            for nid, spec in zip(created, node_specs):
                if isinstance(spec, dict):
                    s = dict(spec)
                    s.setdefault('_trace_created_at', created_at)
                    specs[nid] = s
    return specs


def collect_revisions(logs):
    """id -> {field: latest new value} from node_revised deltas in window."""
    latest = defaultdict(dict)
    rows = logs.execute(
        "SELECT ref_id, created_at, metadata FROM trace_events "
        "WHERE ref_type='node_revised' AND created_at > ? AND created_at < ? "
        "ORDER BY created_at", (WINDOW_START, KILL_TS)).fetchall()
    for ref_id, _ts, md in rows:
        d = json.loads(md or '{}')
        for dlt in d.get('deltas', []) or []:
            if isinstance(dlt, dict) and dlt.get('field') in REVISE_REAL:
                latest[ref_id][dlt['field']] = dlt.get('new')
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--report', default='/tmp/trace_reconstruct_report.json')
    args = ap.parse_args()

    live_db = os.path.realpath(os.path.expanduser('~/AgentsContext/brain/brain.db'))
    if args.apply and os.path.realpath(args.db) == live_db:
        lock = '/tmp/brain-maintenance-%d.lock' % os.getuid()
        if not os.path.exists(lock):
            sys.exit('REFUSED: --apply on LIVE requires maintenance lock %s' % lock)

    logs = open_ro(LOGS)
    dst = sqlite3.connect(args.db) if args.apply else open_ro(args.db)
    cur_ids = {r[0] for r in dst.execute("SELECT id FROM nodes")}
    dst_node_cols = {r[1] for r in dst.execute("PRAGMA table_info(nodes)")}

    journal = parse_journal_titles(logs)
    specs = collect_reconstructions(logs)
    revisions = collect_revisions(logs)

    # ── PASS 2: reconstruct missing nodes ──
    full, thin_skip, title_mismatch, already = [], [], [], 0
    to_insert = {}
    for nid, spec in specs.items():
        if nid in cur_ids:
            already += 1
            continue
        title = spec.get('title')
        # validation gate: reconstructed title must match the journal line
        if nid in journal and title and journal[nid].strip() != title.strip():
            title_mismatch.append((nid, title, journal[nid]))
            continue
        if not (spec.get('type') and title and spec.get('content')):
            thin_skip.append(nid)
            continue
        to_insert[nid] = spec
        full.append(nid)

    # ── PASS 3: which existing/restored nodes have real revisions ──
    fwd = {nid: fields for nid, fields in revisions.items()
           if (nid in cur_ids or nid in to_insert) and fields}
    content_fixes = sum(1 for f in fwd.values() if 'content' in f)

    report = {
        'target': args.db, 'apply': args.apply,
        'pass2_reconstructable_full': len(full),
        'pass2_already_present': already,
        'pass2_thin_skipped_missing_core': len(thin_skip),
        'pass2_title_mismatch_skipped': len(title_mismatch),
        'pass3_nodes_with_revisions': len(fwd),
        'pass3_content_corrections': content_fixes,
        'title_mismatches': title_mismatch[:10],
    }
    print('PASS 2 — reconstruct killed week-nodes:')
    print('  full-content reconstructable: %d' % len(full))
    print('  already present (skip):       %d' % already)
    print('  thin (missing type/title/content), skipped: %d' % len(thin_skip))
    print('  TITLE-MISMATCH guard tripped (skipped):     %d' % len(title_mismatch))
    print('PASS 3 — forward-apply revisions:')
    print('  nodes with real-field revisions: %d (content corrections: %d)'
          % (len(fwd), content_fixes))

    if not args.apply:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
        print('\nDRY RUN — nothing written. report: %s' % args.report)
        return

    dst.execute('BEGIN IMMEDIATE')
    ts_now = None
    # PASS 2 inserts
    n_ins, n_kv = 0, 0
    for nid, spec in to_insert.items():
        cols = {'id': nid, 'encoding_source': spec.get('encoding_source') or 'recovery:trace_reconstruct',
                'created_at': spec.get('_trace_created_at'),
                'updated_at': spec.get('_trace_created_at')}
        for k in NODE_COLS:
            if k in spec and spec[k] is not None and k in dst_node_cols:
                cols[k] = spec[k]
        usecols = [c for c in cols if c in dst_node_cols]
        dst.execute('INSERT OR IGNORE INTO nodes (%s) VALUES (%s)' % (
            ', '.join(usecols), ','.join('?' * len(usecols))),
            [cols[c] for c in usecols])
        n_ins += dst.execute('SELECT changes()').fetchone()[0]
        # backfill kv fields from input (OR IGNORE — surviving rows win)
        for k in KV_FIELDS:
            if spec.get(k):
                dst.execute(
                    "INSERT OR IGNORE INTO node_metadata_kv (node_id, key, value) "
                    "VALUES (?,?,?)", (nid, k, spec[k]))
                n_kv += dst.execute('SELECT changes()').fetchone()[0]
    # PASS 3 forward-apply (only fields that are real node columns; kv via upsert)
    n_upd_node, n_upd_kv = 0, 0
    for nid, fields in fwd.items():
        for f, val in fields.items():
            if val is None:
                continue
            if f in dst_node_cols:
                dst.execute('UPDATE nodes SET %s=? WHERE id=?' % f, (val, nid))
                n_upd_node += 1
            elif f in KV_FIELDS:
                dst.execute(
                    "INSERT INTO node_metadata_kv (node_id,key,value) VALUES (?,?,?) "
                    "ON CONFLICT(node_id,key) DO UPDATE SET value=excluded.value",
                    (nid, f, val))
                n_upd_kv += 1
    dst.commit()
    # FTS rebuild for new content
    for (t,) in dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'"):
        try:
            dst.execute("INSERT INTO %s(%s) VALUES('rebuild')" % (t, t)); dst.commit()
        except sqlite3.Error:
            pass
    report.update({'inserted_nodes': n_ins, 'inserted_kv': n_kv,
                   'pass3_node_field_updates': n_upd_node,
                   'pass3_kv_updates': n_upd_kv,
                   'post_total': dst.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]})
    print('\nAPPLIED: +%d nodes, +%d kv backfilled | pass3: %d node-field + %d kv updates'
          % (n_ins, n_kv, n_upd_node, n_upd_kv))
    print('post total nodes: %d' % report['post_total'])
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One-time backfill: promote the clean archived→live-survivor `_sys_archived_survivor_id`
pointers into first-class `absorbed_into` edges. Goes THROUGH THE DAEMON (send_command,
no second Brain() against live brain.db). Backup taken before running:
brain.db.bak-20260613-absorbedinto.

Clean set only (1-hop, terminal live): archived=1 node whose stamp → archived=0 node.
Excludes the 94 live-with-stale-stamp (a.archived=1 filter), orphans (JOIN), and multi-hop
chains where the survivor is itself archived (s.archived=0 filter) — those wait for the
resolve_live-based pass. Idempotent (connect upsert), so re-running is safe.
Usage: ./dev python3 eval/oracle_audit/backfill_absorbed_into.py [--apply]"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from servers.daemon_client import send_command

APPLY = '--apply' in sys.argv
DESC = ("Absorbed into survivor (backfill from _sys_archived_survivor_id): this archived "
        "node was merged into the live survivor by S2 consolidation; resolution follows "
        "this edge forward to the live node.")
SQL = ("SELECT kv.node_id, kv.value FROM node_metadata_kv kv "
       "JOIN nodes a ON a.id=kv.node_id JOIN nodes s ON s.id=kv.value "
       "WHERE kv.key='_sys_archived_survivor_id' AND a.archived=1 AND s.archived=0")


def unwrap(resp):
    if not isinstance(resp, dict):
        return resp
    for k in ('result', 'value', 'data'):
        if k in resp:
            return resp[k]
    return resp


# 1. read the clean pairs through the daemon
r = send_command('eval', {'code': "brain.conn.execute(%r).fetchall()" % SQL})
pairs = unwrap(r)
if not isinstance(pairs, list):
    print("UNEXPECTED eval response:", r); sys.exit(1)
pairs = [(p[0], p[1]) for p in pairs if p and p[0] and p[1] and p[0] != p[1]]
print("clean archived→live-survivor pairs: %d" % len(pairs))
print("sample:", pairs[:3])

if not APPLY:
    print("\nDRY RUN — re-run with --apply to write the edges."); sys.exit(0)

# 2. write absorbed_into edges via the daemon's connect_batch
conns = [{"source_id": a, "target_id": s, "relation": "absorbed_into",
          "description": DESC, "weight": 1.0} for a, s in pairs]
resp = send_command('connect_batch',
                    {"connections": conns,
                     "encoding_source": "migration:absorbed_into_backfill",
                     "reason": "backfill survivor-redirect edges from _sys_archived_survivor_id (clean 1-hop set)"},
                    timeout=60.0)
print("\nconnect_batch response:", unwrap(resp) if isinstance(resp, dict) else resp)

# 3. verify
v = send_command('eval', {'code': "brain.conn.execute(\"SELECT COUNT(*) FROM edges e JOIN edge_relations er ON er.edge_id=e.edge_id WHERE er.relation='absorbed_into'\").fetchone()[0]"})
print("absorbed_into edges now in graph:", unwrap(v))
